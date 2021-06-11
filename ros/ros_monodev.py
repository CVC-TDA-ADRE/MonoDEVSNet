
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import time
from cv_bridge import CvBridge

import networks
import cv2
import numpy as np
import torch
import yaml
import os


class DevMonoDepth(Node):
    '''
    Depth estimator class for ROS2
    Accepeted arguments:
        topic_in: String value. Message name to subscribe containing the RGB image.
        topic_out: String value. Message name to publish containing the depth image.
        config_file: String value. Configuration file.
        weights_file: String value. Weights file to load the model.
        gpu: Integer value. GPU id to run the model.
    '''

    def __init__(self):
        super().__init__('dev_monodepth_node')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('topic_in', 'image'),
                ('topic_out', 'depth'),
                ('config_file', None),
                ('weights_file', None),
                ('gpu', 0),
            ]
        )

        # get parameters
        topic_in = self.get_parameter('topic_in').get_parameter_value().string_value
        topic_out = self.get_parameter('topic_out').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        weights_file = self.get_parameter('weights_file').get_parameter_value().string_value
        gpu = self.get_parameter('gpu').get_parameter_value().integer_value
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        # modify config file
        self.config = self._init_args(config_file, weights_file)

        # create subscriber
        self.subscription = self.create_subscription(Image, topic_in, self.listener_callback, 5)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Listening to %s topic' % topic_in)

        # create publisher
        self.publisher_ = self.create_publisher(Image, topic_out, 5)

        # initialize variables
        self.last_time = time.time()
        self.cv_br = CvBridge()

        # Segmenter initialization
        self.models = self._init_model(config_file, weights_file)
        self.get_logger().info('depth model initialized')

    def _init_args(self, config_file, weights_file):
        with open(config_file, 'r') as cfg:
            config = yaml.safe_load(cfg)

        return config

    def listener_callback(self, msg):
        # transform message
        img = self.cv_br.imgmsg_to_cv2(msg)

        # segment image
        depth = self._run_model(img)

        # publish results
        img_msg = self.cv_br.cv2_to_imgmsg(depth)
        img_msg.header = msg.header
        self.publisher_.publish(img_msg)

        # compute true fps
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_time)
        self.last_time = curr_time
        self.get_logger().info('Computing depth image at %.01f fps' % fps)

    def _init_model(self, config_file, weights_file):
        models = {}

        # Depth encoder
        models["depth_encoder"] = networks.HRNetPyramidEncoder(self.config).to(torch.device("cuda"))

        # Depth decoder
        models["depth_decoder"] = networks.DepthDecoder(models["depth_encoder"].num_ch_enc).to(torch.device("cuda"))

        # Paths to the models
        encoder_path = os.path.join(weights_file, "depth_encoder.pth")
        decoder_path = os.path.join(weights_file, "depth_decoder.pth")

        # Load model weights
        encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        models["depth_encoder"].load_state_dict({k: v for k, v in encoder_dict.items()
                                                 if k in models["depth_encoder"].state_dict()})
        models["depth_decoder"].load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

        # Move network weights from cpu to gpu device
        models["depth_encoder"].to(torch.device("cuda")).eval()
        models["depth_decoder"].to(torch.device("cuda")).eval()

        return models

    def _run_model(self, img):
        h, w, _ = np.shape(img)
        img = cv2.resize(img, (640, 192), interpolation=cv2.INTER_LINEAR)
        img = torch.tensor(np.array(img, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.device("cuda"))
        features, _ = self.models["depth_encoder"](img)
        output = self.models["depth_decoder"](features)

        # Convert disparity into depth maps
        pred_disp = 1 / 80. + (1 / 0.1 - 1 / 80.) * output[("disp", 0)].detach()
        pred_disp = pred_disp[0, 0].cpu().numpy()
        pred_depth_raw = 3. / pred_disp.copy()
        pred_depth_raw = cv2.resize(pred_depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        return pred_depth_raw


def main(args=None):
    rclpy.init(args=args)

    depth_publisher = DevMonoDepth()

    rclpy.spin(depth_publisher)

    depth_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
