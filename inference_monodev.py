import time

import networks
import cv2
from PIL import Image
import numpy as np
import torch
import yaml
import os
import argparse
from torchvision import transforms
from torch.nn import functional as F

MAX_DEPTH = 80.0
MIN_DEPTH = 0.1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_out", type=str)
    parser.add_argument("--weights_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--files_list", type=str)
    parser.add_argument("--models_fcn_name", type=str, default="HRNet")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hr", action="store_true")
    parser.add_argument("--scales", type=int, default=4)
    parser.add_argument("--color", action="store_true")
    parser.set_defaults(color=False)

    args = parser.parse_args()

    return args


def _init_args(config_file, weights_file):
    with open(config_file, "r") as cfg:
        config = yaml.safe_load(cfg)

    return config


def _network_selection(models, model_key, config, models_fcn_name, num_layers, scales):
    if model_key == "encoder":
        if "hrnet" == models_fcn_name.lower():
            return networks.HRNetPyramidEncoder(config).to("cuda")
        elif "densenet" == models_fcn_name.lower():
            return networks.DensenetPyramidEncoder(densnet_version=num_layers).to("cuda")
        elif "resnet" == models_fcn_name.lower():
            return networks.ResnetEncoder(num_layers, False).to("cuda")
        else:
            raise RuntimeError("Choose a depth encoder within available scope")

    elif model_key == "depth_decoder":
        return networks.DepthDecoder(models["depth_encoder"].num_ch_enc, scales).to("cuda")

    else:
        raise RuntimeError("Don't forget to mention what you want!")

    return models


def _init_model(config, weights_file, models_fcn_name, num_layers, scales):
    models = {}

    # Depth encoder
    models["depth_encoder"] = _network_selection(models, "encoder", config, models_fcn_name, num_layers, scales)

    # Depth decoder
    models["depth_decoder"] = _network_selection(models, "depth_decoder", config, models_fcn_name, num_layers, scales)

    # Paths to the models
    print(weights_file)
    encoder_path = os.path.join(weights_file, "encoder.pth")
    decoder_path = os.path.join(weights_file, "depth_decoder.pth")

    # Load model weights
    encoder_dict = torch.load(encoder_path, map_location=torch.device("cpu"))
    models["depth_encoder"].load_state_dict(
        {k: v for k, v in encoder_dict.items() if k in models["depth_encoder"].state_dict()}
    )
    models["depth_decoder"].load_state_dict(torch.load(decoder_path, map_location=torch.device("cpu")))

    # Move network weights from cpu to gpu device
    models["depth_encoder"].to(torch.device("cuda")).eval()
    models["depth_decoder"].to(torch.device("cuda")).eval()

    return models


def _run_model(img, models):
    # prepare data
    h, w, _ = np.shape(img)
    input_res = (384, 1248) if args.hr else (192, 640)
    resize = transforms.Resize(input_res, interpolation=Image.ANTIALIAS)
    img = resize(img)
    img = torch.tensor(np.array(img, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.device("cuda"))

    # compute depth
    features, _ = models["depth_encoder"](img)

    output = models["depth_decoder"](features)

    # Convert disparity into depth maps
    pred_disp = 1 / MAX_DEPTH + (1 / MIN_DEPTH - 1 / MAX_DEPTH) * output[("disp", 0)]
    pred_disp = pred_disp[0, 0].cpu().numpy()
    pred_depth_raw = 3.0 / pred_disp.copy()
    # pred_depth_raw = cv2.resize(pred_depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_depth_t = torch.tensor(pred_depth_raw).unsqueeze(0).unsqueeze(0)
    pred_depth_t[pred_depth_t < MIN_DEPTH] = MIN_DEPTH
    pred_depth_t[pred_depth_t > MAX_DEPTH] = MAX_DEPTH

    pred_depth_o = Image.fromarray(np.array(F.interpolate(pred_depth_t, (h, w)).squeeze() * 256, dtype=np.uint16))

    return pred_depth_o


def mini_loop(img, models, path_out, name_file, color=False):
    # get depth img
    with torch.no_grad():
        depth = _run_model(img, models)
    # depth = np.asarray(depth * 256, dtype=np.uint16)
    # depth = np.clip(depth, 0, 80 * 256)
    if color:
        depth = cv2.applyColorMap(np.asarray((255.0 * depth) / 80, dtype=np.uint8), cv2.COLORMAP_JET)

    # write img
    # if not os.path.exists(os.path.join(path_out, "depth")):
    os.makedirs(os.path.join(path_out), exist_ok=True)

    filename_out = os.path.join(path_out, "%s" % name_file)
    # cv2.imwrite(filename_out, depth)
    depth.save(filename_out)


def main(args):
    config = _init_args(args.config_file, args.weights_file)
    models = _init_model(config, args.weights_file, args.models_fcn_name, args.num_layers, range(args.scales))

    last_time = time.time()
    last_update = last_time
    with open(args.files_list, "r") as f:
        v_files = [line.rstrip() for line in f.readlines()]

    for file_idx, file in enumerate(v_files):
        img = Image.open(file).convert("RGB")
        name_file = file.split("/")[-1]
        # drive = file.split("/")[-4]
        if os.path.exists(os.path.join(args.path_out, name_file)):
            continue
        mini_loop(img, models, args.path_out, name_file, color=args.color)

        # compute true fps
        curr_time = time.time()
        fps = 1 / (curr_time - last_time)
        last_time = curr_time
        if (curr_time - last_update) > 5:
            last_update = curr_time

        print("Computing depth image %d/%d at %.01f fps" % (file_idx, len(v_files), fps))


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    main(args)
