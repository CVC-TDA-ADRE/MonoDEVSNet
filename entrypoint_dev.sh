#!/bin/bash
set -e

# compile monodepth2 framework
cd ${MONODEVSNET_ROOT}/utils
chmod +x prepare_monodepth2_framework.sh
./prepare_monodepth2_framework.sh
cd ..

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/install/setup.bash"
source "/ROS_LIBS/vision_opencv/install/setup.bash"
source "/ROS_LIBS/cv_tools/ros/install/setup.bash"
exec "$@"

