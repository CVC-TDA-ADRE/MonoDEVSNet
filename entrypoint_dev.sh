#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/install/setup.bash"
source "/ROS_LIBS/vision_opencv/install/setup.bash"
source "/ROS_LIBS/cv_tools/ros/install/setup.bash"
exec "$@"

