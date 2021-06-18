#!/bin/bash
set -e

# compile monodepth2 framework
cd ${MONODEVSNET_ROOT}/utils
chmod +x prepare_monodepth2_framework.sh
./prepare_monodepth2_framework.sh
cd ..
exec "$@"

