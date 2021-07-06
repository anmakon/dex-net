#!/bin/bash
PATH_DSET="/home/anna/Grasping/data"

xhost +local:root

docker run -it \
-e DISPLAY \
-e QT_x11_NO_MITSHM=1 \
--net=host \
-v /snap/pycharm-community/current:${PWD}/pycharm \
-v ${PWD}:${PWD} \
-w ${PWD} \
-v $PATH_DSET:/data \
--gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--name pycharm-dexnet \
dexnet_pycharm:gpu

xhost -local:root

docker stop pycharm-dexnet
docker rm pycharm-dexnet
