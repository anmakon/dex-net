#!/bin/bash
PATH_DSET="/home/anna/Grasping/data"

xhost +local:root

docker run -it \
-e DISPLAY \
-e QT_x11_NO_MITSHM=1 \
-v ${PWD}:${PWD} \
-w ${PWD} \
-v $PATH_DSET:/data \
--gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--name dexnet \
dexnet:gpu

xhost -local:root


#-e DISPLAY=unix$DISPLAY \
#--runtime=nvidia \
