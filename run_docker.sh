#!/bin/bash

xhost +local:docker

docker run -it -e DISPLAY -e QT_X11_NO_MITSHM=1 -v ${PWD}:${PWD} -w ${PWD} -v /tmp/.X11-unix:/tmp/.X11-unix dexnet:latest
