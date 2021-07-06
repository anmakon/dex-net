#!/bin/bash
docker build -t dexnet_pycharm:gpu -f ./pycharm-dockerfile \
	--build-arg USER_ID=$(id -u) \
	--build-arg GROUP_ID=$(id -g) .
