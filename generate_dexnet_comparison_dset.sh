#!/bin/bash

# Script to generate a dexnet 2.0 dataset
# First input for elevation angle, second input for number of objects
CurrentDate=`date +"%Y%m%d"`

source ~/py2/bin/activate
cd dex-net
./tools/generate_dataset.sh $1 $2
cd ..
deactivate
source ~/gq/bin/activate
cd gqcnn
python tools/detailed_analysis.py GQCNN-2.0_benchmark ../dex-net/data/"$CurrentDate"_reprojection_elev_$1/tensors/ --output_dir analysis/SingleFiles/Dataset_Generation/"$CurrentDate"_reprojection_elev_$1
