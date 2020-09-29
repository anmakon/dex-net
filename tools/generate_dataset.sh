#!/bin/bash

# Script to generate dexnet dataset
# First input = value of elevation angle
# Second input (not needed) amount of datapoints to be generated

CurrentDate=`date +"%Y%m%d"`

echo Do you want to save the dataset from the dexnet database? [0/1]
read cond

if [ $cond == 1 ] 
then
	echo save
	if [ -z $2 ]
	then
		python tools/save_objects_from_dset.py
	else
		python tools/save_objects_from_dset.py --num $2
	fi
fi

echo render 
python tools/reprojection.py --dir "$CurrentDate"_reprojection_elev_$1 --elev $1
