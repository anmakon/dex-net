import numpy as np
import os

from dexnet.database import Hdf5Database
from dexnet.constants import READ_ONLY_ACCESS
from autolab_core import YamlConfig

config_filename = 'cfg/tools/generate_gqcnn_dataset.yaml'

config = YamlConfig(config_filename)

database = Hdf5Database(config['database_name'],access_level=READ_ONLY_ACCESS)
target_object_keys = config['target_objects']
gripper_name = config['gripper']
env_rv_params = config['env_rv_params']

dataset_names = target_object_keys.keys()
datasets = [database.dataset(dn) for dn in dataset_names]


for dataset in datasets:
	if target_object_keys[dataset.name] == 'all':
		target_object_keys[dataset.name] = dataset.object_keys
	subset = dataset.subset(0,10)
for key in datasets[1].object_keys:
	print(key)

database.close()
