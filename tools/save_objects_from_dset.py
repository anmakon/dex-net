import argparse
import cPickle as pkl
import json
import csv
import os
import numpy as np

from autolab_core import YamlConfig
import autolab_core.utils as utils
from meshpy import ObjFile, StablePoseFile

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

"""
Script to save dexnet objects for dexnet database to file.
Object, pose and grasp can be specified via the script 
dexnet/tools/create_validation_pointers.py.
Data files are being saved to dexnet/data/meshes/dexnet.
With --num, the maximum amount of grasps in one subset (KIT and 3DNet)
can be specified.
"""

class GraspInfo(object):
	def __init__(self,grasp,collision_free,phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.phi = phi

def save_dexnet_objects(output_path, database, target_object_keys, config, pointers,num):

	file_arr = []

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	for each_file in os.listdir(output_path):
		os.remove(output_path+'/'+each_file)
	gripper = RobotGripper.load(config['gripper'])

	# Setup grasp params:
	table_alignment_params = config['table_alignment']
	min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
	max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
	max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
	num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

	phi_offsets = []
	if max_grasp_approach_offset == min_grasp_approach_offset:
		phi_inc = 1
	elif num_grasp_approach_samples == 1:
		phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
	else:
		phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)

	phi = min_grasp_approach_offset
	while phi <= max_grasp_approach_offset:
		phi_offsets.append(phi)
		phi += phi_inc

	# Setup collision checking
	coll_check_params = config['collision_checking']
	approach_dist = coll_check_params['approach_dist']
	delta_approach = coll_check_params['delta_approach']
	table_offset = coll_check_params['table_offset']

	table_mesh_filename = coll_check_params['table_mesh_filename']
	if not os.path.isabs(table_mesh_filename):
		table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..',table_mesh_filename)
	table_mesh = ObjFile(table_mesh_filename).read()

	dataset_names = target_object_keys.keys()
	datasets = [database.dataset(dn) for dn in dataset_names]

	start = 0
	for dataset in datasets:
		target_object_keys[dataset.name] = []
		end = start + len(dataset.object_keys)
		for cnt, _id in enumerate(pointers.obj_ids):
			if _id >= end or _id < start:
				continue
			target_object_keys[dataset.name].append(dataset.object_keys[_id-start])
			file_arr.append(tuple([dataset.object_keys[_id-start],pointers.tensor[cnt],pointers.array[cnt]]))
		start += end
	print(file_arr)
	file_arr = np.array(file_arr,dtype=[('Object_id',(np.str_,40)),('Tensor',int),('Array',int)])
		
	# Precompute set of valid grasps
	candidate_grasps_dict = {}

	counter = 0
	for dataset in datasets:
		for obj in dataset:
			if obj.key not in target_object_keys[dataset.name]:
				continue
			# Initiate candidate grasp storage
			candidate_grasps_dict[obj.key] = {}

			# Setup collision checker
			collision_checker = GraspCollisionChecker(gripper)
			collision_checker.set_graspable_object(obj)

			# Read in the stable poses of the mesh
			stable_poses = dataset.stable_poses(obj.key)

			# Get the stable pose of the validation point
			# The previous pose number is the pose number of the last grasp of the previous object
			try:
				stable_pose = stable_poses[pointers.pose_num[counter]]
			except:
				print("Problems with reading pose. Tensor %d, Array %d, Pose %d" %\
					(pointers.tensor[counter],pointers.array[counter],pointers.pose_num[counter]))
				counter += 1
				print("Continue.")
				continue
			candidate_grasps_dict[obj.key][stable_pose.id] = []
			
			# Setup table in collision checker
			T_obj_stp = stable_pose.T_obj_table.as_frames('obj','stp')
			T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj','table')
			T_table_obj = T_obj_table.inverse()

			collision_checker.set_table(table_mesh_filename, T_table_obj)

			# read grasp and metrics
			grasps = dataset.grasps(obj.key,gripper=gripper.name)

			#align grasps with the table
			aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]
			i = 0
			# Check grasp validity
			for aligned_grasp in aligned_grasps:
				# Check angle with table plane and skip unaligned grasps
				_, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
				perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
				if not perpendicular_table:
					continue
	
				# Check wheter any valid approach directions are collision free
				collision_free = False
				for phi_offset in phi_offsets:
					rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
					collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
					if not collides:
						collision_free = True
						break

				#Store if aligned to table
				if i == pointers.grasp_num[counter]:
					candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp,collision_free))
					# Add file pointers to file arr
				i += 1
			counter += 1

			# Save files
			print("Saving file: ",obj.key)
			savefile = ObjFile("./data/meshes/dexnet/"+obj.key+".obj") 
			savefile.write(obj.mesh)  
			# Save stable poses 
			save_stp = StablePoseFile("./data/meshes/dexnet/"+obj.key+".stp") 
			save_stp.write(stable_poses) 
			# Save candidate grasp info 
			pkl.dump(candidate_grasps_dict[obj.key], open("./data/meshes/dexnet/"+obj.key+".pkl", 'wb')) 
			# Save grasp metrics 
			candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
			candidate_grasps = [g.grasp for g in candidate_grasp_info]
			grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name) 
			write_metrics = json.dumps(grasp_metrics) 
			f = open("./data/meshes/dexnet/"+obj.key+".json","w") 
			f.write(write_metrics) 
			f.close()

			if num is not None and counter >= num:
				break
	with open('./data/meshes/dexnet/files.csv','w') as csv_file:
		csv_writer = csv.writer(csv_file,delimiter=',')
		for point in file_arr:
			csv_writer.writerow(point)




class ValidationPointers(object):
	def __init__(self,filename='./data/generated_val_indices.txt'):
		f = open(filename,'rb')
		dtype = [('Tensor',int),('Array',int),('Obj_id',int),('Pose_num',int),('Grasp_num',int),('Prev_obj_id',int)]
		data = np.array([tuple(map(int,line.split(','))) for line in f if not 'label' in line],dtype=dtype)
		data = np.sort(data,order='Obj_id')
		self.tensor = data['Tensor']
		self.array = data['Array']
		self.obj_ids = data['Obj_id']
		self.pose_num = data['Pose_num']
		self.grasp_num = data['Grasp_num']
		self.prev_obj_ids = data['Prev_obj_id']

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--num",
				type = int,
				default = None)

	args = parser.parse_args()
	num = args.num

	config_filename = "./cfg/tools/generate_gqcnn_dataset.yaml"
	output_path = "./data/meshes/dexnet"

	config = YamlConfig(config_filename)
	
	database = Hdf5Database(config['database_name'],
				access_level = READ_ONLY_ACCESS)
	
	pointers = ValidationPointers()
	
	target_object_keys = config['target_objects']
	
	save_dexnet_objects(output_path,database,target_object_keys,config,pointers,num)
