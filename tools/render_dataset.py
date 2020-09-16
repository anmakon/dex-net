import numpy as np
import json
import cPickle as pkl
import argparse
import gc
import csv
import os

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, StablePoseFile, RenderMode, SceneObject
from perception import CameraIntrinsics, BinaryImage, DepthImage
import autolab_core.utils as utils
from autolab_core import Point, RigidTransform, YamlConfig
import matplotlib.pyplot as plt
from dexnet.learning import TensorDataset


"""
Script to render dexnet_2.0 dataset. Output directory can be specified with --dir, 
elevation angle (angle between z-axis and camera axis) can be specified with --elev. 
"""

table_file = './data/meshes/table.obj'

data_dir = './data/meshes/dexnet/'
output_dir = './data/render_test/'

config = YamlConfig('./cfg/tools/generate_gqcnn_dataset.yaml')

RANDOM = False
RANDOM_POSITIONS = 1


class GraspInfo(object):
	def __init__(self, grasp, collision_free, phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.phi = phi

def visualise_sample(sample,candidate_grasp_info,grasp_metrics,tensor,array,cur_image_label=0,cur_pose_label=0,cur_obj_label=0):
	"""Applying transformations to the images and camera intrinsics.
	Visualising the binary and depth image of the object. 
	Adding the corresponding values to the dataset"""
	binary_im = sample.renders[RenderMode.SEGMASK].image
	depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image
	
	T_stp_camera = sample.camera.object_to_camera_pose
	shifted_camera_intr = sample.camera.camera_intr

	cx = depth_im_table.center[1]
	cy = depth_im_table.center[0]

	#Compute camera intrinsics
	camera_intr_scale = 32.0 /96.0
	cropped_camera_intr = shifted_camera_intr.crop(96, 96, cy, cx)
	final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)
	# Iterate through grasps
	for grasp_info in candidate_grasp_info:
		grasp = grasp_info.grasp
		collision_free = grasp_info.collision_free

		T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj',T_stp_camera.from_frame)
		grasp_2d = grasp.project_camera(T_obj_camera,shifted_camera_intr)

		dx = cx - grasp_2d.center.x
		dy = cy - grasp_2d.center.y
		translation = np.array([dy,dx])

		# Transform, crop and resize image
		binary_im_tf = binary_im.transform(translation,grasp_2d.angle)
		depth_im_tf_table = depth_im_table.transform(translation,grasp_2d.angle)

		binary_im_tf = binary_im_tf.crop(96,96)
		depth_im_tf_table = depth_im_tf_table.crop(96,96)

		binary_im_tf = binary_im_tf.resize((32,32),interp='nearest')
		depth_im_tf_table = depth_im_tf_table.resize((32,32))

		hand_pose = np.r_[grasp_2d.center.y,
					grasp_2d.center.x,
					grasp_2d.depth,
					grasp_2d.angle,
					grasp_2d.center.y - shifted_camera_intr.cy,
					grasp_2d.center.x - shifted_camera_intr.cx,
					grasp_2d.width_px/3]
	
		tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
		tensor_datapoint['obj_masks'] = binary_im_tf.raw_data
		tensor_datapoint['hand_poses'] = hand_pose
		tensor_datapoint['obj_labels'] = cur_obj_label
		tensor_datapoint['collision_free'] = collision_free
		tensor_datapoint['pose_labels'] = cur_pose_label
		tensor_datapoint['image_labels'] = cur_image_label
		tensor_datapoint['files'] = [tensor,array]
	
		for metric_name,metric_val in grasp_metrics[str(grasp.id)].iteritems():
			coll_free_metric = (1*collision_free) * metric_val
			tensor_datapoint[metric_name] = coll_free_metric
		tensor_dataset.add(tensor_datapoint)
		print("Saved dataset point")

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type= str, default=None)
parser.add_argument('--elev',type=int,default=None,help='Elevation angle between camera z-axis and vertical axis')

args = parser.parse_args()
path = args.dir
elev = args.elev
if path is not None:
	output_dir = './data/'+ path

if elev is not None:
	print("Elevation angle is being set to %d"%elev)
	config['env_rv_params']['min_elev'] = elev
	config['env_rv_params']['max_elev'] = elev
	

tensor_config = config['tensors']
tensor_config['fields']['depth_ims_tf_table']['height'] = 32
tensor_config['fields']['depth_ims_tf_table']['width'] = 32
tensor_config['fields']['obj_masks']['height'] = 32
tensor_config['fields']['obj_masks']['width'] = 32
tensor_config['fields']['robust_ferrari_canny'] = {}
tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
tensor_config['fields']['ferrari_canny'] = {}
tensor_config['fields']['ferrari_canny']['dtype'] = 'float32'
tensor_config['fields']['force_closure'] = {}
tensor_config['fields']['force_closure']['dtype'] = 'float32'

tensor_dataset = TensorDataset(output_dir,tensor_config)
tensor_datapoint = tensor_dataset.datapoint_template

cur_pose_label = 0
cur_obj_label = 0
cur_image_label = 0

dtype = [('Obj_id',(np.str_,40)),('Tensor',int),('Array',int)]
with open(data_dir+'files.csv','r') as csv_file:
	csv_reader = csv.reader(csv_file,delimiter=',')
	data = []
	for row in csv_reader:
		data.append(tuple([int(value) if value.isdigit() else value for value in row]))
	file_arr = np.array(data,dtype=dtype)

files = os.listdir(data_dir)
files = [name.split('.')[0] for name in files]
files.remove('files')
all_objects = list(set(files))

# Iterate through all objects in data directory
for object_id in all_objects:
	# Load data
	grasp_metrics = json.load(open(data_dir+object_id+'.json','r'))
	candidate_grasps_dict = pkl.load(open(data_dir+object_id+'.pkl','rb'))

	obj_reader = ObjFile(data_dir+object_id+'.obj')
	table_mesh = ObjFile(table_file).read()
	stp_reader = StablePoseFile(data_dir+object_id+'.stp')
	stable_poses = stp_reader.read()

	object_mesh = obj_reader.read()

	# Get tensor and array
	print(object_id)
	tensor = file_arr['Tensor'][np.where(file_arr['Obj_id']==object_id)][0]
	array = file_arr['Array'][np.where(file_arr['Obj_id']==object_id)][0]
	print("Tensor %d, array %d" %(tensor, array))

	# Iterate through stable poses
	for i,stable_pose in enumerate(stable_poses):
		T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
		T_obj_stp = object_mesh.get_T_surface_obj(T_obj_stp)
	
		# sample images from random variable
		T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
		scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}
			
		try:
			candidate_grasp_info = candidate_grasps_dict[stable_pose.id]
		except KeyError:
			continue

		urv = UniformPlanarWorksurfaceImageRandomVariable(object_mesh,
									[RenderMode.DEPTH_SCENE,RenderMode.SEGMASK],
									'camera',
									config['env_rv_params'],
									scene_objs=scene_objs,
									stable_pose=stable_pose)
		if not RANDOM:
			print("No randomised camera position used")
			sample = urv.sample()
			visualise_sample(sample,candidate_grasp_info,grasp_metrics,tensor,array)
		else:
			print("Randomised camera position used")
			render_samples = urv.rvs(size = RANDOM_POSITIONS)
			for render_sample in render_samples:
				visualise_sample(render_sample,candidate_grasp_info,grasp_metrics,tensor,array)
				cur_image_label += 1
		cur_pose_label += 1
		gc.collect()
		# next stable pose
	cur_obj_label +=1
	# next object
# Save dataset
tensor_dataset.flush()
