import numpy as np
import json
import cPickle as pkl
import gc
import os

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, StablePoseFile, RenderMode, SceneObject
from perception import CameraIntrinsics, BinaryImage, DepthImage
import autolab_core.utils as utils
from autolab_core import Point, RigidTransform, YamlConfig
import matplotlib.pyplot as plt

from dexnet.learning import TensorDataset

obj_file = './data/meshes/elephant.obj'
stp_file = './data/meshes/elephant.stp'
table_file = './data/meshes/table.obj'
metrics_file = './data/meshes/grasp_metrics.json'
grasps_file = './data/meshes/candidate_grasp_info.pkl'

data_dir = './data/meshes/dexnet/'
output_dir = './data/render_test/'

config = YamlConfig('./cfg/tools/generate_gqcnn_dataset.yaml')
RANDOM = False
SHOW_IMAGE = False

cur_pose_label = 0
cur_obj_label = 0
cur_image_label = 0

class GraspInfo(object):
	def __init__(self, grasp, collision_free, phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.phi = phi

def visualise_sample(sample,candidate_grasp_info,grasp_metrics,cur_image_label=0,cur_pose_label=0,cur_obj_label=0):
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
	for grasp_info in candidate_grasp_info:
		grasp = grasp_info.grasp
		collision_free = grasp_info.collision_free

		T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj',T_stp_camera.from_frame)
		grasp_2d = grasp.project_camera(T_obj_camera,shifted_camera_intr)

		dx = cx - grasp_2d.center.x
		dy = cy - grasp_2d.center.y
		translation = np.array([dy,dx])

		binary_im_tf = binary_im.transform(translation,grasp_2d.angle)
		depth_im_tf_table = depth_im_table.transform(translation,grasp_2d.angle)

		binary_im_tf = binary_im_tf.crop(96,96)
		depth_im_tf_table = depth_im_tf_table.crop(96,96)

		binary_im_tf = binary_im_tf.resize((32,32),interp='nearest')
		depth_im_tf_table = depth_im_tf_table.resize((32,32))

		if SHOW_IMAGE:
			plt.imshow(depth_im_tf_table.data)
			plt.show()
			plt.imshow(binary_im_tf.data)
			plt.show()
		hand_pose = np.r_[grasp_2d.center.y,
					grasp_2d.center.x,
					grasp_2d.depth,
					grasp_2d.angle,
					grasp_2d.center.y - shifted_camera_intr.cy,
					grasp_2d.center.x - shifted_camera_intr.cx,
					grasp_2d.width_px]
	
		tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
		tensor_datapoint['obj_masks'] = binary_im_tf.raw_data
		tensor_datapoint['hand_poses'] = hand_pose
		tensor_datapoint['obj_labels'] = cur_obj_label
		tensor_datapoint['collision_free'] = collision_free
		tensor_datapoint['pose_labels'] = cur_pose_label
		tensor_datapoint['image_labels'] = cur_image_label
	
		for metric_name,metric_val in grasp_metrics[str(grasp.id)].iteritems():
			coll_free_metric = (1*collision_free) * metric_val
			tensor_datapoint[metric_name] = coll_free_metric
		tensor_dataset.add(tensor_datapoint)
		print("Saved datset point")

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
stable_pose_min_p = config['stable_pose_min_p']

files = os.listdir(data_dir)
files = [name.split('.')[0] for name in files]
all_objects = list(set(files))

for object_id in all_objects:
	grasp_metrics = json.load(open(data_dir+object_id+'.json','r'))
	candidate_grasps_dict = pkl.load(open(data_dir+object_id+'.pkl','rb'))

	obj_reader = ObjFile(data_dir+object_id+'.obj')
	table_mesh = ObjFile(table_file).read()
	stp_reader = StablePoseFile(data_dir+object_id+'.stp')
	stable_poses = stp_reader.read()

	object_mesh = obj_reader.read()

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
			visualise_sample(sample,candidate_grasp_info,grasp_metrics)
		else:
			print("Randomised camera position used")
			render_samples = urv.rvs(size = 1)
			print("Rendered samples")
			for render_sample in render_samples:
				visualise_sample(render_sample,candidate_grasp_info,grasp_metrics)
		cur_pose_label += 1
		gc.collect()
	cur_obj_label +=1
tensor_dataset.flush()
