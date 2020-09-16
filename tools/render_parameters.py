import numpy as np
import json
import cPickle as pkl

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, StablePoseFile, RenderMode, SceneObject
from perception import CameraIntrinsics, BinaryImage, DepthImage
from autolab_core import Point, RigidTransform, YamlConfig
import matplotlib.pyplot as plt

"""
Script to test the effect of the environment parameters (camera intrinsics, camera position and object position)
on the rendered images. Rendered depth image is being save in in out_dir
"""


# Paths to data
obj_file = './data/meshes/elephant.obj'
stp_file = './data/meshes/elephant.stp'
table_file = './data/meshes/table.obj'
metrics_file = './data/meshes/grasp_metrics.json'
grasps_file = './data/meshes/candidate_grasp_info.pkl'

config = YamlConfig('./cfg/tools/generate_gqcnn_dataset.yaml')
RANDOM = False
RANDOM_POSITIONS = 1
SHOW_IMAGE = True

out_dir = './data/render_test/'

data = ' 1 - f\n 2 - cx\n 3 - cy\n 4 - radius\n 5 - elev\n 6 - az\n 7 - roll\n 8 - x\n 9 - y'

class GraspInfo(object):
	def __init__(self, grasp, collision_free, phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.phi = phi

def visualise_sample(sample,candidate_grasp_info,grasp_metrics,modification,pose):
	depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image
	
	T_stp_camera = sample.camera.object_to_camera_pose
	shifted_camera_intr = sample.camera.camera_intr

	cx = depth_im_table.center[1]
	cy = depth_im_table.center[0]

	#Compute camera intrinsics
	camera_intr_scale = 32.0 /96.0
	cropped_camera_intr = shifted_camera_intr.crop(96, 96, cy, cx)
	final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)
	for cnt,grasp_info in enumerate(candidate_grasp_info[0:3]):
		grasp = grasp_info.grasp
		collision_free = grasp_info.collision_free

		T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj',T_stp_camera.from_frame)
		grasp_2d = grasp.project_camera(T_obj_camera,shifted_camera_intr)

		dx = cx - grasp_2d.center.x
		dy = cy - grasp_2d.center.y
		translation = np.array([dy,dx])

		depth_im_tf_table = depth_im_table.transform(translation,grasp_2d.angle)

		depth_im_tf_table = depth_im_tf_table.crop(96,96)

		im = plt.imshow(depth_im_tf_table.data,cmap='Greys',vmin = 0.55, vmax = 0.77)
		plt.title('Elephant #'+str(pose)+' '+modification[1:])
		plt.colorbar(im, orientation='vertical')
		plt.savefig(out_dir+str(pose)+'_'+str(cnt)+modification+'.png',facecolor=None)
		plt.close()

grasp_metrics = json.load(open(metrics_file,'r'))
candidate_grasp_info = pkl.load(open(grasps_file,'rb'))

obj_reader = ObjFile(obj_file)
table_mesh = ObjFile(table_file).read()
stp_reader = StablePoseFile(stp_file)
stable_poses = stp_reader.read()

object_mesh = obj_reader.read()

# Iterate through stable poses
for i,stable_pose in enumerate(stable_poses):
	T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
	T_obj_stp = object_mesh.get_T_surface_obj(T_obj_stp)

	# sample images from random variable
	T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
	scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

	stop = ''

	while not (stop == 'y'):
		environment_config = config['env_rv_params']
		modification = ''
		string = ''

		print(data)
		change = str(raw_input("Change config? Input number or 0 for no change. "))
		while change != '0' and change != '':
			if change == '1':
				value = int(input("Value for focal length: "))
				environment_config['min_f'] = value
				environment_config['max_f'] = value
				string = '_focal_length_'+str(value)
			elif change == '2':
				value = float(input("Value for cx: "))
				environment_config['min_cx'] = value
				environment_config['max_cx'] = value
				string = '_cx_'+str(value)
			elif change == '3':
				value = float(input("Value for cy: "))
				environment_config['min_cy'] = value
				environment_config['max_cy'] = value
				string = '_cy_'+str(value)
			elif change == '4':
				value = float(input("Value for radius: "))
				environment_config['min_radius'] = value
				environment_config['max_radius'] = value
				string = '_radius_'+str(value)
			elif change == '5':
				value = float(input("Value for elev: "))
				environment_config['min_elev'] = value
				environment_config['max_elev'] = value
				string = '_elev_'+str(value)
			elif change == '6':
				value = float(input("Value for az: "))
				environment_config['min_az'] = value
				environment_config['max_az'] = value
				string = '_az_'+str(value)
			elif change == '7':
				value = float(input("Value for roll: "))
				environment_config['min_roll'] = value
				environment_config['max_roll'] = value
				string = '_roll_'+str(value)
			elif change == '8':
				value = float(input("Value for x: "))
				environment_config['min_x'] = value
				environment_config['max_x'] = value
				string = '_x_'+str(value)
			elif change == '9':
				value = float(input("Value for y: "))
				environment_config['min_y'] = value
				environment_config['max_y'] = value
				string = '_y_'+str(value)
			modification += string 
			change = str(raw_input("Change config? Input number or 0 for no change. "))

		
		urv = UniformPlanarWorksurfaceImageRandomVariable(object_mesh,
									[RenderMode.DEPTH_SCENE],
									'camera',
									environment_config,
									scene_objs=scene_objs,
									stable_pose=stable_pose)
		if not RANDOM:
			print("No randomised camera position used")
			sample = urv.sample()
			visualise_sample(sample,candidate_grasp_info,grasp_metrics,modification,i)
		else:
			print("Randomised camera position used")
			render_samples = urv.rvs(size = RANDOM_POSITIONS)
			print("Rendered samples")
			for render_sample in render_samples:
				visualise_sample(render_sample,candidate_grasp_info,grasp_metrics,i)
		stop = raw_input("Stop? ") 
