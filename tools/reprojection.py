import numpy as np
import json
import cPickle as pkl
import gc
import csv
import os
import argparse

from PIL import Image

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, StablePoseFile, RenderMode, SceneObject
from perception import CameraIntrinsics, BinaryImage, DepthImage
import autolab_core.utils as utils
from autolab_core import Point, RigidTransform, YamlConfig
import matplotlib.pyplot as plt
from dexnet.learning import TensorDataset

"""
Script to render dexnet_2.0 dataset from oblique view, reproject into 3D, apply transformation to overhead camera at given distance
and project back into depth image. 
"""
class GraspInfo(object):
	def __init__(self,grasp,collision_free,phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.phi = phi

class Reprojection():
	def __init__(self,path,elev):
		table_file = './data/meshes/table.obj'
		self.table_mesh = ObjFile(table_file).read()
		self.data_dir = './data/meshes/dexnet/'
		self.output_dir = './data/rprojection_test'
		self.config = YamlConfig('./cfg/tools/generate_gqcnn_dataset.yaml')
		
		self.random_positions = 1
		self.image_size = (480,640)
		self.elev = 0
		self.use_PCA = False

		if path is not None:
			self.output_dir = './data/'+path
	
		if elev is not None:
			print("Elevation angle is being set to %d"%elev)
			self.config['env_rv_params']['min_elev'] = elev
			self.config['env_rv_params']['max_elev'] = elev
			self.elev = elev
		
		tensor_config = self.config['tensors']
		tensor_config['fields']['depth_ims_tf_table']['height'] = 32
		tensor_config['fields']['depth_ims_tf_table']['width'] = 32
		tensor_config['fields']['obj_masks']['height'] = 32
		tensor_config['fields']['obj_masks']['width'] = 32
		tensor_config['fields']['robust_ferrari_canny']= {}
		tensor_config['fields']['robust_ferrari_canny']['dtype']= 'float32'
		tensor_config['fields']['ferrari_canny']= {}
		tensor_config['fields']['ferrari_canny']['dtype']= 'float32'
		tensor_config['fields']['force_closure']= {}
		tensor_config['fields']['force_closure']['dtype']= 'float32'

		self.tensor_dataset = TensorDataset(self.output_dir,tensor_config)
		self.tensor_datapoint = self.tensor_dataset.datapoint_template

	def _load_file_ids(self):
		dtype = [('Obj_id',(np.str_,40)),('Tensor',int),('Array',int)]

		with open(self.data_dir+'files.csv','r') as csv_file:
			data=[]
			csv_reader = csv.reader(csv_file,delimiter=',')
			for row in csv_reader:
				data.append(tuple([int(value) if value.isdigit() else value for value in row]))
			self.file_arr = np.array(data,dtype=dtype)
		files = os.listdir(self.data_dir)
		files = [name.split('.')[0] for name in files]
		files.remove('files')
		self.all_objects = list(set(files))

	def _load_data(self,obj_id):
		self.grasp_metrics = json.load(open(self.data_dir+obj_id+'.json','r'))
		self.candidate_grasps_dict = pkl.load(open(self.data_dir+obj_id+'.pkl','rb'))
		
		self.object_mesh = ObjFile(self.data_dir+obj_id+'.obj').read()
		self.stable_poses = StablePoseFile(self.data_dir+obj_id+'.stp').read()
		self.tensor = self.file_arr['Tensor'][np.where(self.file_arr['Obj_id']==obj_id)][0]
		self.array = self.file_arr['Array'][np.where(self.file_arr['Obj_id']==obj_id)][0]

	def start_rendering(self):
		self._load_file_ids()
		for object_id in self.all_objects:
			self._load_data(object_id)
			for i, stable_pose in enumerate(self.stable_poses):
				try:
					candidate_grasp_info = self.candidate_grasps_dict[stable_pose.id]
				except KeyError:
#					print("Pose: ",stable_pose.id,"not included")
					continue

				T_obj_stp = stable_pose.T_obj_table.as_frames('obj','stp')
				T_obj_stp = self.object_mesh.get_T_surface_obj(T_obj_stp)
			
				T_table_obj = RigidTransform(from_frame='table',to_frame='obj')
				scene_objs = {'table': SceneObject(self.table_mesh,T_table_obj)}
				
				urv = UniformPlanarWorksurfaceImageRandomVariable(self.object_mesh,
											[RenderMode.DEPTH_SCENE,RenderMode.SEGMASK],
											'camera',
											self.config['env_rv_params'],
											scene_objs = scene_objs,
											stable_pose=stable_pose)
				render_sample = urv.rvs(size = self.random_positions)
				#for render_sample in render_samples:
				binary_im = render_sample.renders[RenderMode.SEGMASK].image
				depth_im = render_sample.renders[RenderMode.DEPTH_SCENE].image
				print(depth_im.data.shape)
				orig_im = Image.fromarray(depth_im.data*100)
				orig_im.show()
				orig_im.convert('RGB').save('./data/reprojection/'+object_id+'_elev_'+str(self.elev)+'_original.png')

				shifted_camera_intr = render_sample.camera.camera_intr
				depth_points = self._reproject_to_3D(depth_im,shifted_camera_intr)
				transformed_points = self._transformation(depth_points)
				projected_depth_im = self._projection(transformed_points,shifted_camera_intr)
				print(projected_depth_im)
				im = Image.fromarray(projected_depth_im * 100)
				im.show()
				im.convert('RGB').save('./data/reprojection/'+object_id+'_elev_'+str(self.elev)+'_reprojected.png')

	def _projection(self,transformed_points,camera_intr):
		# Use Camera intrinsics
		K = camera_intr.proj_matrix
		K[0][-1] = 320
		K[1][-1] = 240
		print("K: ",K)
		
		projected_points = np.dot(K,transformed_points)

		point_depths = projected_points[2,:]
		table = np.median(point_depths)
		point_z = np.tile(point_depths,[3,1])
		points_proj = np.divide(projected_points,point_z)
		
		# Rounding
		points_proj = np.round(points_proj)
		points_proj = points_proj[:2,:].astype(np.int16)
		print("Points proj: ",points_proj)
		
		valid_ind = np.where((points_proj[0,:] >= 0) &\
					(points_proj[0,:] < self.image_size[0]) &\
					(points_proj[1,:] >= 0) &\
					(points_proj[1,:] < self.image_size[0]))[0]
		depth_data = np.full([self.image_size[0],self.image_size[1]],table)
		for ind in valid_ind:
			prev_depth = depth_data[points_proj[1,ind],points_proj[0,ind]]
			if prev_depth == table or prev_depth >= point_depths[ind]:
				depth_data[points_proj[1,ind],points_proj[0,ind]] = point_depths[ind]
		return depth_data

		

	def _transformation(self,points):
		# Points are given in camera frame. Transform to new camera frame!
		camera_new_position = np.array([[0],[0],[0]])
		if self.use_PCA:
			# Extract surface normal via PCA (extract table plane)
			eigenvalues,eigenvectors = self._PCA(points)
			print("Eigenvectors: ",eigenvectors)
			#Get camera position to be above the object 
			# Use eigenvector and camera position to create transformation matrix
			Rt = np.append(eigenvectors,camera_new_position,axis=1)
		else:
			ang = np.deg2rad(self.elev)
			Rot = np.array([[1, 0, 0, 0],\
					[0, np.cos(ang),-np.sin(ang), 0],\
					[0, np.sin(ang),np.cos(ang), 0],\
					[0, 0, 0, 1]])
			dist = self.config['env_rv_params']['min_radius'] * np.sin(ang)
			height = self.config['env_rv_params']['min_radius'] - self.config['env_rv_params']['min_radius'] * np.cos(ang) 
			# Rotation around x axis, therefore translation back to object center alongside y axis
			trans = np.array([[1, 0, 0, 0],\
					[0, 1, 0, dist],\
					[0, 0, 1, height],\
					[0, 0, 0, 1]])
			Rt = np.dot(trans,Rot)
			Rt = Rt[0:3]

		print("Rt: ",Rt)
		# Apply transformation to homogeneous coordinates of the points
		homogeneous_points = np.append(np.transpose(points),np.ones((1,len(points))),axis=0)
		transformed_points = np.dot(Rt,homogeneous_points)
		return transformed_points

	def _PCA(self,points,sorting = True):
		mean = np.mean(points,axis=0)
		data_adjusted = points-mean
		
		matrix = np.cov(data_adjusted.T)
		eigenvalues, eigenvectors = np.linalg.eig(matrix)

		if sorting:
			sort = eigenvalues.argsort()[::-1]
			eigenvalues = eigenvalues[sort]
			eigenvectors = eigenvectors[:,sort]
		return eigenvalues, eigenvectors

	def _reproject_to_3D(self,depth_im,camera_intr):
		# depth points will be given in camera frame!
		depth_points = camera_intr.deproject(depth_im).data.T
		return depth_points

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str,default=None)
	parser.add_argument('--elev',type=int,default=None,help='Elevation angle between camera z-axis and vertical axis')
	
	args = parser.parse_args()
	path = args.dir
	elev = args.elev
	
	reprojection = Reprojection(path,elev)
	reprojection.start_rendering()
