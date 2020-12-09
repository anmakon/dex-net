import numpy as np
import json
import cPickle as pkl
import gc
import csv
import time
import os
import argparse

import open3d as o3d

from PIL import Image,ImageDraw
import cv2

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, StablePoseFile, RenderMode, SceneObject
from perception import CameraIntrinsics, BinaryImage, DepthImage
import autolab_core.utils as utils
from autolab_core import Point, RigidTransform, YamlConfig
from meshpy import Mesh3D
import matplotlib.pyplot as plt
from dexnet.learning import TensorDataset
from dexnet.grasping import GraspableObject3D

"""
Script to render dexnet_2.0 dataset from oblique view, reproject into 3D, apply transformation to overhead camera at given distance
and project back into depth image. 
"""
class GraspInfo(object):
	def __init__(self,grasp,collision_free,contact_points,phi=0.0):
		self.grasp = grasp
		self.collision_free = collision_free
		self.contact_point1 = contact_points[0]
		self.contact_point2 = contact_points[1]
		self.phi = phi

class Reprojection():
	def __init__(self,path,elev):
		table_file = './data/meshes/table.obj'
		self.table_mesh = ObjFile(table_file).read()
		self.data_dir = './data/meshes/dexnet/'
#		self.data_dir = './data/meshes/mug/'
		self.output_dir = './data/rprojection_test'
		self.config = YamlConfig('./cfg/tools/generate_projected_gqcnn_dataset.yaml')
		
		self.random_positions = 1
		self.image_size = (300,300)
		self.elev = 0
		self.show_images = False

		self.cur_obj_label = 0
		self.cur_image_label = 0
		self.cur_pose_label = 0

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

		first_run=True

		for object_id in self.all_objects:
			self._load_data(object_id)
			for i, stable_pose in enumerate(self.stable_poses):
				try:
					candidate_grasp_info = self.candidate_grasps_dict[stable_pose.id]
				except KeyError:
					continue

				if not candidate_grasp_info:
					Warning("Candidate grasp info of object id %s empty"%object_id)
					Warning("Continue.")
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
#				print("Rendered sample")
				#for render_sample in render_samples:

				binary_im = render_sample.renders[RenderMode.SEGMASK].image
				depth_im = render_sample.renders[RenderMode.DEPTH_SCENE].image.crop(300,300)
				orig_im = Image.fromarray(self._scale_image(depth_im.data))
				if self.show_images:
					orig_im.show()
				orig_im.convert('RGB').save('./data/reprojection/'+object_id+'_elev_'+str(self.elev)+'_original.png')
				print("Saved original")

				T_stp_camera = render_sample.camera.object_to_camera_pose
				shifted_camera_intr = render_sample.camera.camera_intr.crop(300,300,240,320)
				depth_points = self._reproject_to_3D(depth_im,shifted_camera_intr)

				transformed_points,T_camera = self._transformation(depth_points)
#				print("T_camera:",T_camera)

				camera_dir = np.dot(T_camera.rotation,np.array([0.0,0.0,-1.0]))

				pcd = o3d.geometry.PointCloud()
#				print(camera_dir)
				pcd.points = o3d.utility.Vector3dVector(transformed_points.T)
				#TODO check normals!!
#				pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#				pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
				normals = np.repeat([camera_dir],len(transformed_points.T),axis=0)
				pcd.normals = o3d.utility.Vector3dVector(normals)

				if False:
					cs_points = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
					cs_lines = [[0,1],[0,2],[0,3]]
					colors = [[1,0,0],[0,1,0],[0,0,1]]
					cs = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(cs_points),lines=o3d.utility.Vector2iVector(cs_lines))
					cs.colors = o3d.utility.Vector3dVector(colors)
					o3d.visualization.draw_geometries([pcd])
	
#				mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,\
#													o3d.utility.DoubleVector([0.0005,0.001])) #0.0015
				depth = self._o3d_meshing(pcd)

#				projected_depth_im,new_camera_intr,table_height = self._projection(new_points,shifted_camera_intr)
				new_camera_intr = shifted_camera_intr
				new_camera_intr.cx=150
				new_camera_intr.cy=150
				print(depth)
				projected_depth_im = np.asarray(depth)
				projected_depth_im[projected_depth_im==0.0] = -1.0
				table_height = np.median(projected_depth_im[projected_depth_im!=-1.0].flatten())
				print("Minimum depth:",min(projected_depth_im.flatten()))
				print("Maximum depth:",max(projected_depth_im.flatten()))

				im = Image.fromarray(self._scale_image(projected_depth_im))

				projected_depth_im = DepthImage(projected_depth_im,frame='new_camera')

				cx = projected_depth_im.center[1]
				cy = projected_depth_im.center[0]

				# Grasp conversion
				T_obj_old_camera = T_stp_camera * T_obj_stp.as_frames('obj',T_stp_camera.from_frame)
				T_obj_camera = T_camera.dot(T_obj_old_camera)
				print("Candidate grasp info:",candidate_grasp_info)
				for grasp_info in candidate_grasp_info:
					grasp = grasp_info.grasp
					collision_free = grasp_info.collision_free

					grasp_2d = grasp.project_camera(T_obj_camera,new_camera_intr)
					dx = cx - grasp_2d.center.x
					dy = cy - grasp_2d.center.y
					translation = np.array([dy,dx])

					# Project 3D old_camera_cs contact points into new camera cs

					contact_points = np.append(grasp_info.contact_point1,1).T
					new_cam = np.dot(T_obj_camera.matrix,contact_points)
					c1 = new_camera_intr.project(Point(new_cam[0:3],frame=new_camera_intr.frame))
					contact_points = np.append(grasp_info.contact_point2,1).T
					new_cam = np.dot(T_obj_camera.matrix,contact_points)
					c2 = new_camera_intr.project(Point(new_cam[0:3],frame=new_camera_intr.frame))

					# Check if there are occlusions at contact points
					if projected_depth_im.data[c1.x,c1.y] == -1.0 or  projected_depth_im.data[c2.x,c2.y] == -1.0:
						print("Contact point at occlusion")
						contact_occlusion = True
					else:
						contact_occlusion = False
					# Mark contact points in image
					im = im.convert('RGB')
					if False:
						im_draw = ImageDraw.Draw(im)
						im_draw.line([(c1[0],c1[1]-10),(c1[0],c1[1]+10)],fill=(255,0,0,255))
						im_draw.line([(c1[0]-10,c1[1]),(c1[0]+10,c1[1])],fill=(255,0,0,255))
						im_draw.line([(c2[0],c2[1]-10),(c2[0],c2[1]+10)],fill=(255,0,0,255))
						im_draw.line([(c2[0]-10,c2[1]),(c2[0]+10,c2[1])],fill=(255,0,0,255))
					if self.show_images:
						im.show()
					im.save('./data/reprojection/'+object_id+'_elev_'+str(self.elev)+'_reprojected.png')

					# Transform and crop image

					depth_im_tf = projected_depth_im.transform(translation,grasp_2d.angle)
					depth_im_tf = depth_im_tf.crop(96,96)

					# Apply transformation to contact points
					trans_map = np.array([[1,0,dx],[0,1,dy]])
					rot_map = cv2.getRotationMatrix2D((cx,cy),np.rad2deg(grasp_2d.angle),1)
					trans_map_aff = np.r_[trans_map,[[0,0,1]]]
					rot_map_aff = np.r_[rot_map,[[0,0,1]]]
					full_map = rot_map_aff.dot(trans_map_aff)
#					print("Full map",full_map)
					c1_rotated = (np.dot(full_map,np.r_[c1.vector,[1]]) - np.array([150-48,150-48,0]))/3
					c2_rotated = (np.dot(full_map,np.r_[c2.vector,[1]]) - np.array([150-48,150-48,0]))/3
#					print("C1",c1_rotated)
#					print("C2",c2_rotated)

					grasp_line = depth_im_tf.data[48]
					occlusions = len(np.where(np.squeeze(grasp_line)==-1)[0])

					# Set occlusions to table height for resizing image
					depth_im_tf.data[depth_im_tf.data == -1.0] = table_height

					depth_im_tf_table = depth_im_tf.resize((32,32,),interp='bilinear')

					im = Image.fromarray(self._scale_image(depth_im_tf_table.data)).convert('RGB')
					draw = ImageDraw.Draw(im)
					draw.line([(c1_rotated[0],c1_rotated[1]-3),(c1_rotated[0],c1_rotated[1]+3)],fill=(255,0,0,255))
					draw.line([(c1_rotated[0]-3,c1_rotated[1]),(c1_rotated[0]+3,c1_rotated[1])],fill=(255,0,0,255))
					draw.line([(c2_rotated[0],c2_rotated[1]-3),(c2_rotated[0],c2_rotated[1]+3)],fill=(255,0,0,255))
					draw.line([(c2_rotated[0]-3,c2_rotated[1]),(c2_rotated[0]+3,c2_rotated[1])],fill=(255,0,0,255))
					if self.show_images:
						im.show()
					im.save('./data/reprojection/'+object_id+'_elev_'+str(self.elev)+'_transformed.png')

					hand_pose = np.r_[grasp_2d.center.y,
								grasp_2d.center.x,
								grasp_2d.depth,
								grasp_2d.angle,
								grasp_2d.center.y - new_camera_intr.cy,
								grasp_2d.center.x - new_camera_intr.cx,
								grasp_2d.width_px/3]

					self.tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
					self.tensor_datapoint['hand_poses'] = hand_pose
					self.tensor_datapoint['obj_labels'] = self.cur_obj_label
					self.tensor_datapoint['collision_free'] = collision_free
					self.tensor_datapoint['pose_labels'] = self.cur_pose_label
					self.tensor_datapoint['image_labels'] = self.cur_image_label
					self.tensor_datapoint['files'] = [self.tensor,self.array]
					self.tensor_datapoint['occlusions'] = occlusions
					self.tensor_datapoint['contact_occlusion'] = contact_occlusion

					for metric_name,metric_val in self.grasp_metrics[str(grasp.id)].iteritems():
						coll_free_metric = (1*collision_free) * metric_val
						self.tensor_datapoint[metric_name] = coll_free_metric
					self.tensor_dataset.add(self.tensor_datapoint)
					print("Saved dataset point")
					self.cur_image_label += 1
				self.cur_pose_label += 1
				gc.collect()
			self.cur_obj_label += 1
			# BREAK
#			break
	
		self.tensor_dataset.flush()

	def _o3d_meshing(self,pcd):
		mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=15)
		densities = np.asarray(densities)
		if False:
			print('visualize densities')
			densities = np.asarray(densities)
			density_colors = plt.get_cmap('plasma')(
				(densities - densities.min()) / (densities.max() - densities.min()))
			density_colors = density_colors[:, :3]
			density_mesh = o3d.geometry.TriangleMesh()
			density_mesh.vertices = mesh.vertices
			density_mesh.triangles = mesh.triangles
			density_mesh.triangle_normals = mesh.triangle_normals
			density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
			o3d.visualization.draw_geometries([density_mesh])
		vertices_to_remove = densities < 7.0 # np.quantile(densities, 0.01)
		mesh.remove_vertices_by_mask(vertices_to_remove)
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color([0.6,0.6,0.6])

#		o3d.visualization.draw_geometries([mesh])
		vis = o3d.visualization.Visualizer()
		vis.create_window(height=300,width=300,visible=False)
		vis.get_render_option().load_from_json("./data/renderoption.json")
		vis.add_geometry(mesh)
		vic = vis.get_view_control()
		params = vic.convert_to_pinhole_camera_parameters()
		(fx,fy) = params.intrinsic.get_focal_length()
		(cx,cy) = params.intrinsic.get_principal_point()
		params.intrinsic.set_intrinsics(300,300,525,525,cx,cy)
		params.extrinsic = np.array([[1,0,0,0], \
						[0,1,0,0], \
						[0,0,1,0], \
						[0,0,0,1]])
		vic.convert_from_pinhole_camera_parameters(params)
		vis.poll_events()
		vis.update_renderer()
		depth = vis.capture_depth_float_buffer(do_render=True)
#		vis.destroy_window()
#		del vis
		return depth
		
	
	def _projection(self,transformed_points,camera_intr):
		# Use Camera intrinsics
		new_camera_intr =  camera_intr
		new_camera_intr.cx = 150
		new_camera_intr.cy = 150
		K = new_camera_intr.proj_matrix
		
		projected_points = np.dot(K,transformed_points)

		point_depths = projected_points[2,:]
		table = np.median(point_depths)
		point_z = np.tile(point_depths,[3,1])
		points_proj = np.divide(projected_points,point_z)
		
		# Rounding
		points_proj = np.round(points_proj)
		points_proj = points_proj[:2,:].astype(np.int16)
		
		valid_ind = np.where((points_proj[0,:] >= 0) &\
					(points_proj[0,:] < self.image_size[0]) &\
					(points_proj[1,:] >= 0) &\
					(points_proj[1,:] < self.image_size[0]))[0]
		# Fill new image with NaN's
		fill_value = -1.0
		depth_data = np.full([self.image_size[0],self.image_size[1]],fill_value)
		for ind in valid_ind:
			prev_depth = depth_data[points_proj[1,ind],points_proj[0,ind]]
			if prev_depth == fill_value or prev_depth >= point_depths[ind]:
				depth_data[points_proj[1,ind],points_proj[0,ind]] = point_depths[ind]

		return depth_data, new_camera_intr, table

		

	def _transformation(self,points):
		# Points are given in camera frame. Transform to new camera frame!
		camera_new_position = np.array([[0],[0],[0]])

		ang = np.deg2rad(self.elev)
		Rot = np.array([[1, 0, 0],\
				[0, np.cos(ang),-np.sin(ang)],\
				[0, np.sin(ang),np.cos(ang)]])
		dist = self.config['env_rv_params']['min_radius'] * np.sin(ang)
		height = self.config['env_rv_params']['min_radius'] - self.config['env_rv_params']['min_radius'] * np.cos(ang) 
		# Rotation around x axis, therefore translation back to object center alongside y axis
		trans = np.array([0,dist,height])
		Rt = np.column_stack((Rot,trans))
		# Apply transformation to homogeneous coordinates of the points
		homogeneous_points = np.append(np.transpose(points),np.ones((1,len(points))),axis=0)
		transformed_points = np.dot(Rt,homogeneous_points)
		return transformed_points,RigidTransform(rotation=Rot,translation=trans,from_frame='camera',to_frame='new_camera')

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

	def _scale_image(self,depth):
		size = depth.shape
		flattend = depth.flatten()
		scaled = np.interp(flattend,(0.5,0.75),(0,255))
		integ = scaled.astype(np.uint8)
		integ.resize(size)
		return integ

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str,default=None)
	parser.add_argument('--elev',type=int,default=None,help='Elevation angle between camera z-axis and vertical axis')
	
	args = parser.parse_args()
	path = args.dir
	elev = args.elev
	
	reprojection = Reprojection(path,elev)
	reprojection.start_rendering()
