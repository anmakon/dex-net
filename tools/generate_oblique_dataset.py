import numpy as np
import argparse
import gc
import os
import logging

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, RenderMode, SceneObject
from autolab_core import RigidTransform, YamlConfig
from dexnet.learning import TensorDataset
from dexnet.visualization import DexNetVisualizer3D as vis
from PIL import Image
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

"""
Script to render an oblique dexnet_2.0 dataset. Output directory can be specified with --dir, 
elevation angle (angle between z-axis and camera axis) can be specified with --elev. 
"""

DATA_DIR = '/data'
table_file = '/data/meshes/table.obj'

data_dir = '/data/meshes/dexnet/'
output_dir = DATA_DIR + '/oblique_testset/'

config = YamlConfig('./cfg/tools/generate_oblique_gqcnn_dataset.yaml')

NUM_OBJECTS = None
VISUALISE_3D = False
SAVE_DEPTH_IMAGES = False


class GraspInfo(object):
    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi


def _scale_image(depth):
    size = depth.shape
    flattend = depth.flatten()
    scaled = np.interp(flattend, (min(flattend), max(flattend)), (0, 255), left=0, right=255)
    integ = scaled.astype(np.uint8)
    integ.resize(size)
    return integ


parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--elev', type=int, default=None, help='Elevation angle between camera z-axis and vertical axis')

args = parser.parse_args()
path = args.dir
elev = args.elev
if path is not None:
    output_dir = '/data/' + path

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

logging.basicConfig(level=logging.WARNING)

config_filename = "./cfg/tools/generate_oblique_gqcnn_dataset.yaml"
output_path = DATA_DIR + "/meshes/test"

config = YamlConfig(config_filename)
database = Hdf5Database(config['database_name'],
                        access_level=READ_ONLY_ACCESS)

target_object_keys = config['target_objects']

# Setup grasp params
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
stable_pose_min_p = config['stable_pose_min_p']

table_mesh_filename = coll_check_params['table_mesh_filename']
if not os.path.isabs(table_mesh_filename):
    table_mesh_filename = os.path.join(DATA_DIR, table_mesh_filename)

dataset_names = target_object_keys.keys()
datasets = [database.dataset(dn) for dn in dataset_names]
if NUM_OBJECTS is not None:
    datasets = [dataset.subset(0, NUM_OBJECTS) for dataset in datasets]

gripper = RobotGripper.load(config['gripper'])

if elev is not None:
    print("Elevation angle is being set to %d" % elev)
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

tensor_dataset = TensorDataset(output_dir, tensor_config)
tensor_datapoint = tensor_dataset.datapoint_template
table_mesh = ObjFile(table_mesh_filename).read()

obj_category_map = {}
pose_category_map = {}

cur_pose_label = 0
cur_obj_label = 0
cur_image_label = 0

render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]
for dataset in datasets:
    logging.info('Generating data for dataset %s' % (dataset.name))

    # iterate through all object keys
    object_keys = dataset.object_keys

    for obj_key in object_keys:
        print("Object number: ", cur_obj_label)
        obj = dataset[obj_key]
        grasps = dataset.grasps(obj.key, gripper=gripper.name)

        # setup collision checker
        collision_checker = GraspCollisionChecker(gripper)
        collision_checker.set_graspable_object(obj)

        # read in the stable poses of the mesh
        stable_poses = dataset.stable_poses(obj.key)

        # Iterate through stable poses
        for i, stable_pose in enumerate(stable_poses):
            if not stable_pose.p > stable_pose_min_p:
                continue
            # add to category maps
            if obj.key not in obj_category_map.keys():
                obj_category_map[obj.key] = cur_obj_label
            pose_category_map['%s_%s' % (obj.key, stable_pose.id)] = cur_pose_label

            # setup table in collision checker
            T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
            T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
            T_table_obj = T_obj_table.inverse()
            T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

            collision_checker.set_table(table_mesh_filename, T_table_obj)

            # sample images from random variable
            T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
            scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

            # Set up image renderer
            urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                              render_modes,
                                                              'camera',
                                                              config['env_rv_params'],
                                                              scene_objs=scene_objs,
                                                              stable_pose=stable_pose)
            # Render images
            render_samples = urv.rvs(size=config['images_per_stable_pose'])
            for sample in render_samples:
                T_stp_camera = sample.camera.object_to_camera_pose
                T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)

                z_axis_in_obj = np.dot(T_obj_camera.inverse().matrix, np.array((0, 0, -1, 1)).reshape(4, 1))
                z_axis = z_axis_in_obj[0:3].squeeze() / np.linalg.norm(z_axis_in_obj[0:3].squeeze())

                aligned_grasps = [grasp.perpendicular_table(z_axis) for grasp in grasps]
                # aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

                binary_im = sample.renders[RenderMode.SEGMASK].image
                depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image

                shifted_camera_intr = sample.camera.camera_intr

                camera_pose = np.r_[sample.camera.radius,
                                      sample.camera.elev,
                                      sample.camera.az,
                                      sample.camera.roll,
                                      sample.camera.focal,
                                      sample.camera.tx,
                                      sample.camera.ty]

                cx = depth_im_table.center[1]
                cy = depth_im_table.center[0]

                # Compute camera intrinsics
                camera_intr_scale = 32.0 / 96.0
                cropped_camera_intr = shifted_camera_intr.crop(96, 96, cy, cx)
                final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)
                # Iterate through grasps
                for cnt, aligned_grasp in enumerate(aligned_grasps):
                    # _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                    _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_camera_z(T_obj_camera)
                    perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                    if not perpendicular_table:
                        continue

                    # check whether any valid approach directions are collision free
                    collision_free = False
                    for phi_offset in phi_offsets:
                        aligned_grasp.grasp_y_axis_offset(phi_offset)
                        collides = collision_checker.collides_along_approach(aligned_grasp, approach_dist,
                                                                             delta_approach)
                        if not collides:
                            collision_free = True
                            break

                    # Load grasp metrics
                    grasp_metrics = dataset.grasp_metrics(obj.key, aligned_grasps, gripper=gripper.name)

                    # Project grasp coordinates in image
                    grasp_2d = aligned_grasp.project_camera(T_obj_camera, shifted_camera_intr)

                    if VISUALISE_3D:
                        vis.figure()
                        T_obj_world = vis.mesh_stable_pose(obj.mesh.trimesh,
                                                           stable_pose.T_obj_world, style='surface', dim=0.5)
                        T_camera_world = T_obj_world * T_obj_camera.inverse()
                        vis.gripper(gripper, aligned_grasp, T_obj_world, color=(0.3, 0.3, 0.3),
                                    T_camera_world=T_camera_world)
                        vis.show()

                    # Get translation image distances to grasp
                    dx = cx - grasp_2d.center.x
                    dy = cy - grasp_2d.center.y
                    translation = np.array([dy, dx])

                    # Transform, crop and resize image
                    binary_im_tf = binary_im.transform(translation, grasp_2d.angle)
                    depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

                    binary_im_tf = binary_im_tf.crop(96, 96)
                    depth_im_tf_table = depth_im_tf_table.crop(96, 96)

                    depth_image = np.asarray(depth_im_tf_table.data)
                    dep_image = Image.fromarray(depth_image).resize((32, 32), resample=Image.BILINEAR)
                    depth_im_tf = np.asarray(dep_image).reshape(32, 32, 1)

                    binary_image = np.asarray(binary_im_tf.data)
                    bin_image = Image.fromarray(binary_image).resize((32, 32), resample=Image.BILINEAR)
                    binary_im_tf = np.asarray(bin_image).reshape(32, 32, 1)

                    # Save the depth image for debugging purposes
                    if SAVE_DEPTH_IMAGES:
                        scaled_depth_image = _scale_image(np.asarray(depth_im_tf_table.data))
                        scaled_dep_image = Image.fromarray(scaled_depth_image).resize((32, 32), resample=Image.BILINEAR)
                        scaled_dep_image.resize((300, 300), resample=Image.NEAREST) \
                            .save(output_dir + '/' + str(cur_image_label) + '_' + str(cnt) + '.png')

                    # Configure hand pose
                    hand_pose = np.r_[grasp_2d.center.y,
                                      grasp_2d.center.x,
                                      grasp_2d.depth,
                                      grasp_2d.angle,
                                      grasp_2d.center.y - shifted_camera_intr.cy,
                                      grasp_2d.center.x - shifted_camera_intr.cx,
                                      grasp_2d.width_px / 3]

                    # Add data to tensor dataset
                    tensor_datapoint['depth_ims_tf_table'] = depth_im_tf
                    tensor_datapoint['obj_masks'] = binary_im_tf
                    tensor_datapoint['hand_poses'] = hand_pose
                    tensor_datapoint['obj_labels'] = cur_obj_label
                    tensor_datapoint['collision_free'] = collision_free
                    tensor_datapoint['pose_labels'] = cur_pose_label
                    tensor_datapoint['image_labels'] = cur_image_label
                    tensor_datapoint['camera_poses'] = camera_pose

                    # Add metrics to tensor dataset
                    for metric_name, metric_val in grasp_metrics[aligned_grasp.id].iteritems():
                        coll_free_metric = (1 * collision_free) * metric_val
                        tensor_datapoint[metric_name] = coll_free_metric

                    tensor_dataset.add(tensor_datapoint)
                cur_image_label += 1
            cur_pose_label += 1
            gc.collect()
            # next stable pose
        cur_obj_label += 1
        # next object
# Save dataset
tensor_dataset.flush()
