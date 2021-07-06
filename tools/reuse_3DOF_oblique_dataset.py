import numpy as np
import gc
import os
import logging
from tqdm import tqdm

from meshpy import ObjFile, StablePoseFile
from autolab_core import RigidTransform, YamlConfig
from dexnet.learning import TensorDataset
from perception import CameraIntrinsics, DepthImage
from PIL import Image
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

""" Script to reuse image information saved by the generate_3DOF_oblique_dataset.py to generate a rotated (grasp
    aligned with image x-axis) and unrotated tensordataset. Spares double-rendering of images. """

DATA_DIR = '/data'
UNROT_DATASET_DIR = DATA_DIR + '/Unrot_3DOF_Viewpoint_DexNet/'
ROT_DATASET_DIR = DATA_DIR + '/Rot_3DOF_Viewpoint_DexNet/'
DATA_ORIGIN_DIR = DATA_DIR + '/3DOF_Viewpoint_DexNet/images/'
CONFIG = './cfg/tools/regenerate_oblique_gqcnn_dataset.yaml'
VISUALISE_3D = False
MODE = 'both'    # dist dist_rot both


class GenerateBalancedObliqueDataset:
    def __init__(self):
        self.data_dir = DATA_DIR + '/meshes/dexnet/'
        if MODE != 'dist':
            if not os.path.exists(UNROT_DATASET_DIR):
                os.mkdir(UNROT_DATASET_DIR)
        if MODE != 'dist_rot':
            if not os.path.exists(ROT_DATASET_DIR):
                os.mkdir(ROT_DATASET_DIR)
        self.config = YamlConfig(CONFIG)

        self.phi_offsets = self._generate_phi_offsets()
        self.datasets, self.target_object_keys = self._load_datasets()
        if MODE == 'dist':
            self.rot_tensor_dataset = TensorDataset(ROT_DATASET_DIR, self._set_tensor_config())
        elif MODE == 'dist_rot':
            self.tensor_dataset = TensorDataset(UNROT_DATASET_DIR, self._set_tensor_config())
        elif MODE == 'both':
            self.tensor_dataset = TensorDataset(UNROT_DATASET_DIR, self._set_tensor_config())
            self.rot_tensor_dataset = TensorDataset(ROT_DATASET_DIR, self._set_tensor_config())
        else:
            raise ValueError

        self.tensor_datapoint = self.tensor_dataset.datapoint_template
        self.gripper = self._set_gripper()
        self._table_mesh_filename = self._set_table_mesh_filename()
        self.table_mesh = self._set_table_mesh()

        self.cur_pose_label = -1
        self.cur_obj_label = -1
        self.cur_image_label = -1

        self.obj = None
        self.T_obj_camera = None

    def _camera_configs(self):
        return self.config['env_rv_params'].copy()

    @property
    def _camera_intr_scale(self):
        return 32.0 / 96.0

    @property
    def _max_grasp_approach_offset(self):
        return np.deg2rad(self.config['table_alignment']['max_approach_offset'])

    @property
    def _min_grasp_approach_offset(self):
        return -np.deg2rad(self.config['table_alignment']['max_approach_offset'])

    @property
    def _max_grasp_approach_table_angle(self):
        return np.deg2rad(self.config['table_alignment']['max_approach_table_angle'])

    @property
    def _num_grasp_approach_samples(self):
        return self.config['table_alignment']['num_approach_offset_samples']

    def _generate_phi_offsets(self):
        phi_offsets = []
        if self._max_grasp_approach_offset == self._min_grasp_approach_offset:
            phi_inc = 1
        elif self._num_grasp_approach_samples == 1:
            phi_inc = self._max_grasp_approach_offset - self._min_grasp_approach_offset + 1
        else:
            phi_inc = (self._max_grasp_approach_offset - self._min_grasp_approach_offset) / \
                      (self._num_grasp_approach_samples - 1)

        phi = self._min_grasp_approach_offset
        while phi <= self._max_grasp_approach_offset:
            phi_offsets.append(phi)
            phi += phi_inc
        return phi_offsets

    @property
    def _approach_dist(self):
        return self.config['collision_checking']['approach_dist']

    @property
    def _delta_approach(self):
        return self.config['collision_checking']['delta_approach']

    @property
    def _table_offset(self):
        return self.config['collision_checking']['table_offset']

    def _set_gripper(self):
        return RobotGripper.load(self.config['gripper'])

    def _set_table_mesh_filename(self):
        table_mesh_filename = self.config['collision_checking']['table_mesh_filename']
        if not os.path.isabs(table_mesh_filename):
            return os.path.join(DATA_DIR, table_mesh_filename)
        return table_mesh_filename

    def _set_table_mesh(self):
        return ObjFile(self._table_mesh_filename).read()

    def _load_datasets(self):
        database = Hdf5Database(self.config['database_name'], access_level=READ_ONLY_ACCESS)
        target_object_keys = self.config['target_objects']
        dataset_names = target_object_keys.keys()
        datasets = [database.dataset(dn) for dn in dataset_names]
        return datasets, target_object_keys

    def _set_tensor_config(self):
        tensor_config = self.config['tensors']
        tensor_config['fields']['depth_ims_tf_table']['height'] = self.config['gqcnn']['final_height']
        tensor_config['fields']['depth_ims_tf_table']['width'] = self.config['gqcnn']['final_width']
        tensor_config['fields']['robust_ferrari_canny'] = {}
        tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
        tensor_config['fields']['ferrari_canny'] = {}
        tensor_config['fields']['ferrari_canny']['dtype'] = 'float32'
        tensor_config['fields']['force_closure'] = {}
        tensor_config['fields']['force_closure']['dtype'] = 'float32'
        return tensor_config

    def align_grasps_with_camera(self, grasps):
        z_axis_in_obj = np.dot(self.T_obj_camera.inverse().matrix, np.array((0, 0, -1, 1)).reshape(4, 1))
        z_axis = z_axis_in_obj[0:3].squeeze() / np.linalg.norm(z_axis_in_obj[0:3].squeeze())
        aligned_grasps = [grasp.perpendicular_table(z_axis) for grasp in grasps]
        return aligned_grasps

    def is_grasp_aligned(self, aligned_grasp, stable_pose=None):
        if stable_pose is not None:
            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
        else:
            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_camera_z(self.T_obj_camera)
        perpendicular_table = (np.abs(grasp_approach_table_angle) < self._max_grasp_approach_table_angle)
        if not perpendicular_table:
            return False
        return True

    def is_grasp_collision_free(self, grasp, collision_checker):
        collision_free = False
        for phi_offset in self.phi_offsets:
            grasp.grasp_y_axis_offset(phi_offset)
            collides = collision_checker.collides_along_approach(grasp, self._approach_dist,
                                                                 self._delta_approach)
            if not collides:
                collision_free = True
                break
        return collision_free

    def get_hand_pose(self, grasp_2d, shifted_camera_intr):
        # Configure hand pose
        return np.r_[grasp_2d.center.y,
                     grasp_2d.center.x,
                     grasp_2d.depth,
                     grasp_2d.angle,
                     grasp_2d.center.y - shifted_camera_intr.cy,
                     grasp_2d.center.x - shifted_camera_intr.cx,
                     grasp_2d.width_px / 3]

    def _crop_and_resize(self, image):
        cropped_im = image.crop(self.config['gqcnn']['crop_height'], self.config['gqcnn']['crop_width'])
        resized_im = Image.fromarray(np.asarray(cropped_im.data)).resize((self.config['gqcnn']['final_height'],
                                                                          self.config['gqcnn']['final_width']),
                                                                         resample=Image.BILINEAR)
        final_im = np.asarray(resized_im).reshape(self.config['gqcnn']['final_height'],
                                                  self.config['gqcnn']['final_width'],
                                                  1)
        return final_im

    @staticmethod
    def scale(array):
        size = array.shape
        flattend = array.flatten()
        scaled = np.interp(flattend, (min(flattend), max(flattend)), (0, 255), left=0, right=255)
        integ = scaled.astype(np.uint8)
        integ.resize(size)
        return integ.squeeze()

    def _show_image(self, image):
        scaled_image = self.scale(image)
        im = Image.fromarray(scaled_image).resize((300, 300), resample=Image.NEAREST)
        im.save('/data/sldfkhjsl.png')
        im.show()

    def prepare_images(self, depth_im_table, grasp_2d):
        # Get translation image distances to grasp
        dx = 150 - grasp_2d.center.x
        dy = 150 - grasp_2d.center.y
        translation = np.array([dy, dx])

        if MODE == 'dist':
            depth_rotated_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)
            depth_rotated_im_tf = self._crop_and_resize(depth_rotated_im_tf_table)
            return None, depth_rotated_im_tf
        elif MODE == 'dist_rot':
            depth_unrotated_im_tf_table = depth_im_table.transform(translation, 0.0)
            depth_unrotated_im_tf = self._crop_and_resize(depth_unrotated_im_tf_table)
            return depth_unrotated_im_tf, None
        elif MODE == 'both':
            depth_rotated_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)
            depth_rotated_im_tf = self._crop_and_resize(depth_rotated_im_tf_table)
            depth_unrotated_im_tf_table = depth_im_table.transform(translation, 0.0)
            depth_unrotated_im_tf = self._crop_and_resize(depth_unrotated_im_tf_table)
            return depth_unrotated_im_tf, depth_rotated_im_tf
        else:
            raise ValueError

    def add_datapoint(self, depth_im_no_rot, depth_im_rot, hand_pose,
                      collision_free, camera_pose, aligned_grasp, grasp_metrics):
        # Add data to tensor dataset

        self.tensor_datapoint['hand_poses'] = hand_pose
        self.tensor_datapoint['collision_free'] = collision_free
        self.tensor_datapoint['camera_poses'] = camera_pose

        # Add metrics to tensor dataset
        for metric_name, metric_val in grasp_metrics[aligned_grasp.id].iteritems():
            coll_free_metric = (1 * collision_free) * metric_val
            self.tensor_datapoint[metric_name] = coll_free_metric

        if MODE == 'dist':
            self.tensor_datapoint['depth_ims_tf_table'] = depth_im_no_rot
            self.tensor_dataset.add(self.tensor_datapoint)
        elif MODE == 'dist_rot':
            self.tensor_datapoint['depth_ims_tf_table'] = depth_im_rot
            self.rot_tensor_dataset.add(self.tensor_datapoint)
        elif MODE == 'both':
            self.tensor_datapoint['depth_ims_tf_table'] = depth_im_no_rot
            self.tensor_dataset.add(self.tensor_datapoint)
            self.tensor_datapoint['depth_ims_tf_table'] = depth_im_rot
            self.rot_tensor_dataset.add(self.tensor_datapoint)
        else:
            raise ValueError

    def save_images(self, depth_im, camera_pose, grasps, camera_intr, collision_checker, grasp_metrics):
        aligned_grasps = self.align_grasps_with_camera(grasps)
        # aligned_grasps = [grasp.perpendicular_table(self.stable_pose) for grasp in grasps]

        for cnt, aligned_grasp in enumerate(aligned_grasps):
            if not self.is_grasp_aligned(aligned_grasp):
                    continue
            # before = time.time()
            collision_free = self.is_grasp_collision_free(aligned_grasp, collision_checker)
            # Project grasp coordinates in image
            grasp_2d = aligned_grasp.project_camera(self.T_obj_camera, camera_intr)

            depth_im_no_rot, depth_im_rot = self.prepare_images(depth_im, grasp_2d)

            # T_grasp_camera = aligned_grasp.gripper_pose(self.gripper).inverse() * self.T_obj_camera.inverse()

            # vis.figure()
            # T_obj_world = vis.mesh_stable_pose(self.obj.mesh.trimesh,
            #                                    self.stable_pose.T_obj_world, style='surface', dim=0.5, plot_table=False)
            # # T_grasp_world = T_obj_world * aligned_grasp.gripper_pose(self.gripper)
            # T_camera_world = T_obj_world * self.T_obj_camera.inverse()
            # vis.gripper(self.gripper, aligned_grasp, T_obj_world, color=(0.3, 0.3, 0.3),
            #             T_camera_world=T_camera_world)
            # vis.show()

            hand_pose = self.get_hand_pose(grasp_2d, camera_intr)

            self.add_datapoint(depth_im_no_rot, depth_im_rot, hand_pose, collision_free,
                               camera_pose, aligned_grasp, grasp_metrics)
            # print("Grasp took: ", time.time() - before)

    def _read_image_ids(self):
        all_files = os.listdir(DATA_ORIGIN_DIR)
        ids = [fl[7:13] for fl in all_files if 'object' in fl]
        ids.sort()
        return ids

    def _read_data(self, id):
        depth_im = Image.open(DATA_ORIGIN_DIR + 'depth_' + id + '.tiff')
        depth_image = DepthImage(np.asarray(depth_im))
        camera_pose = np.loadtxt(DATA_ORIGIN_DIR + 'camera_pose_' + id + '.txt')
        self.T_obj_camera = RigidTransform.load(DATA_ORIGIN_DIR + 'camera_tf_' + id + '.tf')
        with open(DATA_ORIGIN_DIR + 'object_' + id + '.txt', 'r') as f:
            object_id = f.readline()
        stable_pose = StablePoseFile(DATA_ORIGIN_DIR + 'stable_pose_' + id + '.stp').read()[0]
        camera_intr = CameraIntrinsics.load(DATA_ORIGIN_DIR + 'camera_intr_' + id + '.intr')
        return depth_image, camera_pose, stable_pose, camera_intr, object_id

    def render_data(self):
        image_ids = self._read_image_ids()
        logging.basicConfig(level=logging.WARNING)
        self.obj_id = None
        self.stable_pose = None
        for image_id in tqdm(image_ids):
            depth_im, camera_pose, stable_pose, camera_intr, object_id = self._read_data(image_id)
            self.cur_image_label += 1
            self.tensor_datapoint['image_labels'] = self.cur_image_label
            if self.stable_pose is None or (stable_pose.x0 != self.stable_pose.x0).any():
                self.stable_pose = stable_pose
                self.cur_pose_label += 1
                self.tensor_datapoint['pose_labels'] = self.cur_pose_label
            if self.obj_id != object_id:
                self.cur_obj_label += 1
                self.tensor_datapoint['obj_labels'] = self.cur_obj_label
                try:
                    dataset = self.datasets[0]
                    self.obj = dataset[object_id]
                    self.obj_id = object_id
                except ValueError:
                    dataset = self.datasets[1]
                    self.obj = dataset[object_id]
                    self.obj_id = object_id

            grasps = dataset.grasps(self.obj.key, gripper=self.gripper.name)

            # Load grasp metrics
            grasp_metrics = dataset.grasp_metrics(self.obj.key,
                                                  grasps,
                                                  gripper=self.gripper.name)

            # setup collision checker
            collision_checker = GraspCollisionChecker(self.gripper)
            collision_checker.set_graspable_object(self.obj)

            # setup table in collision checker
            T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
            T_obj_table = self.obj.mesh.get_T_surface_obj(T_obj_stp,
                                                          delta=self._table_offset).as_frames('obj', 'table')
            T_table_obj = T_obj_table.inverse()
            collision_checker.set_table(self._table_mesh_filename, T_table_obj)

            self.save_images(depth_im, camera_pose, grasps, camera_intr, collision_checker, grasp_metrics)
            gc.collect()

                # next object
        # Save dataset
        self.tensor_dataset.flush()
        self.rot_tensor_dataset.flush()


if __name__ == "__main__":

    Generator = GenerateBalancedObliqueDataset()
    Generator.render_data()
