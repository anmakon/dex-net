import numpy as np
import gc
import os
import logging

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, RenderMode, SceneObject
from autolab_core import RigidTransform, YamlConfig
from dexnet.learning import TensorDataset
from PIL import Image
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

""" Script to generate an tensor dataset with rotated images (grasp aligned with image x-axis) and unrotated images. """

DATA_DIR = '/data'
DATASET_DIR = DATA_DIR + '/20210702_Rot_DexNet/'
DATASET_DIR_MOD = DATA_DIR + '/20210702_Unrot_DexNet/'


class GenerateBalancedObliqueDataset:
    def __init__(self, output_dir, output_dir_mod):
        self.NUM_OBJECTS = None
        self.table_file = DATA_DIR + '/meshes/table.obj'
        self.data_dir = DATA_DIR + '/meshes/dexnet/'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_dir_mod):
            os.mkdir(output_dir_mod)

        self.config = YamlConfig('./cfg/tools/generate_oblique_gqcnn_dataset.yaml')

        self.phi_offsets = self._generate_phi_offsets()
        self.datasets, self.target_object_keys = self._load_datasets()
        self.tensor_dataset = TensorDataset(output_dir, self._set_tensor_config())
        self.modified_tensor_dataset = TensorDataset(output_dir_mod, self._set_tensor_config())
        self.tensor_datapoint = self.tensor_dataset.datapoint_template
        self.gripper = self._set_gripper()
        self._table_mesh_filename = self._set_table_mesh_filename()
        self.table_mesh = self._set_table_mesh()

        self.cur_pose_label = 0
        self.cur_obj_label = 0
        self.cur_image_label = 0

        self.obj = None
        self.T_obj_camera = None

    def _camera_configs(self):
        return self.config['env_rv_params'].copy()

    @property
    def _camera_intr_scale(self):
        return 32.0 / 96.0

    @property
    def _render_modes(self):
        return [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]

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

    @property
    def _stable_pose_min_p(self):
        return self.config['stable_pose_min_p']

    def _set_table_mesh_filename(self):
        table_mesh_filename = self.config['collision_checking']['table_mesh_filename']
        if not os.path.isabs(table_mesh_filename):
            return os.path.join(DATA_DIR, table_mesh_filename)
        return table_mesh_filename

    def _set_table_mesh(self):
        return ObjFile(self._table_mesh_filename).read()

    def _set_gripper(self):
        return RobotGripper.load(self.config['gripper'])

    def _load_datasets(self):
        database = Hdf5Database(self.config['database_name'], access_level=READ_ONLY_ACCESS)
        target_object_keys = self.config['target_objects']
        dataset_names = target_object_keys.keys()
        datasets = [database.dataset(dn) for dn in dataset_names]
        if self.NUM_OBJECTS is not None:
            datasets = [dataset.subset(0, self.NUM_OBJECTS) for dataset in datasets]
        return datasets, target_object_keys

    def _set_tensor_config(self):
        tensor_config = self.config['tensors']
        tensor_config['fields']['depth_ims_tf_table']['height'] = 32
        tensor_config['fields']['depth_ims_tf_table']['width'] = 32
        tensor_config['fields']['obj_masks']['height'] = 32
        tensor_config['fields']['obj_masks']['width'] = 32
        tensor_config['fields']['robust_ferrari_canny'] = {}
        tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
        return tensor_config

    def render_images(self, scene_objs, stable_pose, num_images, camera_config=None):
        if camera_config is None:
            camera_config = self.config['env_rv_params']
        urv = UniformPlanarWorksurfaceImageRandomVariable(self.obj.mesh,
                                                          self._render_modes,
                                                          'camera',
                                                          camera_config,
                                                          scene_objs=scene_objs,
                                                          stable_pose=stable_pose)
        # Render images
        render_samples = urv.rvs(size=num_images)
        return render_samples

    def align_grasps(self, grasps):
        z_axis_in_obj = np.dot(self.T_obj_camera.inverse().matrix, np.array((0, 0, -1, 1)).reshape(4, 1))
        z_axis = z_axis_in_obj[0:3].squeeze() / np.linalg.norm(z_axis_in_obj[0:3].squeeze())
        aligned_grasps = [grasp.perpendicular_table(z_axis) for grasp in grasps]
        return aligned_grasps

    def get_camera_pose(self, sample):
        return np.r_[sample.camera.radius,
                     sample.camera.elev,
                     sample.camera.az,
                     sample.camera.roll,
                     sample.camera.focal,
                     sample.camera.tx,
                     sample.camera.ty]

    def is_grasp_aligned(self, aligned_grasp):
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
        cropped_im = image.crop(96, 96)
        resized_im = Image.fromarray(np.asarray(cropped_im.data)).resize((32, 32), resample=Image.BILINEAR)
        final_im = np.asarray(resized_im).reshape(32, 32, 1)
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

    def prepare_images(self, depth_im_table, binary_im, grasp_2d, cx, cy):
        # Get translation image distances to grasp
        dx = cx - grasp_2d.center.x
        dy = cy - grasp_2d.center.y
        translation = np.array([dy, dx])

        # Transform, crop and resize image
        binary_im_tf_table = binary_im.transform(translation, grasp_2d.angle)
        depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

        binary_im_tf = self._crop_and_resize(binary_im_tf_table)
        depth_im_tf = self._crop_and_resize(depth_im_tf_table)

        # Translate, crop and resize (no rotation)
        binary_unrotated_im_tf_table = binary_im.transform(translation, 0.0)
        depth_unrotated_im_tf_table = depth_im_table.transform(translation, 0.0)

        binary_unrotated_im_tf = self._crop_and_resize(binary_unrotated_im_tf_table)
        depth_unrotated_im_tf = self._crop_and_resize(depth_unrotated_im_tf_table)

        # self._show_image(depth_im_tf)
        # self._show_image(depth_unrotated_im_tf)

        return depth_im_tf, binary_im_tf, depth_unrotated_im_tf, binary_unrotated_im_tf

    def add_datapoint(self, depth_im_tf, binary_im_tf, depth_im_no_rot, binary_im_no_rot,
                      hand_pose, collision_free, camera_pose, aligned_grasp, grasp_metrics):
        # Add data to tensor dataset
        self.tensor_datapoint['depth_ims_tf_table'] = depth_im_tf
        self.tensor_datapoint['obj_masks'] = binary_im_tf
        self.tensor_datapoint['hand_poses'] = hand_pose
        self.tensor_datapoint['collision_free'] = collision_free
        self.tensor_datapoint['camera_poses'] = camera_pose

        # Add metrics to tensor dataset
        self.tensor_datapoint['robust_ferrari_canny'] = (1 * collision_free) * \
                                                        grasp_metrics[aligned_grasp.id]['robust_ferrari_canny']
        self.tensor_dataset.add(self.tensor_datapoint)

        # Add data to tensor dataset
        self.tensor_datapoint['depth_ims_tf_table'] = depth_im_no_rot
        self.tensor_datapoint['obj_masks'] = binary_im_no_rot

        # Add metrics to tensor dataset
        self.modified_tensor_dataset.add(self.tensor_datapoint)

    def save_samples(self, sample, grasps, T_obj_stp, collision_checker, grasp_metrics):
        self.tensor_datapoint['image_labels'] = self.cur_image_label
        T_stp_camera = sample.camera.object_to_camera_pose
        self.T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)
        aligned_grasps = self.align_grasps(grasps)
        binary_im = sample.renders[RenderMode.SEGMASK].image
        depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image
        camera_pose = self.get_camera_pose(sample)
        shifted_camera_intr = sample.camera.camera_intr

        cx = depth_im_table.center[1]
        cy = depth_im_table.center[0]

        for cnt, aligned_grasp in enumerate(aligned_grasps):
            if not self.is_grasp_aligned(aligned_grasp):
                continue
            collision_free = self.is_grasp_collision_free(aligned_grasp, collision_checker)

            # Project grasp coordinates in image
            grasp_2d = aligned_grasp.project_camera(self.T_obj_camera, shifted_camera_intr)

            depth_im_tf, binary_im_tf, depth_im_no_rot, binary_im_no_rot = self.prepare_images(depth_im_table,
                                                            binary_im,
                                                            grasp_2d,
                                                            cx,
                                                            cy)

            hand_pose = self.get_hand_pose(grasp_2d, shifted_camera_intr)

            self.add_datapoint(depth_im_tf, binary_im_tf, depth_im_no_rot, binary_im_no_rot, hand_pose,
                               collision_free, camera_pose, aligned_grasp, grasp_metrics)
        self.cur_image_label += 1

    def render_data(self):
        logging.basicConfig(level=logging.WARNING)
        for dataset in self.datasets:
            logging.info('Generating data for dataset %s' % dataset.name)
            object_keys = dataset.object_keys

            for obj_key in object_keys:
                if self.cur_obj_label % 10 == 0:
                    logging.info("Object number: %d" % self.cur_obj_label)
                self.obj = dataset[obj_key]
                self.tensor_datapoint['obj_labels'] = self.cur_obj_label

                grasps = dataset.grasps(self.obj.key, gripper=self.gripper.name)

                # Load grasp metrics
                grasp_metrics = dataset.grasp_metrics(self.obj.key,
                                                      grasps,
                                                      gripper=self.gripper.name)

                # setup collision checker
                collision_checker = GraspCollisionChecker(self.gripper)
                collision_checker.set_graspable_object(self.obj)

                # read in the stable poses of the mesh
                stable_poses = dataset.stable_poses(self.obj.key)

                # Iterate through stable poses
                for i, stable_pose in enumerate(stable_poses):
                    if not stable_pose.p > self._stable_pose_min_p:
                        continue
                    self.tensor_datapoint['pose_labels'] = self.cur_pose_label
                    # setup table in collision checker
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_table = self.obj.mesh.get_T_surface_obj(T_obj_stp,
                                                                  delta=self._table_offset).as_frames('obj', 'table')
                    T_table_obj = T_obj_table.inverse()
                    T_obj_stp = self.obj.mesh.get_T_surface_obj(T_obj_stp)

                    collision_checker.set_table(self._table_mesh_filename, T_table_obj)

                    # sample images from random variable
                    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                    scene_objs = {'table': SceneObject(self.table_mesh, T_table_obj)}

                    # Set up image renderer
                    samples = self.render_images(scene_objs,
                                                 stable_pose,
                                                 self.config['images_per_stable_pose'],
                                                 camera_config=self._camera_configs())
                    for sample in samples:
                        self.save_samples(sample, grasps, T_obj_stp, collision_checker, grasp_metrics)
                    self.cur_pose_label += 1
                    # next stable pose
                self.cur_obj_label += 1
                gc.collect()
                # next object
        # Save dataset
        self.tensor_dataset.flush()


if __name__ == "__main__":
    Generator = GenerateBalancedObliqueDataset(DATASET_DIR, DATASET_DIR_MOD)
    Generator.render_data()
