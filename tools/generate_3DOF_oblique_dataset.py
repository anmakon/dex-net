import numpy as np
import gc
import os
import logging
import time

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, RenderMode, SceneObject, StablePoseFile
from autolab_core import RigidTransform, YamlConfig
from dexnet.learning import TensorDataset
from dexnet.visualization import DexNetVisualizer3D as vis
from PIL import Image
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

"""Script to generate a 4DOF grasp quality dataset from a DexNet mesh dataset."""

DATA_DIR = '/data'
DATASET_DIR = DATA_DIR + '/Test/'
CONFIG = './cfg/tools/generate_3DOF_gqcnn_dataset.yaml'
VISUALISE_3D = True


class GenerateBalancedObliqueDataset:
    def __init__(self, output_dir):
        self.NUM_OBJECTS = None
        self.table_file = DATA_DIR + '/meshes/table.obj'
        self.data_dir = DATA_DIR + '/meshes/dexnet/'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.image_dir = output_dir + '/images/'
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)

        self.config = YamlConfig(CONFIG)

        self.phi_offsets = self._generate_phi_offsets()
        self.datasets = self._load_datasets()
        self.tensor_dataset = TensorDataset(output_dir, self._set_tensor_config())
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
        """ Load the datasets in the Hdf5 database given through the config file.

            Returns
            -------
            datasets (list): list of datasets
        """
        database = Hdf5Database(self.config['database_name'], access_level=READ_ONLY_ACCESS)
        target_object_keys = self.config['target_objects']
        dataset_names = target_object_keys.keys()
        datasets = [database.dataset(dn) for dn in dataset_names]
        if self.NUM_OBJECTS is not None:
            datasets = [dataset.subset(0, self.NUM_OBJECTS) for dataset in datasets]
        return datasets

    def _set_tensor_config(self):
        """ Sets the tensor config based on the used config file.

            Returns
            -------
            tensor_config (dict): tensor config that can be used to initiate a tensor dataset.
        """
        tensor_config = self.config['tensors']
        tensor_config['fields']['depth_ims_tf_table']['height'] = self.config['gqcnn']['final_height']
        tensor_config['fields']['depth_ims_tf_table']['width'] = self.config['gqcnn']['final_width']
        tensor_config['fields']['obj_masks']['height'] = self.config['gqcnn']['final_height']
        tensor_config['fields']['obj_masks']['width'] = self.config['gqcnn']['final_width']
        tensor_config['fields']['robust_ferrari_canny'] = {}
        tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
        tensor_config['fields']['ferrari_canny'] = {}
        tensor_config['fields']['ferrari_canny']['dtype'] = 'float32'
        tensor_config['fields']['force_closure'] = {}
        tensor_config['fields']['force_closure']['dtype'] = 'float32'
        return tensor_config

    def render_images(self, scene_objs, stable_pose, num_images, camera_config=None):
        """ Renders depth and binary images from self.obj at the given stable pose. The camera
            sampling occurs within urv.rvs.

            Parameters
            ----------
            scene_objs (dict): Objects occuring in the scene, mostly includes the table mesh.
            stable_pose (StablePose): Stable pose of the object
            num_images (int): Numbers of images to render
            camera_config (dict): Camera sampling parameters with minimum/maximum values for radius, polar angle, ...

            Returns
            -------
            render_samples (list): list of rendered images including sampled camera positions
        """
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

    def align_grasps_with_camera(self, grasps):
        """ Attempts to align all grasps in grasps with the z axis of the camera in self.T_obj_camera.

            Parameters
            ----------
            grasps (list): List of grasps for the object mesh in the object frame.

            Returns
            -------
            aligned_grasps (list): List of aligned grasps for the object mesh in the object frame
        """
        z_axis_in_obj = np.dot(self.T_obj_camera.inverse().matrix, np.array((0, 0, -1, 1)).reshape(4, 1))
        z_axis = z_axis_in_obj[0:3].squeeze() / np.linalg.norm(z_axis_in_obj[0:3].squeeze())
        aligned_grasps = [grasp.perpendicular_table(z_axis) for grasp in grasps]
        return aligned_grasps

    def get_camera_pose(self, sample):
        """ Attempts to align all grasps in grasps with the z axis of the camera in self.T_obj_camera.

            Parameters
            ----------
            sample (meshpy rendersample): Image sample

            Returns
            -------
            camera pose (np.array): Array including the camera radius, elevation angle, polar angle, roll, focal length,
                                    and table translation in x and y.
        """
        return np.r_[sample.camera.radius,
                     sample.camera.elev,
                     sample.camera.az,
                     sample.camera.roll,
                     sample.camera.focal,
                     sample.camera.tx,
                     sample.camera.ty]

    def is_grasp_aligned(self, aligned_grasp, stable_pose=None):
        """ Checks if the grasp is aligned with the camera z axis or the z axis of the stable pose, if given.

            Parameters
            ----------
            aligned_grasp (Grasp)
            stable_pose (StablePose): stable pose of object. Default: None

            Returns
            -------
            Aligned (bool): True if grasp is aligned with the desired z axis.
        """
        if stable_pose is not None:
            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
        else:
            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_camera_z(self.T_obj_camera)
        perpendicular_table = (np.abs(grasp_approach_table_angle) < self._max_grasp_approach_table_angle)
        if not perpendicular_table:
            return False
        return True

    def is_grasp_collision_free(self, grasp, collision_checker):
        """ Checks if any of the +- phi grasp linear approaches are collision free with OpenRave collision checker.

            Parameters
            ----------
            grasp (Grasp)
            collision_checker (GraspCollisionChecker)

            Returns
            -------
            collision_free (bool): True if grasp is collision free.
        """
        collision_free = False
        for phi_offset in self.phi_offsets:
            grasp.grasp_y_axis_offset(phi_offset)
            collides = collision_checker.collides_along_approach(grasp, self._approach_dist,
                                                                 self._delta_approach)
            if not collides:
                collision_free = True
                break
        return collision_free

    def get_hand_pose(self, grasp_2d, grasp_3d):
        """ Organises numpy array for hand_pose tensor.

            Parameters
            ----------
            grasp_2d (Grasp2D)
            grasp_3d (Grasp3D)

            Returns
            -------
            hand_pose (np.array): Hand_pose tensor including distance between camera and grasp tcp and grasp rotation
                                    quaternion in camera frame.
        """
        return np.r_[grasp_2d.depth,
                     grasp_3d.quaternion]  # w x y z layout

    def _crop_and_resize(self, image):
        """ Crop and resize an image to the final height and width.

            Parameters
            ----------
            image (DepthImage)

            Returns
            -------
            final_im (np.array): cropped and resized according to crop_height/width and final_height/width in
                                self.config['gqcnn']
        """
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
        """ Scale image for visualisation purposes."""
        size = array.shape
        flattend = array.flatten()
        scaled = np.interp(flattend, (min(flattend), max(flattend)), (0, 255), left=0, right=255)
        integ = scaled.astype(np.uint8)
        integ.resize(size)
        return integ.squeeze()

    def _show_image(self, image):
        """ Show image by saving it in /data/test_image.png"""
        scaled_image = self.scale(image)
        im = Image.fromarray(scaled_image).resize((300, 300), resample=Image.NEAREST)
        im.save('/data/test_image.png')
        im.show()

    def prepare_images(self, depth_im_table, binary_im, grasp_2d):
        # Get translation image distances to grasp
        dx = 300 - grasp_2d.center.x
        dy = 300 - grasp_2d.center.y
        translation = np.array([dy, dx])

        # Translate, crop and resize (no rotation)
        binary_unrotated_im_tf_table = binary_im.transform(translation, 0.0)
        depth_unrotated_im_tf_table = depth_im_table.transform(translation, 0.0)

        binary_unrotated_im_tf = self._crop_and_resize(binary_unrotated_im_tf_table)
        depth_unrotated_im_tf = self._crop_and_resize(depth_unrotated_im_tf_table)

        # self._show_image(depth_im_tf)

        return depth_unrotated_im_tf, binary_unrotated_im_tf

    def add_datapoint(self, depth_im_no_rot, binary_im_no_rot, hand_pose,
                      collision_free, aligned_grasp, grasp_metrics):
        # Add data to tensor dataset
        self.tensor_datapoint['depth_ims_tf_table'] = depth_im_no_rot
        self.tensor_datapoint['obj_masks'] = binary_im_no_rot
        self.tensor_datapoint['hand_poses'] = hand_pose
        self.tensor_datapoint['collision_free'] = collision_free
        self.tensor_datapoint['image_labels'] = self.cur_image_label

        # Add metrics to tensor dataset
        for metric_name, metric_val in grasp_metrics[aligned_grasp.id].iteritems():
            coll_free_metric = (1 * collision_free) * metric_val
            self.tensor_datapoint[metric_name] = coll_free_metric

        self.tensor_dataset.add(self.tensor_datapoint)

    def save_samples(self, sample, grasps, T_obj_stp, collision_checker, grasp_metrics, stable_pose):
        T_stp_camera = sample.camera.object_to_camera_pose
        self.T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)
        aligned_grasps = self.align_grasps_with_camera(grasps)

        binary_im = sample.renders[RenderMode.SEGMASK].image
        depth_im_table = sample.renders[RenderMode.DEPTH_SCENE].image
        camera_pose = self.get_camera_pose(sample)
        self.tensor_datapoint['camera_poses'] = camera_pose
        shifted_camera_intr = sample.camera.camera_intr
        self.save_orig_image(depth_im_table, camera_pose, stable_pose, shifted_camera_intr)

        for cnt, aligned_grasp in enumerate(aligned_grasps):
            collision_free = self.is_grasp_collision_free(aligned_grasp, collision_checker)
            # Project grasp coordinates in image
            grasp_2d = aligned_grasp.project_camera(self.T_obj_camera, shifted_camera_intr)

            depth_im_no_rot, binary_im_no_rot = self.prepare_images(depth_im_table,
                                                                    binary_im,
                                                                    grasp_2d)

            T_grasp_camera = aligned_grasp.gripper_pose(self.gripper).inverse() * self.T_obj_camera.inverse()

            if VISUALISE_3D:
                vis.figure()
                T_obj_world = vis.mesh_stable_pose(self.obj.mesh.trimesh,
                                                   stable_pose.T_obj_world, style='surface', dim=0.5, plot_table=False)
                T_camera_world = T_obj_world * self.T_obj_camera.inverse()
                vis.gripper(self.gripper, aligned_grasp, T_obj_world, color=(0.3, 0.3, 0.3),
                            T_camera_world=T_camera_world)
                vis.show()

            hand_pose = self.get_hand_pose(grasp_2d, T_grasp_camera)

            self.add_datapoint(depth_im_no_rot, binary_im_no_rot, hand_pose,
                               collision_free, aligned_grasp, grasp_metrics)
        self.cur_image_label += 1

    def save_orig_image(self, depth, camera_pose, stable_pose, camera_intr):
        depth_im = Image.fromarray(depth.data).crop((150, 150, 450, 450))
        depth_im.save(self.image_dir + 'depth_{:06d}.tiff'.format(self.cur_image_label))
        with open(self.image_dir + 'object_{:06d}.txt'.format(self.cur_image_label), 'w') as f:
            f.write(self.obj.key)
        self.T_obj_camera.save(self.image_dir + 'camera_tf_{:06d}.tf'.format(self.cur_image_label))
        np.savetxt(self.image_dir + 'camera_pose_{:06d}.txt'.format(self.cur_image_label), camera_pose)
        save_stp = StablePoseFile(self.image_dir + 'stable_pose_{:06d}.stp'.format(self.cur_image_label))
        save_stp.write([stable_pose])
        camera_intr.save(self.image_dir + 'camera_intr_{:06d}.intr'.format(self.cur_image_label))

    def render_data(self):
        logging.basicConfig(level=logging.WARNING)
        for dataset in self.datasets:
            logging.info('Generating data for dataset %s' % dataset.name)
            object_keys = dataset.object_keys

            for obj_key in object_keys:
                self.tensor_datapoint['obj_labels'] = self.cur_obj_label
                if self.cur_obj_label % 10 == 0:
                    logging.info("Object number: %d" % self.cur_obj_label)
                self.obj = dataset[obj_key]

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
                    before = time.time()
                    samples = self.render_images(scene_objs,
                                                 stable_pose,
                                                 self.config['images_per_stable_pose'],
                                                 camera_config=self._camera_configs())
                    # print("Rendering took {:05f} seconds".format(time.time()-before))
                    # times = []
                    for sample in samples:
                        # before = time.time()
                        self.save_samples(sample, grasps, T_obj_stp, collision_checker, grasp_metrics, stable_pose)
                        # times.append(time.time() - before)
                    # print("Saving one sample took avg {:05f} seconds".format(sum(times)/len(times)))
                    self.cur_pose_label += 1
                    gc.collect()
                    # next stable pose
                self.cur_obj_label += 1
                # next object
        # Save dataset
        self.tensor_dataset.flush()


if __name__ == "__main__":

    Generator = GenerateBalancedObliqueDataset(DATASET_DIR)
    Generator.render_data()
