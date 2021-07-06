import numpy as np
import gc
import os
import logging
from scipy.stats import poisson

from meshpy import UniformPlanarWorksurfaceImageRandomVariable, ObjFile, RenderMode, SceneObject
from autolab_core import RigidTransform, YamlConfig
from dexnet.learning import TensorDataset
from PIL import Image
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

DATA_DIR = '/data'
DATASET_DIR = '/Balanced_Viewpoint_DexNet/'


class GenerateBalancedObliqueDataset:
    def __init__(self, output_dir):
        self.NUM_OBJECTS = None
        self.table_file = DATA_DIR + '/meshes/table.obj'
        self.data_dir = DATA_DIR + '/meshes/dexnet/'

        output_dir = DATA_DIR + output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

        self.config = YamlConfig('./cfg/tools/generate_oblique_gqcnn_dataset.yaml')

        self.phi_offsets = self._generate_phi_offsets()
        self.datasets, self.target_object_keys = self._load_datasets()
        self.tensor_dataset = TensorDataset(self.output_dir, self._set_tensor_config())
        self.tensor_datapoint = self.tensor_dataset.datapoint_template
        self.gripper = self._set_gripper()
        self._table_mesh_filename = self._set_table_mesh_filename()
        self.table_mesh = self._set_table_mesh()

        self.cur_pose_label = 0
        self.cur_obj_label = 0
        self.cur_image_label = 0

        self.grasp_upsample_distribuitions = self._get_upsample_distributions()
        self.camera_distributions = self._get_camera_distributions()

        self.obj = None
        self.T_obj_camera = None

    def _get_camera_distributions(self):
        # Load relative amount of grasps per elevation angle phi
        relative_elevation_angles = np.load(DATA_DIR + '/meshes/relative_elevation_angles.npy')
        relative_elevation_angles[relative_elevation_angles > 1] = 1.0
        # Load positivity rate per elevation angle phi
        positivity_rate = np.load(DATA_DIR + '/meshes/positivity_rate.npy')
        # Get additional sample ratio through resampling positive grasps per elevation angle phi
        additional_sample_ratio = 1 + self.config['gt_positive_ratio'] - positivity_rate / 100
        # Add additonal sample ratio to relative amount of grasps per elevation angle phi
        camera_frequencies = 1 / (relative_elevation_angles * additional_sample_ratio)
        camera_frequencies = np.round(camera_frequencies / sum(camera_frequencies) * 10000).astype(int)
        # Get camera distribution according to goal sampling ratios
        camera_distribution = [0] * camera_frequencies[0] + [1] * camera_frequencies[1] + \
                              [2] * camera_frequencies[2] + [3] * camera_frequencies[3] + \
                              [4] * camera_frequencies[4] + [5] * camera_frequencies[5] + \
                              [6] * camera_frequencies[6] + [7] * camera_frequencies[7] + \
                              [8] * camera_frequencies[8] + [9] * camera_frequencies[9] + \
                              [10] * camera_frequencies[10] + [11] * camera_frequencies[11]
        return camera_distribution

    def _get_upsample_distributions(self):
        # Get ratio of ground truth positive grasps in unbalanced dataset
        positivity_rate = np.load(DATA_DIR + '/meshes/positivity_rate.npy')
        # Get desired ratio of ground truth positive grasps
        gt_positive_ratio = self.config['gt_positive_ratio'] * 100
        # Get upsample ratio for positive grasps
        upsample_ratio = (gt_positive_ratio - positivity_rate) / positivity_rate
        # Set upsample ratio of negative sampling ratios to zero
        upsample_ratio[upsample_ratio < 0] = 0
        X = [0, 1, 2, 3, 4, 5]
        sample_distributions = []
        # Fit poisson distribution to ratios
        for ratio in upsample_ratio:
            poisson_pd = np.round(poisson.pmf(X, ratio) * 1000).astype(int)
            # Generate vector with possible sample sizes to randomly choose from
            sample_distribution = [0] * poisson_pd[0] + [1] * poisson_pd[1] + [2] * poisson_pd[2] + \
                                  [3] * poisson_pd[3] + [4] * poisson_pd[4] + [5] * poisson_pd[5]
            sample_distributions.append(sample_distribution)
        return sample_distributions

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
        tensor_config['fields']['ferrari_canny'] = {}
        tensor_config['fields']['ferrari_canny']['dtype'] = 'float32'
        tensor_config['fields']['force_closure'] = {}
        tensor_config['fields']['force_closure']['dtype'] = 'float32'
        return tensor_config

    def get_positive_grasps(self, dataset, grasps):
        metrics = dataset.grasp_metrics(self.obj.key, grasps, gripper=self.gripper.name)
        positive_grasps = []
        for cnt in range(0, len(metrics)):
            if metrics[cnt]['robust_ferrari_canny'] >= 0.002:
                positive_grasps.append(grasps[cnt])
        return positive_grasps

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

    def prepare_images(self, depth_im_table, binary_im, grasp_2d, cx, cy):
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
        return depth_im_tf, binary_im_tf

    def add_datapoint(self, depth_im_tf, binary_im_tf, hand_pose, collision_free,
                      camera_pose, aligned_grasp, grasp_metrics):

        # Add data to tensor dataset
        self.tensor_datapoint['depth_ims_tf_table'] = depth_im_tf
        self.tensor_datapoint['obj_masks'] = binary_im_tf
        self.tensor_datapoint['hand_poses'] = hand_pose
        self.tensor_datapoint['obj_labels'] = self.cur_obj_label
        self.tensor_datapoint['collision_free'] = collision_free
        self.tensor_datapoint['pose_labels'] = self.cur_pose_label
        self.tensor_datapoint['image_labels'] = self.cur_image_label
        self.tensor_datapoint['camera_poses'] = camera_pose

        # Add metrics to tensor dataset
        for metric_name, metric_val in grasp_metrics[aligned_grasp.id].iteritems():
            coll_free_metric = (1 * collision_free) * metric_val
            self.tensor_datapoint[metric_name] = coll_free_metric
        self.tensor_dataset.add(self.tensor_datapoint)

    def save_samples(self, sample, grasps, T_obj_stp, collision_checker, grasp_metrics, only_positive=False):
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
            if only_positive and not collision_free:
                continue

            # Project grasp coordinates in image
            grasp_2d = aligned_grasp.project_camera(self.T_obj_camera, shifted_camera_intr)

            depth_im_tf, binary_im_tf = self.prepare_images(depth_im_table,
                                                            binary_im,
                                                            grasp_2d,
                                                            cx,
                                                            cy)

            hand_pose = self.get_hand_pose(grasp_2d, shifted_camera_intr)

            self.add_datapoint(depth_im_tf, binary_im_tf, hand_pose,
                               collision_free, camera_pose, aligned_grasp, grasp_metrics)
        self.cur_image_label += 1

    def render_data(self):
        logging.basicConfig(level=logging.WARNING)
        for dataset in self.datasets:
            logging.info('Generating data for dataset %s' % dataset.name)
            object_keys = dataset.object_keys

            for obj_key in object_keys:
                logging.info("Object number: %d" % self.cur_obj_label)
                self.obj = dataset[obj_key]

                grasps = dataset.grasps(self.obj.key, gripper=self.gripper.name)
                positive_grasps = self.get_positive_grasps(dataset, grasps)

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
                    for _ in range(self.config['images_per_stable_pose']):
                        elev_angle = np.random.choice(self.camera_distributions)
                        camera_config = self._camera_configs()
                        camera_config['min_elev'] = elev_angle*5.0
                        camera_config['max_elev'] = (elev_angle+1)*5.0
                        sample = self.render_images(scene_objs,
                                                    stable_pose,
                                                    1,
                                                    camera_config=camera_config)
                        self.save_samples(sample, grasps, T_obj_stp, collision_checker, grasp_metrics)
                        # Get camera elevation angle for current sample
                        elev_angle = sample.camera.elev * 180 / np.pi
                        elev_bar = int(elev_angle // 5)
                        # Sample number of positive images from distribution
                        number_positive_images = np.random.choice(self.grasp_upsample_distribuitions[elev_bar])
                        # Render only positive images
                        new_config = self._camera_configs()
                        new_config['min_elev'] = elev_angle
                        new_config['max_elev'] = elev_angle
                        positive_render_samples = self.render_images(scene_objs,
                                                                     stable_pose,
                                                                     number_positive_images,
                                                                     camera_config=new_config)
                        if type(positive_render_samples) == list:
                            for pos_sample in positive_render_samples:
                                self.save_samples(pos_sample, positive_grasps, T_obj_stp,
                                                  collision_checker, grasp_metrics, only_positive=True)
                        else:
                            self.save_samples(positive_render_samples, positive_grasps, T_obj_stp,
                                              collision_checker, grasp_metrics, only_positive=True)

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
