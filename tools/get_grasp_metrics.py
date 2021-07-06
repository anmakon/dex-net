import numpy as np
import os
import logging

from autolab_core import YamlConfig

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import RobotGripper
from dexnet.database import Hdf5Database

DATA_DIR = '/data'
DATASET_DIR = DATA_DIR + '/Test/'
CONFIG = './cfg/tools/generate_3DOF_gqcnn_dataset.yaml'
VISUALISE_3D = True


class CalculateGraspWidth:
    def __init__(self, output_dir):
        self.NUM_OBJECTS = None

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.config = YamlConfig(CONFIG)
        self.output_dir = output_dir

        self.datasets, self.target_object_keys = self._load_datasets()
        self.gripper = self._set_gripper()

        self.cur_obj_label = 0

    def _set_gripper(self):
        return RobotGripper.load(self.config['gripper'])

    def _load_datasets(self):
        database = Hdf5Database(self.config['database_name'], access_level=READ_ONLY_ACCESS)
        target_object_keys = self.config['target_objects']
        dataset_names = target_object_keys.keys()
        datasets = [database.dataset(dn) for dn in dataset_names]
        return datasets, target_object_keys

    def render_data(self):
        logging.basicConfig(level=logging.WARNING)
        for dataset in self.datasets:
            logging.info('Generating data for dataset %s' % dataset.name)
            object_keys = dataset.object_keys

            for obj_key in object_keys:
                if self.cur_obj_label % 10 == 0:
                    logging.info("Object number: %d" % self.cur_obj_label)
                obj = dataset[obj_key]

                grasps = dataset.grasps(obj.key, gripper=self.gripper.name)
                metrics = dataset.grasp_metrics(obj.key, grasps, gripper=self.gripper.name)
                grasp_data = []
                # Load grasp metrics
                for cnt, grasp in enumerate(grasps):
                    grasp_point = grasp.close_fingers(obj, vis=True)
                    if grasp_point[0]:
                        width = grasp.width_from_endpoints(grasp_point[1][0].point, grasp_point[1][1].point)
                    else:
                        width = None
                    fc = metrics[cnt]['robust_ferrari_canny']
                    grasp_data.append([width, fc])
                np.save(self.output_dir + '/' + obj_key + '.npy', np.array(grasp_data))
                self.cur_obj_label += 1


if __name__ == "__main__":

    Generator = CalculateGraspWidth(DATASET_DIR)
    Generator.render_data()
