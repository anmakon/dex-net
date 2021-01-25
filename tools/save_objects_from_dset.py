import argparse
import cPickle as pkl
import json
import csv
import os
import numpy as np
import logging

from autolab_core import YamlConfig
from meshpy import ObjFile, StablePoseFile

from dexnet.visualization import DexNetVisualizer3D as vis
from dexnet.constants import READ_ONLY_ACCESS
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.database import Hdf5Database

DATA_DIR = '/data'
"""
Script to save dexnet objects for dexnet database to file.
Object, pose and grasp can be specified via the script 
dexnet/tools/create_validation_pointers.py.
Data files are being saved to dexnet/data/meshes/dexnet.
With --num, the maximum amount of grasps in one subset (KIT and 3DNet)
can be specified.
"""


class GraspInfo(object):
    def __init__(self, grasp, collision_free, contact_points, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.contact_point1 = contact_points[0]
        self.contact_point2 = contact_points[1]
        self.phi = phi


def save_dexnet_objects(output_path, database, target_object_keys, config, pointers, num):
    slice_dataset = False
    files = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for each_file in os.listdir(output_path):
        os.remove(output_path + '/' + each_file)
    gripper = RobotGripper.load(config['gripper'])

    # Setup grasp params:
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
    #     #table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    # table_mesh = ObjFile(table_mesh_filename).read()

    dataset_names = target_object_keys.keys()
    datasets = [database.dataset(dn) for dn in dataset_names]

    if slice_dataset:
        datasets = [dataset.subset(0, 100) for dataset in datasets]

    start = 0
    for dataset in datasets:
        target_object_keys[dataset.name] = []
        end = start + len(dataset.object_keys)
        for cnt, _id in enumerate(pointers.obj_ids):
            if _id >= end or _id < start:
                continue
            target_object_keys[dataset.name].append(dataset.object_keys[_id - start])
            files.append(tuple([dataset.object_keys[_id - start], pointers.tensor[cnt],
                                   pointers.array[cnt], pointers.depths[cnt]]))
        start += end
    print(target_object_keys)
    print("target object keys:", len(target_object_keys['3dnet']), len(target_object_keys['kit']))
    files = np.array(files, dtype=[('Object_id', (np.str_, 40)), ('Tensor', int), ('Array', int), ('Depth', float)])

    # Precompute set of valid grasps
    candidate_grasps_dict = {}

    counter = 0
    for dataset in datasets:
        for obj in dataset:
            if obj.key not in target_object_keys[dataset.name]:
                continue
            print("Object in subset")
            # Initiate candidate grasp storage
            candidate_grasps_dict[obj.key] = {}

            # Setup collision checker
            collision_checker = GraspCollisionChecker(gripper)
            collision_checker.set_graspable_object(obj)

            # Read in the stable poses of the mesh
            stable_poses = dataset.stable_poses(obj.key)
            try:
                stable_pose = stable_poses[pointers.pose_num[counter]]
            except IndexError:
                print("Problems with reading pose. Tensor %d, Array %d, Pose %d" %
                      (pointers.tensor[counter], pointers.array[counter], pointers.pose_num[counter]))
                print("Stable poses:", stable_poses)
                print("Pointers pose:", pointers.pose_num[counter])
                counter += 1
                print("Continue.")
                continue
            print("Read in stable pose")
            candidate_grasps_dict[obj.key][stable_pose.id] = []

            # Setup table in collision checker
            T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
            T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
            T_table_obj = T_obj_table.inverse()

            collision_checker.set_table(table_mesh_filename, T_table_obj)

            # read grasp and metrics
            grasps = dataset.grasps(obj.key, gripper=gripper.name)

            # align grasps with the table
            aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]
            i = 0
            found = False
            if len(aligned_grasps) < pointers.grasp_num[counter]:
                raise IndexError
            print("pointers grasp num", pointers.grasp_num[counter])
            print("tensor", pointers.tensor[counter])
            print("array", pointers.array[counter])
            # Check grasp validity
            for aligned_grasp in aligned_grasps:
                # Check angle with table plane and skip unaligned grasps
                _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                if not perpendicular_table:
                    continue

                # Check whether any valid approach directions are collision free
                collision_free = False
                for phi_offset in phi_offsets:
                    rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                    collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                    if not collides:
                        collision_free = True
                        break

                # Store if aligned to table
                if i == pointers.grasp_num[counter]:
                    found, contact_points = aligned_grasp.close_fingers(obj)
                    print("Contact points", contact_points)
                    if not found:
                        print("Could not find contact points. continue.")
                        break
                    else:
                        print("Original metric: ", pointers.metrics[counter])
                        print("Metrics mapped point: ", dataset.grasp_metrics(obj.key, [aligned_grasp], gripper=gripper.name))
                        candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp,
                                                                                        collision_free,
                                                                                        [contact_points[0].point,
                                                                                         contact_points[1].point]))
                        # logging.info('Grasp %d' % (aligned_grasp.id))
                        # vis.figure()
                        # vis.gripper_on_object(gripper, aligned_grasp, obj, stable_pose.T_obj_world, plot_table=False)
                        # vis.show()
                        break

                i += 1
            counter += 1
            if found:
                # Save files
                print("Saving file: ", obj.key)
                savefile = ObjFile(DATA_DIR + "/meshes/dexnet/" + obj.key + ".obj")
                savefile.write(obj.mesh)
                # Save stable poses
                save_stp = StablePoseFile(DATA_DIR + "/meshes/dexnet/" + obj.key + ".stp")
                save_stp.write(stable_poses)
                # Save candidate grasp info
                pkl.dump(candidate_grasps_dict[obj.key], open(DATA_DIR + "/meshes/dexnet/" + obj.key + ".pkl", 'wb'))
                # Save grasp metrics
                candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                candidate_grasps = [g.grasp for g in candidate_grasp_info]
                grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name)
                write_metrics = json.dumps(grasp_metrics)
                f = open(DATA_DIR + "/meshes/dexnet/" + obj.key + ".json", "w")
                f.write(write_metrics)
                f.close()

            if num is not None and counter >= num:
                break
    with open(DATA_DIR + '/meshes/dexnet/files.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for point in files:
            csv_writer.writerow(point)


class ValidationPointers(object):
    def __init__(self, filename=DATA_DIR + '/generated_val_indices.txt'):
        f = open(filename, 'rb')
        dtype = [('Tensor', int), ('Array', int), ('Obj_id', int),
                 ('Pose_num', int), ('Grasp_num', int), ('Metric', float), ('Depth', float)]
        _data = []
        for line in f:
            if not 'label' in line:
                val = line.split(',')
                _data.append(tuple([int(val[0]), int(val[1]), int(val[2]),
                                    int(val[3]), int(val[4]), float(val[5]), float(val[6])]))
        data = np.array(_data, dtype=dtype)
        # data = np.array([tuple(map(int, line.split(',')[:-1])) for line in f if not 'label' in line], dtype=dtype)
        data = np.sort(data, order='Obj_id')
        self.tensor = data['Tensor']
        self.array = data['Array']
        self.obj_ids = data['Obj_id']
        self.pose_num = data['Pose_num']
        self.grasp_num = data['Grasp_num']
        self.metrics = data['Metric']
        self.depths = data['Depth']
        print("Amount of grasps:", len(self.tensor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num",
                        type=int,
                        default=None)

    args = parser.parse_args()
    num = args.num
    logging.basicConfig(level=logging.WARNING)

    config_filename = "./cfg/tools/generate_gqcnn_dataset.yaml"
    output_path = DATA_DIR + "/meshes/dexnet"

    config = YamlConfig(config_filename)
    database = Hdf5Database(config['database_name'],
                            access_level=READ_ONLY_ACCESS)

    pointers = ValidationPointers()
    target_object_keys = config['target_objects']

    save_dexnet_objects(output_path, database, target_object_keys, config, pointers, num)
