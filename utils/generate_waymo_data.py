import os
import glob
import argparse
import tqdm
import numpy as np
import pickle
import logging
import multiprocessing
import tensorflow as tf
import concurrent.futures as futures
from functools import partial
from pathlib import Path

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2


WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
WAYMO_GROUND = 0.34


def parse_range_image_flow_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    
    for laser in frame.lasers:
        
        if len(laser.ri_return1.range_image_flow_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
            
        if len(laser.ri_return2.range_image_flow_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
            
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud_flow(frame, range_images, range_images_flow, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """    
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []
    points_flow = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))   # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )   # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_flow = range_images_flow[c.name][ri_index]
        
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_flow_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]

        flow_x = range_image_flow_tensor[..., 0]
        flow_y = range_image_flow_tensor[..., 1]
        flow_z = range_image_flow_tensor[..., 2]

        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))

        points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, tf.compat.v1.where(range_image_mask)), axis=1)
        points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, tf.compat.v1.where(range_image_mask)), axis=1)
        points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, tf.compat.v1.where(range_image_mask)), axis=1)

        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

        points_flow.append(tf.concat([points_flow_x_tensor, points_flow_y_tensor, points_flow_z_tensor], axis=-1).numpy())

    return points, points_flow, cp_points, points_NLZ, points_intensity, points_elongation


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def gen_flow_func(args):
    seq_list = []
    for seq_path in glob.glob(os.path.join(args.root_data_dir, 'raw_data', '*.tfrecord')):
        seq_list.append(seq_path)
    
    for sequence_file in seq_list:
        sequence_name_k = sequence_file.split('/')[-1][:-9]
        seq_path_k = os.path.join(args.root_data_dir, 'processed', sequence_name_k)
        savez_path_k = os.path.join(args.root_data_dir, 'scene_flow', sequence_name_k)
        os.makedirs(savez_path_k, exist_ok=True)
        
        num_skipped_infos = 0
        if not os.path.exists(seq_path_k):
            num_skipped_infos += 1
            print('not exist: ', seq_path_k)
            print('*********************************\n')
            continue
        
        # ANCHOR: get scene flow from tfrecord files
        flow_data_list = []
        dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
        for cnt, data in enumerate(dataset):
            if cnt < 0:
                continue
            
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            range_images, camera_projections, _, range_image_top_pose = \
                frame_utils.parse_range_image_and_camera_projection(frame)

            range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
            _, flows, _, _, _, _ = convert_range_image_to_point_cloud_flow(frame, range_images, range_images_flow, camera_projections, range_image_top_pose)

            flows_all = np.concatenate(flows, axis=0)   # scene flow
            flow_data_list.append(flows_all)
        
        # ANCHOR: generate per-pair point cloud and flow
        npy_file_list = glob.glob(os.path.join(seq_path_k, '*.npy'))   # xxx.npy
        info_file = glob.glob(os.path.join(seq_path_k, '*.pkl'))[0]   # xxx.pkl
        with open(info_file, 'rb') as f:
            labels = pickle.load(f)

        npy_file_list.sort(key=lambda x: int(x.split('/')[-1][:-4]))

        for idx in range(len(npy_file_list) - 1):
            info_1 = labels[idx]   # ['point_cloud', 'frame_id', 'image', 'pose', 'annos', 'num_points_of_each_lidar']
            name_from_info1 = '%04d' % info_1['point_cloud']['sample_idx']
            pose_1 = info_1['pose']  # 4x4
            info_2 = labels[idx + 1]
            name_from_info2 = '%04d' % info_2['point_cloud']['sample_idx']
            pose_2 = info_2['pose']  # 4x4

            pc1_path = npy_file_list[idx]
            pc2_path = npy_file_list[idx + 1]
            pc1_name = pc1_path.split('/')[-1][:-4]
            pc2_name = pc2_path.split('/')[-1][:-4]
            
            if name_from_info1 != pc1_name:
                print('error!')
                print('info1: ', name_from_info1, 'pc1: ', pc1_name)
                exit(1)
            if name_from_info2 != pc2_name:
                print('error!')
                print('info2: ', name_from_info2, 'pc2: ', pc2_name)
                exit(1)

            pc1_features = np.load(pc1_path)   # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
            pc1_data, _, _, flag1 = pc1_features[:, 0:3], pc1_features[:, 3], pc1_features[:, 4], pc1_features[:, 5]
            pc1_data = pc1_data[flag1 == -1]
            
            pc2_features = np.load(pc2_path)   # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
            pc2_data, _, _, flag2 = pc2_features[:, 0:3], pc2_features[:, 3], pc2_features[:, 4], pc2_features[:, 5]
            pc2_data = pc2_data[flag2 == -1]
        
            flow_data = flow_data_list[idx+1]
            flow_data = flow_data[flag2 == -1]
            
            # NOTE: convert flow data to be consistent with our implementation
            sf = flow_data * 0.1
            translation_1 = pose_1[0:3, 3]
            rotation_1 = pose_1[0:3, 0:3]
            translation_2 = pose_2[0:3, 3]
            rotation_2 = pose_2[0:3, 0:3]
            rotation_inv_2 = np.linalg.inv(rotation_2)
            sf = pc2_data - ((pc2_data - sf) @ rotation_inv_2 + translation_2 - translation_1) @ rotation_1
            
            # NOTE: need to remove ground first
            non_ground_pc1 = pc2_data[..., 2] > WAYMO_GROUND
            non_ground_pc2 = pc1_data[..., 2] > WAYMO_GROUND
            pc1 = pc2_data[non_ground_pc1]
            pc2 = pc1_data[non_ground_pc2]
            flow = -sf[non_ground_pc1]
            
            savez_path = os.path.join(savez_path_k, pc1_name + '_' + pc2_name + '.npz')
            print(savez_path)
            # NOTE: the flow label is in reverse order
            np.savez_compressed(savez_path, pc1=pc1, pc2=pc2, flow=flow)
            

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def generate_labels(frame):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, _, _, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = Path(os.path.join(save_path, sequence_name))
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        print('Skip sequence since it has been processed before: %s' % pkl_file)
        return sequence_infos

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame)
            info['annos'] = annotations

        num_points_of_each_lidar = save_lidar_points(frame, cur_save_dir / ('%04d.npy' % cnt))
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos


def save_lidar_points(frame, cur_save_path):
    range_images, camera_projections, seg_labels, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, _, points_in_NLZ_flag, points_intensity, points_elongation = \
        convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    
    return num_points_of_each_lidar


def create_waymo_infos(args, logger=None):
    # ANCHOR: set dataset split
    split_dir = os.path.join(args.root_data_dir, 'ImageSets', args.data_split + '.txt')
    sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
    infos = []
    waymo_infos = []

    num_skipped_infos = 0
    for k in range(len(sample_sequence_list)):
        sequence_name = os.path.splitext(sample_sequence_list[k])[0]
        info_path = Path(os.path.join(args.data_save_path, sequence_name, '%s.pkl' % sequence_name))
        
        if '_with_camera_labels' not in str(info_path) and not info_path.exists():
            info_path = Path(str(info_path)[:-9] + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(info_path) and not info_path.exists():
            info_path = Path(str(info_path).replace('_with_camera_labels', ''))
            
        if not info_path.exists():
            num_skipped_infos += 1
            continue
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_infos.extend(infos)

    infos.extend(waymo_infos[:])
    logger.info('Total skipped info %s' % num_skipped_infos)
    logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))
    
    # sampled_waymo_infos = []
    # for k in range(0, len(infos), 5):
    #     sampled_waymo_infos.append(infos[k])
    # infos = sampled_waymo_infos
    # logger.info('Total sampled samples for Waymo dataset: %d' % len(infos))
        
    # ANCHOR: get dataset infos
    print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
            % (1, len(sample_sequence_list)))

    process_single_sequence_fn = partial(process_single_sequence, save_path=args.data_save_path, sampled_interval=1, has_label=True)
    sample_sequence_file_list = []
    for seq_file in sample_sequence_list:
        seq_file_dir = Path(os.path.join(args.root_data_dir, 'raw_data', seq_file))
        print(seq_file_dir)
        if '_with_camera_labels' not in str(seq_file_dir) and not seq_file_dir.exists():
            sample_sequence_file_list.append(Path(str(seq_file_dir)[:-9] + '_with_camera_labels.tfrecord'))
        if '_with_camera_labels' in str(seq_file_dir) and not seq_file_dir.exists():
            sample_sequence_file_list.append(Path(str(seq_file_dir).replace('_with_camera_labels', '')))
        else:
            sample_sequence_file_list.append(seq_file_dir)
            
    with futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
        sequence_infos = list(tqdm.tqdm(executor.map(process_single_sequence_fn, sample_sequence_file_list), total=len(sample_sequence_file_list)))
    all_sequences_infos = [item for infos in sequence_infos for item in infos]
    
    return all_sequences_infos


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('[%(asctime)s  %(filename)s %(lineno)d '
                                  '%(levelname)5s]  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_split', type=str, default='val', choices=['train', 'val', 'val_test'], help='specific data split.')
    parser.add_argument('--root_data_dir', type=str, default='/dataset/test_waymo', help='path to the waymo dataset under specific split.')
    args = parser.parse_args()
    
    logger = create_logger()
    
    args.data_save_path = os.path.join(args.root_data_dir, 'processed')
    os.makedirs(args.data_save_path, exist_ok=True)

    waymo_infos = create_waymo_infos(args, logger=logger)
    
    filename = os.path.join(args.data_save_path, 'waymo_infos_%s.pkl' % args.data_split)
    with open(filename, 'wb') as f:
        pickle.dump(waymo_infos, f)
    
    # NOTE: generate waymo open validation dataset
    gen_flow_func(args)
