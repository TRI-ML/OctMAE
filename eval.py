import os
import json

from scipy.spatial import KDTree
import numpy as np
import open3d as o3d
import torch as th
import webdataset as wds

from main import BaseTrainer
from octmae.utils.config import parse_config
from octmae.utils.misc import (
    make_sample_wrapper,
    decode_depth,
    unnormalize_pts,
    project_to_image_plane,
    batch2cuda
)
from octmae.nets.utils import get_xyz_from_octree


EVAL_DATA_SET_NAMES = ['synth_eval', 'ycb_video', 'hope', 'hb']
IMG_SIZES = {
    'synth_eval': (480, 640),
    'ycb_video': (480, 640),
    'hope': (1080, 1920),
    'hb': (480, 640),
}
INTRINSICS_K = {
    'synth_eval': [[572.41136339, 0., 325.2611084], [0., 573.57043286, 242.04899588], [0., 0., 1.]],
    'ycb_video': [[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]],
    'hope': [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    'hb': [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
}

S3_URL = 's3://tri-ml-datasets/mirage_datasets_2/eval_datasets'
WDS_URL = {
    'synth_eval': os.path.join(S3_URL, 'synth_eval', 'shard-{000000..000009}.tar'),
    'ycb_video': os.path.join(S3_URL, 'ycb_video', 'shard-{000000..000009}.tar'),
    'hope': os.path.join(S3_URL, 'hope', 'shard-000000.tar'),
    'hb': os.path.join(S3_URL, 'hb', 'shard-{000000..000009}.tar'),
}

def evaluate():
    config = parse_config()
    config_name = os.path.basename(config.config).replace('.yaml', '')
    dataset_name = config.eval_dataset_name
    if dataset_name not in EVAL_DATA_SET_NAMES:
        raise Exception('this dataset is not supported')

    # update config
    img_size = IMG_SIZES[dataset_name]
    config.img_height = img_size[0]
    config.img_width = img_size[1]
    config.update_octree = True

    result_dir = os.path.join('eval_results', config.model_name, config_name, dataset_name)
    mesh_dir = os.path.join(result_dir, 'meshes')
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir, exist_ok=True)

    print('Loading a model...')
    model = BaseTrainer.load_from_checkpoint(config.checkpoint, config=config)
    model.cuda()
    model.eval()

    print('Fetching data...')
    url = f'pipe:s5cmd cat {WDS_URL[dataset_name]}'
    dataset = (
        wds.WebDataset(url, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue, shardshuffle=False)
        .decode(decode_depth, 'pil')
        .map(make_sample_wrapper(config, K=INTRINSICS_K[dataset_name], is_eval=True), handler=wds.warn_and_continue)
        .batched(1)
    )
    dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True
    )

    results = predict_surface(model, dataloader, mesh_dir, config)

    print(results)

    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f)


def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.array([5.0, 10.0, 20.0, 30.0])):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score-5': F[0], # threshold = 5mm
            'f-score-10': F[1], # threshold = 10mm
            'f-score-20': F[2], # threshold = 20mm
            'f-score-30': F[3], # threshold = 30mm
        }

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def predict_surface(model, dataloader, mesh_dir, config):
    grid_res = 1 << config.min_lod

    out_dict = {
        'frame_idxs': [],
        'runtime_reg': [],
        'completeness': [],
        'accuracy': [],
        'normals completeness': [],
        'normals accuracy': [],
        'normals': [],
        'completeness2': [],
        'accuracy2': [],
        'chamfer-L2': [],
        'chamfer-L1': [],
        'f-score-5':[], # threshold = 5mm
        'f-score-10': [], # threshold = 10mm
        'f-score-20': [], # threshold = 20mm
        'f-score-30': [], # threshold = 30mm
    }

    for batch in dataloader:
        metrics = {}
        batch = batch2cuda(batch)

        frame_idx = batch[-1][0]
        K = batch[-3][0]
        z_min = batch[-2][0]
        pts_3d_gt = batch[4][0]
        pcd_path = os.path.join(mesh_dir, f'{frame_idx}.pts')

        # if frame_idx != '000000_000864':
        #     continue

        if os.path.exists(pcd_path):
            pcd_ = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd_.points)
            normals = np.asarray(pcd_.colors) * 2 - 1.0
        else:
            print(f'Running inference for {frame_idx}')
            with th.no_grad():
                output = model.model(batch)
                # metrics['runtime_reg'] = output['runtime_reg']

                octrees_out = output['octrees_out']
                pcd = get_xyz_from_octree(octrees_out, config.max_lod, True)
                pcd = unnormalize_pts(pcd, z_min, config.grid_size, grid_res)
                normals = octrees_out.normals[config.max_lod]
                sdf = octrees_out.features[config.max_lod][:, :1]

                pcd = pcd.cpu().numpy()
                normals = normals.cpu().numpy()
                sdf = sdf.cpu().numpy()
                pcd = pcd - normals * sdf

            pcd_sdf_vis = o3d.geometry.PointCloud()
            pcd_sdf_vis.points = o3d.utility.Vector3dVector(pcd)
            if normals is not None:
                pcd_sdf_vis.normals = o3d.utility.Vector3dVector(normals)
                pcd_sdf_vis.colors = o3d.utility.Vector3dVector(((normals + 1) / 2))
            else:
                pcd_sdf_vis.estimate_normals()
                normals = np.asarray(pcd_sdf_vis.normals)
            o3d.io.write_point_cloud(pcd_path, pcd_sdf_vis)
            print('pcd is exported to', pcd_path)

        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        unnorm_pts_3d_gt = unnormalize_pts(pts_3d_gt.points.clone(), z_min, config.grid_size, grid_res)
        pts_2d_gt = project_to_image_plane(unnorm_pts_3d_gt.clone(), K, (config.img_height, config.img_width))
        pts_mask_min = th.all(pts_2d_gt > -1.0, dim=1)
        pts_mask_max = th.all(pts_2d_gt < 1.0, dim=1)
        pts_mask = th.logical_and(pts_mask_min, pts_mask_max)
        gt_pcd = unnorm_pts_3d_gt[pts_mask].cpu().numpy()
        gt_normals = pts_3d_gt.normals[pts_mask].cpu().numpy()

        metric = eval_pointcloud(pcd, gt_pcd, normals, gt_normals)
        print(metric)
        metrics.update(metric)

        for k in metrics.keys():
            out_dict[k].append(metrics[k])
        out_dict['frame_idxs'].append(frame_idx)

    for k in out_dict.keys():
        if k == 'frame_idxs':
            continue
        out_dict[k] = np.mean(np.asarray(out_dict[k])).item()

    return out_dict


if __name__ == '__main__':
    evaluate()
