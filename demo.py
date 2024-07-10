import os

import torch as th
import open3d as o3d

from main import BaseTrainer
from octmae.utils.misc import fetch_data, unnormalize_pts
from octmae.utils.config import parse_config
from octmae.nets.utils import get_xyz_from_octree


def main():
    config = parse_config()
    config.update_octree = True

    print('Loading a model...')
    model = BaseTrainer.load_from_checkpoint(config.checkpoint, config=config, strict=False)
    model.cuda()
    model.eval()

    print('Fetching data...')
    batch = fetch_data(config.img_path, config.depth_path, config.mask_path, config.camera_info_path, config, 1.0)
    grid_res = 1 << config.min_lod
    with th.no_grad():
        print('Running inference...')
        output = model.model(batch)
        octrees_out = output['octrees_out']
        z_min = batch[-2][0]
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
        pcd_sdf_vis.normals = o3d.utility.Vector3dVector(normals)
        pcd_sdf_vis.colors = o3d.utility.Vector3dVector(((normals + 1) / 2))
        o3d.io.write_point_cloud(os.path.join('demo', 'output.pts'), pcd_sdf_vis)
        print('saved!')

if __name__ == '__main__':
    main()
