import io
import random
import collections
import json

import cv2
import numpy as np
import torch as th
from PIL import Image
import imageio.v3 as iio
from torchvision import transforms
from ocnn.octree import Points


def rle_to_binary_mask(rle, bbox_visib=None):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')
    
    start = 0

    if bbox_visib is not None and len(counts) % 2 == 0 and bbox_visib[0] == 0 and bbox_visib[1] == 0:
        counts.insert(0, 0)

    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2
    
    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def get_camera_rays(K, img_size, inv_scale=1):
    u, v = np.meshgrid(np.arange(0, img_size[1], inv_scale), np.arange(0, img_size[0], inv_scale))
    
    # Convert to homogeneous coordinates
    u = u.reshape(-1)
    v = v.reshape(-1)
    ones = np.ones(u.shape[0])
    uv1 = np.stack((u, v, ones), axis=-1)  # shape (H*W, 3)
    
    K_inv = np.linalg.inv(K)
    pts = np.dot(uv1, K_inv.T)  # shape (H*W, 3)
    pts = pts.reshape((img_size[0] // inv_scale, img_size[1] // inv_scale, 3))

    return pts


def batch2cuda(batch):
    new_batch = []
    for elem in batch:
        if th.is_tensor(elem):
            new_batch.append(elem.cuda())
        elif isinstance(elem, collections.abc.Sequence):
            new_seq = []
            for s in elem:
                if th.is_tensor(s) or isinstance(s, Points):
                    new_seq.append(s.cuda())
                else:
                    new_seq.append(s)
            new_batch.append(new_seq)
        else:
            new_batch.append(elem)
    return new_batch


def decode_depth(key, data):
    if not key.endswith('depth.png'):
        return None
    return np.asarray(iio.imread(io.BytesIO(data)), dtype=np.float32)


def normalize_pts(pts, z_min, grid_size, grid_res):
    pts[:, :2] = pts[:, :2] / (grid_res * grid_size // 2)    # (-1, 1)
    pts[:, 2] = (((pts[:, 2] - z_min) / (grid_res * grid_size)) - 0.5) * 2.0    # (-1, 1)
    return pts


def unnormalize_pts(pts, z_min, grid_size, grid_res):
    pts[:, :2] = (grid_res * grid_size // 2) * pts[:, :2]
    pts[:, 2] = (0.5 * pts[:, 2] + 0.5) * grid_res * grid_size + z_min
    return pts


def project_to_image_plane(pts_3d, K, img_size):
    pts_2d_homogeneous = pts_3d @ K.T  # shape (N, 3)
    pts_2d = pts_2d_homogeneous[:, :2] / (pts_2d_homogeneous[:, 2:3] + 1e-4)
    pts_2d[:, 0] /= img_size[1]
    pts_2d[:, 1] /= img_size[0]
    pts_2d = (pts_2d - 0.5) * 2.0 # (-1, 1)
    return pts_2d


def make_sample_wrapper(config, is_eval=False,
                        K=[[572.41136339, 0., 325.2611084], [0., 573.57043286, 242.04899588], [0., 0., 1.]]):
    img_size = (config.img_height, config.img_width) # should use a config
    resized_img_size = (480, 640)
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    K = np.asarray(K, dtype=np.float32) # should use a config
    camera_rays = get_camera_rays(K, img_size)

    transform = transforms.Compose([
        transforms.Resize(resized_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    def make_sample_surface(sample):
        rgb = transform(sample['rgb.jpg'])
        depth = sample['camera.json']['depth_scale'] * sample['depth.png']
        if not is_eval:
            depth = depth * (1 + 0.005 * np.random.normal(loc=0.0, scale=1.0, size=depth.shape)) # add noise
        depth = depth.astype(np.float32)
        obj_pose = sample['gt.json']
        obj_info = sample['gt_info.json']
        mask_rle = sample['mask_visib.json']
        K = np.asarray(sample['camera.json']['cam_K']).astype(np.float32).reshape(3, 3)
        spc = th.from_numpy(sample['spc.npz']['spc'].astype(np.float32))
        obj_ids = th.from_numpy(sample['spc.npz']['obj_ids'].astype(np.int32))
        _, obj_ids = th.unique(obj_ids, sorted=True, return_inverse=True)

        mask = np.zeros(img_size, dtype=bool)
        for ind, (oi, op) in enumerate(zip(obj_info, obj_pose)):
            # print('selected_id', selected_id, op['obj_id'])
            if oi['visib_fract'] < 0.2:    # get rid of heavily occluded objects
                continue
            # print('selected!', ind)
            imask_rle = mask_rle[str(ind)]
            imask = rle_to_binary_mask(imask_rle, oi['bbox_visib'])
            mask = np.logical_or(mask, imask)

        if not is_eval:
            kernel_size = random.choice([1, 3, 5])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask.astype(np.float32), kernel, iterations=1) > 0.5
        mask = np.logical_and(mask, depth > 10.0)
        # camera_rays = get_camera_rays(K, img_size)
        full_pts_3d = th.from_numpy((camera_rays * depth[:, :, None])[mask > 0.0].astype(np.float32)).reshape(-1, 3)

        z_min = (th.min(full_pts_3d[:, 2]) // grid_size) * grid_size - 5 * grid_size

        vdb_grid_size = 1.25
        offset = vdb_grid_size * 0.5 # this offset is to fix the misalignment in VDB

        spc_mask = th.logical_and(spc[:, 3] < vdb_grid_size, spc[:, 3] > -vdb_grid_size)
        spc = spc[spc_mask]
        obj_ids = obj_ids[spc_mask]

        pts_3d_in = Points(normalize_pts(full_pts_3d, z_min, grid_size, grid_res))
        pts_3d_gt = Points(points=normalize_pts(spc[:, :3] + offset, z_min, grid_size, grid_res), normals=spc[:, 4:7], features=spc[:, 3:4], labels=obj_ids)
        pts_3d_in.clip()
        pts_3d_gt.clip()

        if pts_3d_in.points.shape[0] < 100 or pts_3d_gt.points.shape[0] < 1000:
            frame_idx = sample['__key__']
            raise Exception('This item does not have enough points', frame_idx)

        return (rgb, mask, depth, pts_3d_in, pts_3d_gt, K, z_min, sample['__key__'])
    
    return make_sample_surface

def fetch_data(rgb_path, depth_path, mask_path, camera_path, config, depth_scale=1.0, device='cuda'):
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    img_size = (config.img_height, config.img_width) # should use a config
    resized_img_size = (480, 640)
    transform = transforms.Compose([
        transforms.Resize(resized_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb = transform(Image.open(rgb_path))
    depth = np.asarray(iio.imread(depth_path), dtype=np.float32)
    depth = depth_scale * depth
    mask = np.asarray(Image.open(mask_path))
    with open(camera_path, 'r') as f:
        K = np.array(json.load(f)['cam_K']).astype(np.float32).reshape(3, 3)
    camera_rays = get_camera_rays(K, img_size)
    full_pts_3d = th.from_numpy((camera_rays * depth[:, :, None])[mask > 0.0].astype(np.float32)).reshape(-1, 3)
    z_min = (th.min(full_pts_3d[:, 2]) // grid_size) * grid_size - 5 * grid_size
    pts_3d_in = Points(normalize_pts(full_pts_3d, z_min, grid_size, grid_res))
    pts_3d_in.clip()

    # Transfer data to a GPU
    rgb = rgb.to(device)
    mask = th.from_numpy(mask).to(device)
    depth = th.from_numpy(depth).to(device)
    pts_3d_in = pts_3d_in.to(device)
    K = th.from_numpy(K).to(device)
    z_min = z_min.to(device)

    return (rgb[None], mask[None], depth[None], [pts_3d_in], [None], K[None], z_min[None], [rgb_path])
