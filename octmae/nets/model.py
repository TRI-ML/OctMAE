import torch as th
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import ocnn
from ocnn.octree import Points, Octree
from vot import VoxelOcclusionTester

from octmae.utils.misc import normalize_pts, unnormalize_pts, project_to_image_plane
from octmae.nets.utils import get_xyz_from_octree, octree_align, octree_search
from octmae.nets.mae_encoder import MAEEncoder
from octmae.nets.mae_decoder import MAEDecoder
from octmae.nets.ocnn_blocks import OctreeConvBnElu, OctreeDeconvBnElu, OctreeResBlocks, Conv1x1BnElu


class OctMAE(pl.LightningModule):

    def __init__(self, config) -> None:
        super(OctMAE, self).__init__()
        self.config = config

        self.dim_model = config.dim_model
        self.dim_mae = config.dim_mae
        self.max_lod = config.max_lod
        self.min_lod = config.min_lod
        self.grid_res = 1 << self.min_lod
        self.grid_size = config.grid_size
        self.update_octree = config.update_octree

        self.channel_in = self.dim_model
        self.channels = [512, 512, 256, 256, 256, self.dim_mae, 96, 96, 48, 48]
        self.channels_out = [2, 2, 2, 2, 2, 2, 2, 2, 2, 6]

        self.backbone = smp.Unet(
            encoder_name='resnext50_32x4d',
            encoder_weights='imagenet',
            in_channels=3,
            classes=config.dim_model,
        )
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.decoder.parameters():
            param.requires_grad = False

        self.mae_encoder = MAEEncoder(config)
        self.mae_decoder = MAEDecoder(config)

        # if config.model_name.endswith('siren'):
        self.pred_head = th.nn.ModuleList([self._make_predict_module(
            self.channels[d], self.channels_out[d]) for d in range(self.min_lod, self.max_lod + 1)])

        self.mask_token = nn.Embedding(1, self.dim_mae)

        self.conv1 = OctreeConvBnElu(
            self.channel_in, self.channels[self.max_lod], nempty=False)
        self.encoder_blks = th.nn.ModuleList([OctreeResBlocks(
            self.channels[d], self.channels[d], 2, nempty=False)
            for d in range(self.max_lod, self.min_lod-1, -1)])
        self.downsample = th.nn.ModuleList([OctreeConvBnElu(
            self.channels[d], self.channels[d-1], kernel_size=[2], stride=2,
            nempty=False) for d in range(self.max_lod, self.min_lod, -1)])
        self.upsample = th.nn.ModuleList([OctreeDeconvBnElu(
            self.channels[d-1], self.channels[d], kernel_size=[2], stride=2,
            nempty=False) for d in range(self.min_lod+1, self.max_lod+1)])
        self.decoder_blks = th.nn.ModuleList([OctreeResBlocks(
            self.channels[d], self.channels[d], 2, nempty=False)
            for d in range(self.min_lod+1, self.max_lod+1)])

        self.vot_scale_factor = config.vot_scale_factor
        self.vot_inv_scale_factor = 1.0 / self.vot_scale_factor
        self.vot = VoxelOcclusionTester(round(config.img_height * self.vot_inv_scale_factor), round(config.img_width * self.vot_inv_scale_factor), self.grid_size)

    def get_input_feature(self, octree, feature_type='F'):
        r''' Get the input feature from the input `octree`.
        '''
        octree_feature = ocnn.modules.InputFeature(feature_type, nempty=False)
        out = octree_feature(octree)
        return out

    def get_ground_truth_signal(self, octree):
        octree_feature = ocnn.modules.InputFeature('NF', nempty=True)
        data = octree_feature(octree)
        return data

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
        return th.nn.Sequential(
            Conv1x1BnElu(channel_in, num_hidden),
            ocnn.modules.Conv1x1(num_hidden, channel_out, use_bias=True))

    def encoder(self, octree, min_lod, max_lod):
        convs = dict()
        feat = self.get_input_feature(octree)
        convs[max_lod] = self.conv1(feat, octree, max_lod)
        for i, d in enumerate(range(max_lod, min_lod-1, -1)):
          convs[d] = self.encoder_blks[i](convs[d], octree, d)
          if d > min_lod:
            convs[d-1] = self.downsample[i](convs[d], octree, d)
        return convs

    def process_batch(self, batch, rgb_feat):
        _, mask, depth, pts_3d_in_list, pts_3d_gt_list, K, z_min, _ = batch

        octrees_in = []
        octrees_out = []
        octrees_mid = []

        B = mask.shape[0]
        device = mask.device
        
        min_lod = self.config.min_lod
        max_lod = self.config.max_lod
        img_height = self.config.img_height
        img_width = self.config.img_width

        with th.no_grad():
            xy_ = th.linspace(0., self.grid_size * (self.grid_res-1), self.grid_res, device=device) - (self.grid_size * (self.grid_res-1)) // 2
            z_ = xy_ + (self.grid_size * self.grid_res) // 2
            all_voxel_centers = th.stack(th.meshgrid(xy_, xy_, z_, indexing='ij'), dim=-1).reshape(-1, 3)
            all_voxel_centers = all_voxel_centers.unsqueeze(0).repeat(B, 1, 1)
            all_voxel_centers[:, :, 2] += z_min.unsqueeze(-1)
            K_norm = K.clone()
            K_norm[:, :2] = K_norm[:, :2] * self.vot_inv_scale_factor
            depth_norm = -F.max_pool2d(-depth.unsqueeze(1), self.vot_scale_factor).squeeze(1)
            mask_norm = F.interpolate(mask.float().unsqueeze(1), scale_factor=self.vot_inv_scale_factor, mode='bilinear', align_corners=True).bool().squeeze(1)
            occ_flag_map = self.vot(all_voxel_centers, mask_norm, depth_norm, K_norm)

        for i, (pts_3d_in, pts_3d_gt) in enumerate(zip(pts_3d_in_list, pts_3d_gt_list)):
            #  Reconstruct an input octree
            unnorm_pts_3d_in = unnormalize_pts(pts_3d_in.points.clone(), z_min[i], self.grid_size, self.grid_res)
            pts_2d_in = project_to_image_plane(unnorm_pts_3d_in, K[i], (img_height, img_width))
            pc_features = F.grid_sample(rgb_feat[i:i+1], pts_2d_in[None, None], align_corners=True)[0, :, 0]
            pts_3d_in.features = pc_features.transpose(0, 1)
            octree_in = Octree(max_lod, min_lod - 1, device=pts_3d_in.device)
            octree_in.build_octree(pts_3d_in)
            octrees_in.append(octree_in)

            #  Reconstruct an intermediate octree
            octree_mid = Octree(max_lod, min_lod - 1, device=pts_3d_in.device)
            pts_3d_mi = Points(normalize_pts(all_voxel_centers[i:i+1][occ_flag_map[i:i+1]], z_min[i], self.grid_size, self.grid_res))
            octree_mid.build_octree(pts_3d_mi)
            octrees_mid.append(octree_mid)

            if not self.update_octree:
                #  Reconstruct a ground-truth octree
                octree_out = Octree(max_lod, min_lod - 1, device=pts_3d_in.device)
                unnorm_pts_3d_gt = unnormalize_pts(pts_3d_gt.points.clone(), z_min[i], self.grid_size, self.grid_res)
                pts_2d_gt = project_to_image_plane(unnorm_pts_3d_gt, K[i], (img_height, img_width))
                pts_mask_min = th.all(pts_2d_gt > -1.0, dim=1)
                pts_mask_max = th.all(pts_2d_gt < 1.0, dim=1)
                pts_mask = th.logical_and(pts_mask_min, pts_mask_max)
                tmp = pts_3d_gt.__getitem__(pts_mask)
                pts_3d_gt.__dict__.update(tmp.__dict__)
                octree_out.build_octree(pts_3d_gt)
                octrees_out.append(octree_out)

        octrees_in = ocnn.octree.merge_octrees(octrees_in)
        octrees_mid = ocnn.octree.merge_octrees(octrees_mid)
        octrees_in.construct_all_neigh()

        if self.update_octree:
            octrees_out = octrees_mid
        else:
            octrees_out = ocnn.octree.merge_octrees(octrees_out)
        octrees_out.construct_all_neigh()

        batch = {
            'octrees_in': octrees_in,
            'octrees_mid': octrees_mid,
            'octrees_out': octrees_out
        }

        return batch


    def forward(self, batch):
        x = batch[0]
        x = self.backbone(x)

        batch = self.process_batch(batch, x)

        octrees_in = batch['octrees_in']
        octrees_mid = batch['octrees_mid']
        octrees_out = batch['octrees_out']

        # start = th.cuda.Event(enable_timing=True)
        # end = th.cuda.Event(enable_timing=True)
        # start.record()

        convs = self.encoder(octrees_in, self.min_lod, self.max_lod)

        kv_index = get_xyz_from_octree(octrees_in, self.min_lod)
        q_index = get_xyz_from_octree(octrees_mid, self.min_lod)

        grid_features = self.mae_encoder(
                convs[self.min_lod], kv_index, octrees_in.batch_nnum[self.min_lod].tolist())

        # Perform cross attention
        query_features = self.mask_token.weight.repeat(int(octrees_mid.nnum[self.min_lod]), 1)
        query_features = self.mae_decoder(
            query_features, q_index, octrees_mid.batch_nnum[self.min_lod].tolist(),
            grid_features, kv_index, octrees_in.batch_nnum[self.min_lod].tolist())

        output = self.pred_head[0](query_features)
        occ = output[:, :2]
        data = {'occs': [occ]}

        query_features, _ = octree_align(query_features, octrees_mid, octrees_out, self.min_lod, nempty=False)
        if self.update_octree:
            split = occ.argmax(1).int()
            octrees_out.octree_split(split, self.min_lod)
            octrees_out.octree_grow(self.min_lod+1)
        else:
            gt_occ = octree_search(octrees_out, octrees_mid, self.min_lod, nempty=True)
            gt_occ = ocnn.nn.octree_pad(gt_occ.unsqueeze(-1), octrees_mid, self.min_lod, 0.0).squeeze(-1)
            data['gt_occs'] = [gt_occ.long()]

        for l, d in enumerate(range(self.min_lod, self.max_lod)):
            query_features = self.upsample[l](query_features, octrees_out, d)
            skip = ocnn.nn.octree_align(convs[d+1], octrees_in, octrees_out, d+1, nempty=False)
            query_features = query_features + skip
            query_features = self.decoder_blks[l](query_features, octrees_out, depth=d+1)
            output = self.pred_head[l+1](query_features)
            occ = output[:, :2]
            data['occs'].append(occ)
            if self.update_octree:
                split = occ.argmax(1).int()
                octrees_out.octree_split(split, d+1)
                if d < (self.max_lod - 1):
                    octrees_out.octree_grow(d+2)
            else:
                gt_occ = octrees_out.nempty_mask(d+1).long()
                data['gt_occs'].append(gt_occ)

        data['signal'] = ocnn.nn.octree_depad(output[:, 2:], octrees_out, self.max_lod)
        if self.update_octree:
            octrees_out.normals[self.max_lod] = data['signal'][:, :3]
            octrees_out.features[self.max_lod] = data['signal'][:, 3:]
        else:
            data['gt_signal'] = self.get_ground_truth_signal(octrees_out)
        data['octrees_out'] = octrees_out

        # end.record()
        # th.cuda.synchronize()
        # data['runtime_reg'] = start.elapsed_time(end)

        return data
