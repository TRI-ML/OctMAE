import argparse

import yaml


def parse_config(config_file_path=None):
    parser = argparse.ArgumentParser('Train a network for 3D reconstruction from a single stereo image.')
    parser.add_argument('--config', default='configs/mirage/config.yaml', help='config file')

    # General parameters
    parser.add_argument('--project_name', type=str, default='octmae')
    parser.add_argument('--model_name', type=str, default='mirage')
    parser.add_argument('--run_name', type=str, help='Run name of WandB')
    parser.add_argument('--train_dataset_name', type=str, default='mirage', help='Evaluation dataset name')
    parser.add_argument('--eval_dataset_name', type=str, default=None, help='Evaluation dataset name')

    # Training parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file')
    parser.add_argument('--train_dataset_url', type=str, default='', help='URL to a webdataset for training')
    parser.add_argument('--val_dataset_url', type=str, default='', help='URL to a webdataset for validation')
    parser.add_argument('--train_dataset_size', type=int, default=999600)
    parser.add_argument('--val_dataset_size', type=int, default=400)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scheduler_step', type=int, default=3000)
    parser.add_argument('--scheduler_decay', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)

    # For demo
    parser.add_argument('--img_path', type=str, help='Image path for demo')
    parser.add_argument('--depth_path', type=str, help='Depth map path for demo')
    parser.add_argument('--mask_path', type=str, help='Mask path for demo')
    parser.add_argument('--camera_info_path', type=str, help='Camera info path for demo')

    # Mirage
    parser.add_argument('--grid_size', type=float, default=0.2, help='Grid size in meter')
    parser.add_argument('--num_enc_layers', type=int, default=2, help='Number of layers for PIVOT encoder')
    parser.add_argument('--num_dec_layers', type=int, default=2, help='Number of layers for PIVOT decoder')
    parser.add_argument('--update_octree', default=False, action='store_true', help='Should update an octree for prediction?')
    parser.add_argument('--init_lod', type=int, default=6, help='Initial LoD')
    parser.add_argument('--max_lod', type=int, default=9, help='Number of maximum LoD')
    parser.add_argument('--min_lod', type=int, default=6, help='Number of maximum LoD')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads for multi-head attention (MHA)')
    parser.add_argument('--max_freq', type=int, default=128, help='Max freq (resolution) of positional encoding')
    parser.add_argument('--pos_emb_dim', type=int, default=32, help='Positional embedding dimension for Transformer')
    parser.add_argument('--dim_model', type=int, default=32, help='Latent feature dimension for Backbone')
    parser.add_argument('--dim_mae', type=int, default=64, help='Latent feature dimension for Transformer')
    parser.add_argument('--resid_dropout', type=float, default=0.0, help='Dropout rate for MHA')
    parser.add_argument('--ff_dropout', type=float, default=0.1, help='Dropout rate for the feedforward network')
    parser.add_argument('--ff_activation', type=str, default='gelu', help='Activation function for the feedforward network')
    parser.add_argument('--ff_hidden_layer_multiplier', type=int, default=4, help='Hidden layer multiplier for the feedforward network')
    parser.add_argument('--vot_scale_factor', type=int, default=2, help='Scale factor of an image for a voxel occlusion tester')

    args, _ = parser.parse_known_args()
    args = vars(args)
    args_default = {k: parser.get_default(k) for k in args}
    if config_file_path is None:
        args_config = yaml.load(open(args['config']), Loader=yaml.FullLoader)
    else:
        args_config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    args_inline = {k: v for (k, v) in args.items() if v != args_default[k]}
    args = args_default.copy()
    args.update(args_config)
    args.update(args_inline)
    args = argparse.Namespace(**args)
    print(args)
    return args
