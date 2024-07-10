import os

import torch as th
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification import binary_f1_score, binary_accuracy, binary_recall, binary_precision

import webdataset as wds

from octmae.nets import OctMAE
from octmae.utils.config import parse_config
from octmae.utils.misc import make_sample_wrapper, decode_depth

th.set_float32_matmul_precision('medium')
th.backends.cuda.matmul.allow_tf32 = True

INTRINSICS_K = {
    'mirage': [[572.41136339, 0., 325.2611084], [0., 573.57043286, 242.04899588], [0., 0., 1.]],
    'ycb_video': [[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]],
    'hope': [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    'hb': [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
}

class BaseTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model = OctMAE(config)

    def forward(self, batch):
        output = self.model(batch)
        occs = output['occs']
        gt_occs = output['gt_occs']

        loss_occ = 0
        acc, rec, pre, f1 = 0.0, 0.0, 0.0, 0.0
        for occ, gt_occ in zip(occs, gt_occs):
            loss_occ += F.cross_entropy(occ, gt_occ)
            preds = occ.argmax(1).long()
            acc += binary_accuracy(preds, gt_occ.long())
            rec += binary_recall(preds, gt_occ.long())
            pre += binary_precision(preds, gt_occ.long())
            f1 += binary_f1_score(preds, gt_occ.long())
        acc /= len(occs)
        rec /= len(occs)
        pre /= len(occs)
        f1 /= len(occs)
        loss_dict = {'loss_occ': loss_occ}
        stats_dict = {'acc': acc, 'rec': rec, 'pre': pre, 'f1': f1}
        if 'signal' in output:
            signal = output['signal']
            gt_signal = output['gt_signal']
            loss_nrm = th.mean(th.sum((gt_signal[:, :3] - signal[:, :3])**2, dim=1))
            loss_sdf = th.mean((gt_signal[:, 3:] - signal[:, 3:])**2) # this is a hyperparameter
            loss_dict['loss_nrm'] = loss_nrm
            loss_dict['loss_sdf'] = loss_sdf

        return loss_dict, stats_dict

    def training_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = 0.0
        for name, value in loss_dict.items():
            loss += value
            self.log(f"train_{name}", value, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        for name, value in stats_dict.items():
            self.log(f"train_{name}", value, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True )
        self.log(f"train_loss", loss, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = 0.0
        for name, value in loss_dict.items():
            loss += value
            self.log(f"valid_{name}", value, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        for name, value in stats_dict.items():
            self.log(f"valid_{name}", value, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log(f"valid_loss", loss, on_step=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        if self.config.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            raise Exception(f'{self.config.optimizer} is not supported!')
        scheduler = th.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=120000, power=0.9)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def train_dataloader(self):
        url = f'pipe:s5cmd cat {self.config.train_dataset_url}'
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers
        max_epochs = self.config.max_epochs
        dataset_size = self.config.train_dataset_size
        iter_per_epoch = dataset_size // (batch_size * self.trainer.num_devices)

        dataset = (
            wds.WebDataset(url, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue, shardshuffle=True)
            .decode(decode_depth, 'pil')
            .map(make_sample_wrapper(self.config, K=INTRINSICS_K[self.config.train_dataset_name]), handler=wds.warn_and_continue)
            .batched(batch_size, partial=True)
        )

        dataloader = (
            wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=False)
            .repeat(max_epochs)
            .with_epoch(iter_per_epoch)
            .with_length(iter_per_epoch)
        )

        return dataloader

    def val_dataloader(self):
        url = f'pipe:s5cmd cat {self.config.val_dataset_url}'
        dataset_size = self.config.val_dataset_size

        dataset = (
            wds.WebDataset(url, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue, shardshuffle=False)
            .decode(decode_depth, 'pil')
            .map(make_sample_wrapper(self.config, K=INTRINSICS_K[self.config.train_dataset_name]), handler=wds.warn_and_continue)
            .batched(1)
        )

        dataloader = (
            wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=False)
            .with_epoch(dataset_size // (self.trainer.num_devices))
            .with_length(dataset_size // (self.trainer.num_devices))
        )

        return dataloader


def main():
    config = parse_config()

    model = BaseTrainer(config)
    # Store configurations in WandB
    checkpoint_path = os.path.join('checkpoints', config.project_name, config.run_name)
    callbacks = [ModelCheckpoint(dirpath=checkpoint_path, save_top_k=-1, save_on_train_epoch_end=True, every_n_train_steps=5000)]
    logger = WandbLogger(project=config.project_name, name=config.run_name, log_model=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    trainer = pl.Trainer(max_epochs=config.max_epochs,
                         logger=logger,
                         log_every_n_steps=config.log_every_n_steps,
                         strategy='ddp_find_unused_parameters_true',
                         gradient_clip_val=0.5,
                         callbacks=callbacks)

    if trainer.global_rank == 0:
        logger.experiment.config.update(config)
    trainer.fit(model=model, ckpt_path=config.checkpoint)


if __name__ == '__main__':
    main()
