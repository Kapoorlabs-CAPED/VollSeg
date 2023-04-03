import os
from pathlib import Path
from .cellposeutils3D import (
    prepare_images,
    prepare_masks,
    create_csv,
    save_json,
)
from .TrainTiledLoader import TrainTiled
from torch.utils.data import DataLoader
from .UNet3D import UNet3D_module
import torch
import torch.nn.functional as F
from torch import optim
from collections import OrderedDict
import lightning.pytorch as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class CellPose3D(pl.LightningModule):
    def __init__(
        self,
        base_dir,
        model_dir,
        model_name,
        patch_size=(8, 256, 256),
        epochs=100,
        in_channels=1,
        out_channels=4,
        feat_channels=16,
        samples_per_epoch=-1,
        batch_size=16,
        learning_rate=0.001,
        background_weight=1,
        flow_weight=1,
        out_activation="tanh",
        raw_dir="raw/",
        real_mask_dir="real_mask/",
        identifier="*.tif",
        save_train="_train.csv",
        save_test="_test.csv",
        save_val="_val.csv",
        axis_norm=(0, 1, 2),
        variance_size=(5, 5, 5),
        fg_footprint_size=5,
        bg_label=0,
        channel=0,
        corrupt_prob=0,
        num_gpu=1,
        zoom_factor=(1, 1, 1),
    ):
        super().__init__()

        self.base_dir = base_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.patch_size = patch_size
        self.epochs = epochs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.background_weight = background_weight
        self.flow_weight = flow_weight
        self.out_activation = out_activation
        self.raw_dir = os.path.join(base_dir, raw_dir)
        self.real_mask_dir = os.path.join(base_dir, real_mask_dir)
        self.identifier = identifier
        self.save_train = save_train
        self.save_test = save_test
        self.save_val = save_val
        self.axis_norm = axis_norm
        self.variance_size = variance_size
        self.fg_footprint_size = fg_footprint_size
        self.channel = channel
        self.bg_label = bg_label
        self.corrupt_prob = corrupt_prob
        self.zoom_factor = zoom_factor
        self.num_gpu = num_gpu
        self.save_raw_h5_name = "raw_h5/"
        self.save_real_mask_h5_name = "real_mask_h5/"

        self.save_raw_h5 = os.path.join(base_dir, self.save_raw_h5_name)
        Path(self.save_raw_h5).mkdir(exist_ok=True)

        self.save_real_mask_h5 = os.path.join(
            base_dir, self.save_real_mask_h5_name
        )
        Path(self.save_real_mask_h5).mkdir(exist_ok=True)
        self.save_hyperparameters()

    def _create_training_h5(self):

        prepare_images(
            data_path=self.raw_dir,
            save_path=self.save_raw_h5,
            identifier=self.identifier,
            axis_norm=self.axis_norm,
            get_distance=False,
            get_illumination=False,
            get_variance=False,
            variance_size=self.variance_size,
            fg_footprint_size=self.fg_footprint_size,
            channel=self.channel,
        )

        prepare_masks(
            data_path=self.real_mask_dir,
            save_path=self.save_real_mask_h5,
            identifier=self.identifier,
            bg_label=self.bg_label,
            get_flows=True,
            get_boundary=False,
            get_seeds=False,
            get_distance=True,
            corrupt_prob=self.corrupt_prob,
            zoom_factor=self.zoom_factor,
        )
        data_list = []
        for imagename in os.listdir(self.save_raw_h5):
            data_list.append(
                [
                    os.path.join(self.save_raw_h5, imagename),
                    os.path.join(self.save_real_mask_h5, imagename),
                ]
            )
        create_csv(
            data_list,
            self.base_dir,
            save_train=self.save_train,
            save_test=self.save_test,
            save_val=self.save_val,
        )
        self.train_list = os.path.join(self.base_dir, self.save_train)
        self.val_list = os.path.join(self.base_dir, self.save_val)
        self.test_list = os.path.join(self.base_dir, self.save_test)
        hparams = {
            "train_list": self.train_list,
            "test_list": self.test_list,
            "val_list": self.val_list,
            "data_root": "",
            "patch_size": self.patch_size,
            "epochs": self.epochs,
            "image_groups": ("data/image"),
            "mask_groups": ("data/distance", "data/seeds", "data/boundary"),
            "dist_handling": "bool_inv",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "feat_channels": self.feat_channels,
            "norm_method": "instance",
            "samples_per_epoch": self.samples_per_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "background_weight": self.background_weight,
            "flow_weight": self.flow_weight,
            "out_activation": self.out_activation,
        }

        save_json(
            hparams, str(self.base_dir) + "/" + self.model_name + ".json"
        )

        train_dataset = TrainTiled(
            hparams["train_list"],
            hparams["data_root"],
            patch_size=hparams["patch_size"],
            image_groups=hparams["image_groups"],
            mask_groups=hparams["mask_groups"],
            dist_handling=hparams["dist_handling"],
            norm_method=hparams["data_norm"],
            sample_per_epoch=hparams["samples_per_epoch"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        val_dataset = TrainTiled(
            hparams["val_list"],
            hparams["data_root"],
            patch_size=hparams["patch_size"],
            image_groups=hparams["image_groups"],
            mask_groups=hparams["mask_groups"],
            dist_handling=hparams["dist_handling"],
            norm_method=hparams["data_norm"],
            sample_per_epoch=hparams["samples_per_epoch"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        self.model = UNet3D_module(
            patch_size=hparams["patch_size"],
            in_channels=hparams["in_channels"],
            out_channels=hparams["out_channels"],
            feat_channels=hparams["feat_channels"],
            out_activation=hparams["out_activation"],
            norm_method=hparams["norm_method"],
        )
        pretrained_file = os.path.join(self.model_dir, self.model_name)
        if os.path.isfile(pretrained_file):
            self.load_pretrained(pretrained_file)

        checkpoint_callback = ModelCheckpoint(
            filepath=pretrained_file,
            save_top_k=1,
            monitor="epoch",
            mode="max",
            verbose=True,
            period=5,
        )

        logger = CSVLogger(
            save_dir=self.base_dir,
            name="lightning_logs_" + pretrained_file.lower(),
        )

        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            logger=logger,
            callbacks=checkpoint_callback,
            min_epochs=self.epochs,
            max_epochs=self.epochs * 2,
            default_root_dir=self.model_dir,
            enable_checkpointing=True,
        )

        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=pretrained_file,
        )

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]

        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]

        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        # Get parameter dict of current model
        param_dict = dict(self.model.named_parameters())

        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(
                    state_dict["network." + layer].data
                )
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.model.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def background_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def flow_loss(self, y_hat, y, mask):
        loss = F.mse_loss(y_hat, y, reduction="none")
        weight = torch.clamp(mask, min=0.01, max=1.0)
        loss = torch.mul(loss, weight)
        loss = torch.sum(loss)
        loss = torch.div(loss, torch.clamp(torch.sum(weight), 1, mask.numel()))
        return loss

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        # Get image ans mask of current batch
        self.last_imgs, self.last_masks = batch["image"], batch["mask"]

        # generate images
        self.predictions = self.forward(self.last_imgs)

        # get the losses
        loss_bg = self.background_loss(
            self.predictions[:, 0, ...], self.last_masks[:, 0, ...]
        )

        loss_flowx = self.flow_loss(
            self.predictions[:, 1, ...],
            self.last_masks[:, 1, ...],
            self.last_masks[:, 0, ...],
        )
        loss_flowy = self.flow_loss(
            self.predictions[:, 2, ...],
            self.last_masks[:, 2, ...],
            self.last_masks[:, 0, ...],
        )
        loss_flowz = self.flow_loss(
            self.predictions[:, 3, ...],
            self.last_masks[:, 3, ...],
            self.last_masks[:, 0, ...],
        )
        loss_flow = (loss_flowx + loss_flowy + loss_flowz) / 3

        loss = self.background_weight * loss_bg + self.flow_weight * loss_flow

        return loss

    def test_step(self, batch, batch_idx):

        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):

        x, y = batch["image"], batch["mask"]
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log(
            f"{prefix}_loss", loss, on_step=True, on_epoch=True, sync_dist=True
        )

    def validation_step(self, batch, batch_idx):

        self._shared_eval(batch, batch_idx, "validation")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        schedular = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=10
        )
        optimizer_scheduler = OrderedDict(
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": schedular,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        )
        return optimizer_scheduler
