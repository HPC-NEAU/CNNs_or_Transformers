# -*- coding: UTF-8 -*-
'''
@Author  ：LQR
@Project ：bird_sed_c
@File    ：train_efficientnet.py
@IDE     ：PyCharm
@Date    ：2021/7/1 11:53
'''
import datetime
import os
import warnings
import pandas as pd
import torch.utils.data as data
from adabelief_pytorch import AdaBelief
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms

import config_moco
from dataset.dataset_ssl import get_dataset
from models.model_simsiam import get_model
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl, metrics

def k_fold(k, df_all):
    skf = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
    splits = list(skf.split(X=df_all, y=df_all["kind"]))  # 返回k—fold数据集（5组训练集和验证集）的索引（一行对于一个索引）

    for i, (train_idx, val_idx) in enumerate(splits):
        # print(f"\n-------------   Fold {i + 1} / {k}  -------------\n")
        # df.iloc[i，j] 拿出df的第i行与第j行（第i个样本和第j个样本）
        df_train = df_all.iloc[train_idx].copy()  # 通过train_idx来从dataset中挑选出训练dataset
        df_val = df_all.iloc[val_idx].copy()  # 通过test_idx来从dataset中挑选出训练dataset
    return df_train, df_val


class CustomRunner(dl.Runner):

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def handle_batch(self, batch):
        x1 = batch["features1"]
        x2 = batch["features2"]
        loss = self.model(x1, x2)
        loss = loss["loss"]

        # Update metrics (includes the metric that tracks the loss)
        self.batch_metrics.update({"loss": loss})

        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)



if __name__ == '__main__':
    df_all = pd.read_csv(r"../data/df_train.csv", dtype="object")
    df_train, df_val = k_fold(5, df_all)

    # train_labels = config.Data.NUM_CLASSES_LIST
    val_labels = df_val["kind"].to_list()
    train_labels = df_train["kind"].to_list()
    # train_sampler = BalanceClassSampler(train_labels, mode=250)
    # val_sampler = BalanceClassSampler(val_labels, mode=50)

    dataset_train = get_dataset(df=df_train)
    dataset_val = get_dataset(df=df_val)

    model_name = "resnet50-simsiam"
    print(f"{model_name} is training")


    model = get_model(mode="ssl",
                      Test=False,
                      load_pretrain=False,
                      linear_prone=False,
                      weights_path=None
                      )

    # loaders
    loaders = {
        "train": data.DataLoader(dataset_train,
                                 batch_size=config_moco.batch_size,
                                 # sampler=train_sampler,
                                 shuffle=True,
                                 num_workers=6,
                                 pin_memory=True,
                                 drop_last=True),

        "valid": data.DataLoader(dataset_val,
                                 batch_size=config_moco.batch_size,
                                 # sampler=val_sampler,
                                 # shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=True)
    }

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=config_moco.lr, weight_decay=1e-4, momentum=0.9)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_moco.T_max, eta_min=config_moco.eta_min)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config_moco.T_0, T_mult=1, eta_min=config_moco.eta_min)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(loaders["train"]), verbose=True, epochs=config.epochs)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-4, verbose=True)

    # Loss
    # criterion = CircleLoss(margin=0.25, gamma=1)
    # criterion = nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLossMultiClass()
    # criterion = SmoothBCEwLogits()
    # criterion = AMSoftmax(in_feats=1792, n_classes=100)

    callbacks = [
        dl.OptimizerCallback(metric_key="loss"),
        dl.SchedulerCallback()
    ]

    TODAY = str(datetime.date.today())
    CP_TODAY = f"../checkpoint/{TODAY}/"

    if not os.path.exists(CP_TODAY):
        os.makedirs(CP_TODAY)

    checkpoint_path = f"../checkpoint/{TODAY}/{model_name}"

    warnings.simplefilter("ignore")

    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        # criterion=criterion,  # you could also pass any PyTorch criterion for loss computation
        loaders=loaders,
        num_epochs=config_moco.epochs,
        verbose=True,
        callbacks=callbacks,
        logdir=checkpoint_path,
        valid_metric='loss',
        valid_loader="train"
    )