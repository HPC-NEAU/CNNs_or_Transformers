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
import torch
import torch.utils.data as data
from adabelief_pytorch import AdaBelief
from catalyst import dl
from catalyst.contrib.nn import CircleLoss, FocalLossMultiClass, OneCycleLRWithWarmup
from catalyst.data import BatchBalanceClassSampler, BalanceClassSampler
from numpy import ceil
from sklearn.model_selection import StratifiedKFold
from torch import binary_cross_entropy_with_logits, optim, nn
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import config_moco
from criterion.SmoothBCELogits import SmoothBCEwLogits
from dataset.dataset_sl import get_dataset
from models.model_simsiam import get_model


def k_fold(k, df_all):
    skf = StratifiedKFold(n_splits=k, random_state=2112007044, shuffle=True)
    splits = list(skf.split(X=df_all, y=df_all["kind"]))  # 返回k—fold数据集（5组训练集和验证集）的索引（一行对于一个索引）

    for i, (train_idx, val_idx) in enumerate(splits):
        # print(f"\n-------------   Fold {i + 1} / {k}  -------------\n")
        # df.iloc[i，j] 拿出df的第i行与第j行（第i个样本和第j个样本）
        df_train = df_all.iloc[train_idx].copy()  # 通过train_idx来从dataset中挑选出训练dataset
        df_val = df_all.iloc[val_idx].copy()  # 通过test_idx来从dataset中挑选出训练dataset
    return df_train, df_val


if __name__ == '__main__':
    df_all = pd.read_csv(r"../data/df_train.csv", dtype="object")
    df_train, df_val = k_fold(5, df_all)

    # train_labels = config.Data.NUM_CLASSES_LIST
    # val_labels = df_val["bird_kind"].to_list()
    # train_labels = df_train["bird_kind"].to_list()
    # train_sampler = BalanceClassSampler(train_labels, mode=250)
    # val_sampler = BalanceClassSampler(val_labels, mode=50)

    Birddataset_train = get_dataset(df=df_train)
    Birddataset_val = get_dataset(df=df_val)
    # train_sampler = ImbalancedDatasetSampler(Birddataset_train)

    model_name = "resnet-50-finetune"
    print(f"{model_name} is training" )

    model = get_model(mode="ds",
                      Test=False,
                      load_pretrain=True,
                      linear_prone=False,
                      weights_path=r"..\checkpoint\2022-03-21\resnet50-simsiam\checkpoints\best.pth"
                      )

    # loaders
    loaders = {
        "train": data.DataLoader(Birddataset_train,
                                 batch_size=config_moco.batch_size,
                                 # sampler=train_sampler,
                                 shuffle=True,
                                 num_workers=6,
                                 pin_memory=True,
                                 drop_last=True),

        "valid": data.DataLoader(Birddataset_val,
                                 batch_size=config_moco.batch_size,
                                 # sampler=val_sampler,
                                 # shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=True)
    }

    # Optimizer
    optimizer = AdaBelief(model.parameters(), lr=config_moco.lr, eps=1e-8, weight_decay=1e-2, betas=(0.9, 0.999), weight_decouple=True,
                          rectify=False)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-2, momentum=0.9)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_moco.T_max, eta_min=config_moco.eta_min)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config_moco.T_0, T_mult=1, eta_min=config_moco.eta_min)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config_moco.lr, steps_per_epoch=len(loaders["train"]), verbose=True, epochs=config_moco.epochs)
    # scheduler = OneCycleLRWithWarmup(optimizer, num_steps=122, lr_range=(0.001, 0.00001), init_lr=0.0001, warmup_steps=22)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-4, verbose=True)

    # Loss
    # criterion = CircleLoss(margin=0.25, gamma=1)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.Cross()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLossMultiClass()
    criterion = SmoothBCEwLogits()
    # criterion = AMSoftmax(in_feats=1792, n_classes=100)

    callbacks = [
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", num_classes=config_moco.Data.NUM_CLASSES
        ),
        # dl.PrecisionRecallF1SupportCallback(
        #     input_key="scores", target_key="targets", num_classes=config.Data.NUM_CLASSES
        # ),
        dl.SchedulerCallback()
    ]

    TODAY = str(datetime.date.today())
    CP_TODAY = f"../checkpoint/{TODAY}/"

    if not os.path.exists(CP_TODAY):
        os.makedirs(CP_TODAY)

    checkpoint_path = f"../checkpoint/{TODAY}/{model_name}"

    warnings.simplefilter("ignore")

    runner = dl.SupervisedRunner(  # catalyst的工具，https://github.com/catalyst-team/catalyst
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )


    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config_moco.epochs,
        verbose=True,
        logdir=checkpoint_path,
        callbacks=callbacks,
        valid_metric='loss',
        valid_loader="valid"
)