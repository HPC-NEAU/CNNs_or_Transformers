# -*- coding: UTF-8 -*-
'''
@Author  ：LQR
@Project ：bird_yu1 
@File    ：config.py
@IDE     ：PyCharm 
@Date    ：2021/6/5 20:24 
'''
import os
import torch

sys_config = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "seed": 1,
}




class Data:
    # 以训练为准
    cut_out_path = r"/mnt/inspurfs/user-fs/HPC01/simsiam/data/maize_disease_small"
    NUM_CLASSES_LIST = sorted(os.listdir(cut_out_path))
    NUM_CLASSES = len(NUM_CLASSES_LIST)
    CODE = {j:i for i, j in enumerate(NUM_CLASSES_LIST)}
    INV_CODE = {v: k for k, v in CODE.items()}

class dataset_config:

    model_name = "efficientnet_b0_(lp_train_100%)"




model_config = {
    "weights_path": rf"G:\LQR\bird_byol\checkpoint\2022-02-19\efficientnet_b0_mel_nobal_noaug_(f_num_64,hop_size_513,mix_num_0)\checkpoints\best.pth",

    "config": {
    }
}

lr = 1e-3
eta_min = 1e-5
T_max = 150
epochs = 150
batch_size = 32


if __name__ == '__main__':
    cut_out_path = r"F:\kaggle\competitions\birdclef-2021\birdclef-2021\train_short_audio"
    filelist = sorted(os.listdir(cut_out_path))
    filedict = {j:i for i, j in enumerate(filelist)}
