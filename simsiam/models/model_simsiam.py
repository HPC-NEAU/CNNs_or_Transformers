# -*- coding: UTF-8 -*-
'''
@Author  ：LQR
@Project ：bird_r
@File    ：efficiv2.py
@IDE     ：PyCharm
@Date    ：2021/7/14 13:35
'''
import numpy as np
import torch
from torch import nn
from torchsummary import summary
import config_moco
from models.simsiam import SimSiam, DownStreamModel
from utils.freeze_weight import freeze_by_idxs


class Model(nn.Module):
    def __init__(self,
                 ):
        super().__init__()



        self.SimSiam = SimSiam()

    def forward(self, x1, x2):
        """
            Input:
        """
        loss = self.SimSiam(x1, x2)
        return {"loss": loss
                }



def get_model(weights_path: str, mode:str, load_pretrain:str, Test:str, linear_prone = True):
    if mode == "ssl":
       model = Model()
       if load_pretrain == True:
            state_dict = torch.load(weights_path, map_location=config_moco.sys_config["device"])
            model.load_state_dict(state_dict['model_state_dict'], strict=True)

    elif mode == "ds":
        model = Model()
        state_dict = torch.load(weights_path, map_location=config_moco.sys_config["device"])
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        SimSiam = model.SimSiam
        model = DownStreamModel(SimSiam, n_classes=3, linear_prone=linear_prone)

    elif mode == "sl":
        model = Model()
        SimSiam = model.SimSiam
        model = DownStreamModel(SimSiam, n_classes=3, linear_prone=linear_prone)
        if load_pretrain == True:
            state_dict = torch.load(weights_path, map_location=config_moco.sys_config["device"])
            model.load_state_dict(state_dict['model_state_dict'], strict=True)




    if Test == True:
        model.eval()
    else:
        model.train()

    model.to(config_moco.sys_config["device"])
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=100)
    model.to(device)
    summary(model, (3, 26, 312))
