import random

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchsummary import summary


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()              # stop gradient
        p = F.normalize(p, dim=1)   # l2-normalize
        z = F.normalize(z, dim=1)   # l2-normalize
        return -(p * z).sum(dim=1).mean()


class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        backbone = timm.create_model(model_name="resnet50",
                                       pretrained=False,
                                       in_chans=3,
                                       drop_rate=0.2,
                                       )
        proj_hid, proj_out = 2048, 2048
        pred_hid, pred_out = 512, 2048


        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone_in_channels = backbone.fc.in_features

        self.projection = nn.Sequential(
            nn.Linear(backbone_in_channels, backbone_in_channels),
            nn.BatchNorm1d(backbone_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_in_channels, backbone_in_channels),
            nn.BatchNorm1d(backbone_in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_in_channels, proj_out),
            nn.BatchNorm1d(proj_out)
        )

        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hid, pred_out),
        )

        self.d = D()


    def forward(self, x1, x2):

        out1 = self.backbone(x1).squeeze()
        z1 = self.projection(out1)
        p1 = self.prediction(z1)

        out2 = self.backbone(x2).squeeze()
        z2 = self.projection(out2)
        p2 = self.prediction(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.
        loss = d1+d2

        return loss




class DownStreamModel(nn.Module):
    def __init__(self, simsiam, n_classes=23, linear_prone=False):
        super(DownStreamModel, self).__init__()
        hidden = 512

        self.net_backbone = nn.Sequential(
            simsiam.backbone,
        )

        self.classifier = nn.Linear(2048, n_classes)

        self.net_projection = nn.Sequential(
            simsiam.projection,
        )
        if linear_prone == True:
            # for name, param in self.net_projection.named_parameters():
            #     param.requires_grad = False
            # for name, param in self.net_backbone.named_parameters():
            #     param.requires_grad = False
            for name, param in self.classifier.named_parameters():
                param.requires_grad = False

        # self.out = nn.Sequential(
        #     nn.Linear(2048, hidden),
        #     nn.BatchNorm1d(hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden, n_classes),
        # )

    def forward(self, x):
        out = self.net_backbone(x).squeeze()
        out = self.net_projection(out)
        out = self.classifier(out)
        # out = self.out(out)

        return out

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SimSiam()
    # model.to(device)
    # summary(model, (1, 128, 312))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SimSiam = SimSiam()
    model = DownStreamModel(SimSiam)
    model.to(device)
    summary(model, (1, 128, 312))