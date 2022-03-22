import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from collections import OrderedDict


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            hidden=hid_dim,
            channels=x_dim,
            max_pool=True,
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder, base_width=32):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(OrderedDict([
            ('h_conv1', nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm1', nn.BatchNorm2d(base_width * 4)),
            ('h_relu1', nn.ReLU(inplace=True)),
            ('h_mp1', nn.MaxPool2d(2)),
            ('h_conv2', nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm2', nn.BatchNorm2d(base_width * 2)),
            ('h_relu2', nn.ReLU(inplace=True)),
            ('h_flatten', nn.Flatten()),
            # ('h_dropout', nn.Dropout(p=0.25)),
            # ('h_linear1', nn.Linear(base_width * 32, 512)),
            # ('h_relu3', nn.ReLU(inplace=True)),
            # ('h_linear2', nn.Linear(128, 32)),
            # ('h_relu4', nn.ReLU(inplace=True)),
            # ('h_linear3', nn.Linear(32, n_output))
            ])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


class ProtoNetv2(nn.Module):
    def __init__(self, encoder, seg, base_width, n_output=2):
        super(ProtoNetv2, self).__init__()
        self.encoder = encoder
        self.seg = seg

        self.classifier = nn.Sequential(OrderedDict([
            ('h_conv1', nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm1', nn.BatchNorm2d(16)),
            ('h_relu1', nn.ReLU(inplace=True)),
            ('h_mp1', nn.MaxPool2d(2)),
            ('h_conv2', nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm2', nn.BatchNorm2d(32)),
            ('h_relu2', nn.ReLU(inplace=True)),
            ('h_mp2', nn.MaxPool2d(2)),
            ('h_conv3', nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm3', nn.BatchNorm2d(64)),
            ('h_relu3', nn.ReLU(inplace=True)),
            ('h_flatten', nn.Flatten()),
            # ('h_dropout', nn.Dropout(p=0.25)),
            # ('h_linear1', nn.Linear(16*4*4, 64)),
            # ('h_relu3', nn.ReLU(inplace=True)),
            # ('h_linear2', nn.Linear(64, 16)),
            # ('h_relu4', nn.ReLU(inplace=True)),
            # ('h_linear3', nn.Linear(16, n_output))
        ])
        )

    def forward(self, x):
        e = self.encoder(x)
        joined_in = torch.cat((e, x), dim=1)
        m = self.seg(joined_in)
        #o = self.classifier(m)
        m = m[:, 1:, :, :]
        return m


class SegNet(nn.Module):
    def __init__(self, reconstruct, seg):
        super(SegNet, self).__init__()
        self.reconstruct = reconstruct
        self.seg = seg

    def forward(self, x):
        r = self.reconstruct(x)
        joined_in = torch.cat((r, x), dim=1)
        m = self.seg(joined_in)
        m_f = m[:,1:,:,:]
        return m_f.view(m_f.size(0), -1)


class SegNetv3(nn.Module):
    def __init__(self, reconstruct, seg):
        super(SegNetv3, self).__init__()
        self.reconstruct = reconstruct
        self.seg = seg

    def forward(self, x):
        r = self.reconstruct(x)
        joined_in = torch.cat((r, x), dim=1)
        m = self.seg(joined_in)
        m_f = m[:,1:,:,:]
        return m, m_f.view(m_f.size(0), -1)
