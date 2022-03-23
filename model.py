import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from collections import OrderedDict
from torchvision.models import resnet18


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

    def forward(self, x):  # x - image
        x = self.encoder(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1) # flatten


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
        o = self.classifier(m)
        return o


class ProtoForRes18(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(ProtoForRes18, self).__init__()
        self.backbone = backbone

    def forward(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


class ProtoNetv3(nn.Module):
    def __init__(self, encoder=resnet18(pretrained=True), base_width=32):
        super(ProtoNetv3, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Linear(512, 1)

    def forward(self, x):  # x - image
        x = self.encoder(x)
        #x = self.classifier(x)
        return x.view(x.size(0), -1) # flatten
