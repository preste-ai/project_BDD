import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from DRAEM.model_unet import ReconstructiveSubNetwork
from mvtec_dataset import MVTecDataset, BDDDataset
from collections import OrderedDict



# class Convnet(nn.Module):
#
#     def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
#         super().__init__()
#         self.encoder = l2l.vision.models.CNN4Backbone(
#             hidden=hid_dim,
#             channels=x_dim,
#             max_pool=True,
#         )
#         self.out_channels = 1600
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)


class ProtoBase(nn.Module):
    def __init__(self, encoder, base_width=32, n_output=2):
        super(ProtoBase, self).__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(OrderedDict([
            ('h_mp1', nn.MaxPool2d(2)),
            ('h_conv1', nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1, stride=2)),
            #('h_batch_norm1', nn.BatchNorm2d(base_width * 4)),
            ('h_relu1', nn.ReLU(inplace=True)),
            ('h_conv2', nn.Conv2d(base_width * 4, base_width, kernel_size=3, padding=1, stride=2)),
            #('h_batch_norm2', nn.BatchNorm2d(base_width)),
            ('h_relu2', nn.ReLU(inplace=True)),
            ('h_flatten', nn.Flatten()),
            #('h_dropout', nn.Dropout(p=0.25)),
            #('h_linear1', nn.Linear(base_width * 4, 256)),
            #('h_relu3', nn.ReLU(inplace=True)),
            #('h_linear2', nn.Linear(128, 32)),
            #('h_relu4', nn.ReLU(inplace=True)),
            #('h_linear3', nn.Linear(32, n_output))
        ])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)



def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    #print('v1: data :', data.shape)
    #print('v1: labels :', labels.shape)
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    #print('support_indices ->', support_indices)
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    #print('v1: support | ', support.shape)
    support = support.reshape(ways, shot, -1).mean(dim=1)
    #print('v1: support | ', support.shape)
    query = embeddings[query_indices]
    #print('v1: query   | ', query.shape)
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def prepare_task_sets(ds, query, ways, shots):
    # Create Tasksets using the benchmark interface
    dataset = l2l.data.MetaDataset(ds)  # any PyTorch dataset
    transforms = [  # Easy to define your own transform
        NWays(dataset, ways),
        KShots(dataset, query + shots),
        LoadData(dataset),
        RemapLabels(dataset),
    ]
    taskset = l2l.data.TaskDataset(dataset, transforms)
    return taskset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--shot', type=int, default=8)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--test-shot', type=int, default=8)
    parser.add_argument('--test-query', type=int, default=8)
    parser.add_argument('--train-query', type=int, default=8)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(42)
        device = torch.device('cuda')
    random.seed(42)
    # Create model
    reconstructive_pretrain_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=32)
    reconstructive_pretrain_model.load_state_dict(
        torch.load("DRAEM/checkpoints//home/preste-nakam/Documents/projects/project_BDD/train_proto.py", map_location='cuda:0'))
    model = ProtoBase(reconstructive_pretrain_model.encoder,  base_width=32)
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    model.to(device)

    train_classes_names = ['wood', 'capsule', 'cable', 'carpet', 'grid', 'hazelnut', 'zipper', 'dagm_c1', 'dagm_c3', 'dagm_c5', 'kolectorsdd2_train']
    raw_train_ds_list = [MVTecDataset(root_path='data', 
                                      class_names_list=[class_n], 
                                      is_train=False, 
                                      resize=(256,256)) for class_n in train_classes_names]
    
    val_classes_names = ['bottle','dagm_c2']
    raw_val_ds_list = [BDDDataset(root_path='data',
                       bdd_folder_path = 'mvtech_cleaned',
                       class_names_list=[class_n], 
                       is_train=False,
                       resize=(256,256),
                       ways = args.test_way,
                       shots = args.test_shot,
                       query = args.test_query) for class_n in val_classes_names]

    test_classes_names = ['pill', 'tile', 'leather', 'dagm_c4', 'dagm_c6', 'kolectorsdd2_test']
    raw_test_ds_list = [BDDDataset(root_path='data',
                        bdd_folder_path = 'mvtech_cleaned',
                        class_names_list=[class_n],
                        is_train=False,
                        resize=(256,256),
                        ways = args.test_way,
                        shots = args.test_shot,
                        query = args.test_query) for class_n in test_classes_names]
    

    train_taskset_list = [prepare_task_sets(ds, args.train_query, args.train_way, args.shot) for ds in raw_train_ds_list] # l2l.data.TaskDataset(dataset_train, transforms_train)
    #val_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_val_ds_list]     # l2l.data.TaskDataset(dataset_val, transforms_val)
    #test_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_test_ds_list]   # l2l.data.TaskDataset(dataset_test, transforms_test)

    train_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in train_taskset_list]
    val_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_val_ds_list]
    test_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_test_ds_list]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)
    
    run_name = 'BDD_train_lr_0008'+'_e_'

    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(10):

            batch = next(iter(random.choice(train_loader_list)))

            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
              epoch, n_loss / loss_ctr, n_acc / loss_ctr))
        
        if (epoch+1) % 25 == 0:
            torch.save(model.state_dict(), os.path.join('./checkpoints/', run_name+str(epoch+1)+".pckl"))
        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for v_i, val_loader in enumerate(val_loader_list):
            for i, batch in enumerate(val_loader):
                loss, acc = fast_adaptv2(model,
                                       batch,
                                       args.test_way,
                                       args.test_shot,
                                       args.test_query,
                                       metric=pairwise_distances_logits,
                                       device=device)

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc

            print('CLASS {}: epoch {}, val, loss={:.4f} acc={:.4f}'.format(
                  val_classes_names[v_i], epoch, n_loss / loss_ctr, n_acc / loss_ctr))

    model.eval()
    for c_i, test_loader in enumerate(test_loader_list):
        loss_ctr = 0
        n_acc = 0
        for i, batch in enumerate(test_loader):
            loss, acc = fast_adaptv2(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)
            loss_ctr += 1
            n_acc += acc
    
        print('CLASS {}: {}: {:.2f}({:.2f})'.format(
                test_classes_names[c_i], i, n_acc / loss_ctr * 100, acc * 100))

