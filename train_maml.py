import random
import numpy as np

import torch
from torch import nn, optim
import random

from collections import OrderedDict

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)

from DRAEM.model_unet import ReconstructiveSubNetwork
from mvtec_dataset import MVTecDataset


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


class MamlBase(nn.Module):
    def __init__(self, encoder, base_width, n_output=2):
        super(MamlBase, self).__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(OrderedDict([
            ('h_mp1', nn.MaxPool2d(2)),
            ('h_conv1', nn.Conv2d(base_width * 8, base_width * 2, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm1', nn.BatchNorm2d(base_width * 2)),
            ('h_relu1', nn.ReLU(inplace=True)),
            ('h_conv2', nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1, stride=2)),
            ('h_batch_norm2', nn.BatchNorm2d(base_width)),
            ('h_relu2', nn.ReLU(inplace=True)),
            ('h_flatten', nn.Flatten()),
            ('h_dropout', nn.Dropout(p=0.25)),
            ('h_linear1', nn.Linear(base_width * 2 * 2, 128)),
            ('h_relu3', nn.ReLU(inplace=True)),
            ('h_linear2', nn.Linear(128, 32)),
            ('h_relu4', nn.ReLU(inplace=True)),
            ('h_linear3', nn.Linear(32, n_output))
        ])
        )

    def forward(self, x):
        e = self.encoder(x)
        o = self.classifier(e)
        return o


def prepare_task_sets(ds, ways, shots):
    # Create Tasksets using the benchmark interface
    dataset = l2l.data.MetaDataset(ds)  # any PyTorch dataset
    transforms = [  # Easy to define your own transform
        l2l.data.transforms.NWays(dataset, n=ways),
        l2l.data.transforms.KShots(dataset, k=shots * 2),  # . .  *2 ???
        l2l.data.transforms.LoadData(dataset)
    ]
    taskset = l2l.data.TaskDataset(dataset, transforms)
    return taskset

def main(
        ways=2,
        shots=10,
        meta_lr=0.003,
        fast_lr=0.005,
        meta_batch_size=16,
        adaptation_steps=5,
        num_iterations=20,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    #
    raw_train_ds_list = [MVTecDataset(root_path='data', class_names_list=[class_n], is_train=False, resize=256) for class_n in ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                                                                    'hazelnut', 'leather']]
    raw_val_ds_list = [MVTecDataset(root_path='data', class_names_list=[class_n], is_train=False, resize=256) for class_n in ['metal_nut', 'pill', 'screw']]

    raw_test_ds_list = [MVTecDataset(root_path='data', class_names_list=[class_n], is_train=False, resize=256) for class_n in ['tile', 'toothbrush', 'transistor', 'wood', 'zipper']]

    # Create Tasksets using the benchmark interface
    #dataset_train = l2l.data.MetaDataset(raw_train_ds)  # any PyTorch dataset
    # transforms_train = [  # Easy to define your own transform
    #     l2l.data.transforms.NWays(dataset_train, n=ways),
    #     l2l.data.transforms.KShots(dataset_train, k=shots * 2),  # . .  *2 ???
    #     l2l.data.transforms.LoadData(dataset_train)
    # ]

    #dataset_val = l2l.data.MetaDataset(raw_val_ds)  # any PyTorch dataset
    # transforms_val = [  # Easy to define your own transform
    #     l2l.data.transforms.NWays(dataset_val, n=ways),
    #     l2l.data.transforms.KShots(dataset_val, k=shots * 2),
    #     l2l.data.transforms.LoadData(dataset_val)
    # ]

    #dataset_test = l2l.data.MetaDataset(raw_test_ds)  # any PyTorch dataset
    # transforms_test = [  # Easy to define your own transform
    #     l2l.data.transforms.NWays(dataset_test, n=ways),
    #     l2l.data.transforms.KShots(dataset_test, k=shots * 2),
    #     l2l.data.transforms.LoadData(dataset_test)
    # ]

    train_taskset_list = [prepare_task_sets(ds, ways, shots) for ds in raw_train_ds_list] # l2l.data.TaskDataset(dataset_train, transforms_train)
    val_taskset_list = [prepare_task_sets(ds, ways, shots) for ds in raw_val_ds_list]     # l2l.data.TaskDataset(dataset_val, transforms_val)
    test_taskset_list = [prepare_task_sets(ds, ways, shots) for ds in raw_test_ds_list]   # l2l.data.TaskDataset(dataset_test, transforms_test)

    # Create model
    reconstructive_pretrain_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=32)
    reconstructive_pretrain_model.load_state_dict(
        torch.load("DRAEM/checkpoints/DRAEM_train_0.0001_400_bs8_texture_w32c32.pckl", map_location='cuda:0'))
    model = MamlBase(reconstructive_pretrain_model.encoder, 32)

    # freeze encoder
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    model.to(device)

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            tmp_train_taskset = random.choice(train_taskset_list)
            batch = tmp_train_taskset.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            tmp_val_taskset = random.choice(val_taskset_list)
            batch = tmp_val_taskset.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.requires_grad:
                p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        tmp_test_taskset = random.choice(test_taskset_list)
        batch = tmp_test_taskset.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--obj_id', action='store', type=int, required=True)
    # parser.add_argument('--bs', action='store', type=int, required=True)
    # parser.add_argument('--lr', action='store', type=float, required=True)
    # parser.add_argument('--epochs', action='store', type=int, required=True)
    # parser.add_argument('--gpu_id', action='store', type=int, default=-1, required=False)
    # parser.add_argument('--data_path', action='store', type=str, required=True)
    # parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    # parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    # parser.add_argument('--log_path', action='store', type=str, required=True)
    # parser.add_argument('--visualize', action='store_true')

    # args = parser.parse_args()
    # print('Visualize -> ', args.visualize)
    # obj_batch = [['texture']]

    main(cuda=True)
