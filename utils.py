import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import torch
import torch.nn.functional as F
import numpy as np


def pairwise_distances_logits(a, b):
    tmp = torch.cdist(a, b)
    # n = a.shape[0]
    # m = b.shape[0]
    # logits = -((a.unsqueeze(1).expand(n, m, -1) -
    #            b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return -tmp


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


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


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch

    data = data.to(device)
    labels = labels.to(device)

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
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    # print('logits->', logits)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    acc = accuracy(logits, labels)
    return loss, acc


def fast_adaptv2(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels, s_data, s_labels = batch

    data = data.to(device)
    labels = labels.to(device)
    s_data = s_data.to(device)
    s_labels = s_labels.to(device)

    data = data.squeeze(0)
    labels = labels.squeeze(0)
    s_data = s_data.squeeze(0)
    s_labels = s_labels.squeeze(0)

    # Compute support and query embeddings
    support = model(s_data)
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = model(data)
    labels = labels.long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    acc = accuracy(logits, labels)
    return loss, acc
