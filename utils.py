import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import torch
import torch.nn.functional as F
import numpy as np
from DRAEM.loss import FocalLoss
import copy
import random


def pairwise_distances_logits(a, b):
    # tmp = torch.cdist(a, b)
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
               b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    # return -tmp
    return  logits

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


def fast_adaptv3(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    loss_focal = FocalLoss()

    data, labels, mask, s_data, s_labels, s_mask = batch

    data = data.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    s_data = s_data.to(device)
    s_labels = s_labels.to(device)
    s_mask = s_mask.to(device)

    data = data.squeeze(0)
    labels = labels.squeeze(0)
    mask = mask.squeeze(0)
    s_data = s_data.squeeze(0)
    s_labels = s_labels.squeeze(0)
    s_mask = s_mask.squeeze(0)

    # Compute support and query embeddings
    support, support_f = model(s_data)
    support_f = support_f.reshape(ways, shot, -1).mean(dim=1)
    query, query_f = model(data)
    labels = labels.long()

    logits = pairwise_distances_logits(query_f, support_f)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    out_mask_sm = torch.softmax(support, dim=1)

    segment_loss = loss_focal(out_mask_sm, s_mask)

    acc = accuracy(logits, labels)

    return loss + segment_loss, acc


def fast_adapt_train(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    loss_focal = FocalLoss()

    data, labels, mask = batch

    data = data.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    mask = mask.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings_mask, embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support_mask = embeddings_mask[support_indices]
    s_mask = mask[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    # print('logits->', logits)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    out_mask_sm = torch.softmax(support_mask, dim=1)
    segment_loss = loss_focal(out_mask_sm, s_mask)
    acc = accuracy(logits, labels)
    return loss + segment_loss, acc


def adapt_and_test(model, batch, ways, shot, query_num, metric=None, device=None, nb_adapt_steps=5):
    adapted_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.0001)

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

    ################################################################################

    sort = torch.sort(s_labels)
    s_data = s_data.squeeze(0)[sort.indices].squeeze(0)
    s_labels = s_labels.squeeze(0)[sort.indices].squeeze(0)
    # print('s_labels -> ', s_labels)
    # adapt_support_indices = np.zeros(s_data.size(0), dtype=bool)
    # selection = np.arange(ways) * (shot)
    # for offset in range(shot):
    #    adapt_support_indices[selection + offset] = True

    # adapt_query_indices = torch.from_numpy(~adapt_support_indices)
    # adapt_support_indices = torch.from_numpy(adapt_support_indices)
    # print('adapt_query_indices -> ', adapt_query_indices)
    # print('adapt_support_indices -> ', adapt_support_indices)
    loss = 0
    adapted_model.train()
    for adapt_step in range(nb_adapt_steps):
        tmp_good_ind = random.sample(range(0, shot), shot)
        tmp_def_ind = random.sample(range(shot, shot * 2), shot)

        tmp_support = adapted_model(s_data[tmp_good_ind[::2] + tmp_def_ind[::2]])
        tmp_support = tmp_support.reshape(ways, shot // 2, -1).mean(dim=1)

        tmp_query = adapted_model(s_data[tmp_good_ind[1::2] + tmp_def_ind[1::2]])
        tmp_labels = s_labels[tmp_good_ind[1::2] + tmp_def_ind[1::2]].long()

        tmp_logits = pairwise_distances_logits(tmp_query, tmp_support)
        loss = F.cross_entropy(tmp_logits, tmp_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ################################################################################

    adapted_model.eval()
    # Compute support and query embeddings
    support = adapted_model(s_data)
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = adapted_model(data)
    labels = labels.long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    acc = accuracy(logits, labels)
    return loss, acc


def fast_adaptv3_t(model, batch, ways, shot, query_num, metric=None, device=None, nb_adapt_steps=10):

    adapted_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.0001)

    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    loss_focal = FocalLoss()

    data, labels, mask, s_data, s_labels, s_mask = batch

    data = data.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    s_data = s_data.to(device)
    s_labels = s_labels.to(device)
    s_mask = s_mask.to(device)

    data = data.squeeze(0)
    labels = labels.squeeze(0)
    mask = mask.squeeze(0)
    s_data = s_data.squeeze(0)
    s_labels = s_labels.squeeze(0)
    s_mask = s_mask.squeeze(0)

    ################################################################################

    sort = torch.sort(s_labels)
    s_data = s_data.squeeze(0)[sort.indices].squeeze(0)
    s_labels = s_labels.squeeze(0)[sort.indices].squeeze(0)
    s_mask = s_mask.squeeze(0)[sort.indices].squeeze(0)

    loss = 0
    segment_loss = 0

    adapted_model.train()
    for adapt_step in range(nb_adapt_steps):
        tmp_good_ind = random.sample(range(0, shot), shot)
        tmp_def_ind = random.sample(range(shot, shot * 2), shot)

        tmp_support, tmp_support_f = adapted_model(s_data[tmp_good_ind[::2] + tmp_def_ind[::2]])
        tmp_support_f = tmp_support_f.reshape(ways, shot // 2, -1).mean(dim=1)

        tmp_query, tmp_query_f = adapted_model(s_data[tmp_good_ind[1::2] + tmp_def_ind[1::2]])
        tmp_labels = s_labels[tmp_good_ind[1::2] + tmp_def_ind[1::2]].long()

        tmp_logits = pairwise_distances_logits(tmp_query_f, tmp_support_f)
        loss = F.cross_entropy(tmp_logits, tmp_labels)

        out_mask_sm = torch.softmax(tmp_support, dim=1)
        segment_loss = loss_focal(out_mask_sm, s_mask[tmp_good_ind[::2] + tmp_def_ind[::2]])

        optimizer.zero_grad()
        (loss+segment_loss).backward()
        optimizer.step()
    ################################################################################

    adapted_model.eval()
    # Compute support and query embeddings
    support, support_f = adapted_model(s_data)
    support_f = support_f.reshape(ways, shot, -1).mean(dim=1)
    query, query_f = adapted_model(data)
    labels = labels.long()

    logits = pairwise_distances_logits(query_f, support_f)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    out_mask_sm = torch.softmax(support, dim=1)

    segment_loss = loss_focal(out_mask_sm, s_mask)

    acc = accuracy(logits, labels)

    return loss + segment_loss, acc