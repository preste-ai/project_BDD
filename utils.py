import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


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
    embeddings = model(data)  # forward is called here
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)  # усереднений ембендінг по класах (2, 256)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    # print('logits->', logits)
    loss = F.cross_entropy(logits, labels)
    # loss = F.binary_cross_entropy(torch.tensor(torch.argmax(logits, dim=1), dtype=torch.float32, requires_grad=True), labels.float())

    acc = accuracy(logits, labels)
    torch.cuda.empty_cache()
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

# ToDo visualize batch
def display_batch(batch):
    data, labels = batch

    for idx, label in zip(range(list(labels.shape)[0]), labels):
        print(idx, label)

def save_history(history, epoch, n_loss, n_acc, loss_ctr, mode):
    if epoch == 1:
        history['Epoch'] = [epoch]
        history['Loss'] = [n_loss / loss_ctr]
        history['Accuracy'] = [(n_acc / loss_ctr).item()]
    elif epoch == 10 and mode == 'val':
        history['Epoch'] = [epoch]
        history['Loss'] = [n_loss / loss_ctr]
        history['Accuracy'] = [(n_acc / loss_ctr).item()]
    else:
        history['Epoch'].append(epoch)
        history['Loss'].append(n_loss / loss_ctr)
        history['Accuracy'].append((n_acc / loss_ctr).item())

def write_training_history(epoch, n_loss, loss_ctr, n_acc, history, val_class_name, mode = 'train'):
    """
        Set mode either to 'train' or 'val'
    """

    if not os.path.exists('checkpoints/learning_curves'): os.makedirs('checkpoints/learning_curves', exist_ok=True)

    if mode == 'val':
        if f'{val_class_name}'not in history:
            history[f'{val_class_name}'] = {}
        save_history(history[f'{val_class_name}'], epoch, n_loss, n_acc, loss_ctr, mode=mode)
    elif mode == 'train':
        save_history(history, epoch, n_loss, n_acc, loss_ctr, mode=mode)



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_training_history(training_history, shot, way, mode):

    """
    Set mode either to 'train' or 'val'
    """
    if mode == 'train':
        plt.figure()
        plt.title(f'{shot}-shot {way}-way learning curve')
        plt.plot(training_history['Epoch'], training_history['Loss'], color='blue', label='loss')
        plt.plot(training_history['Epoch'], training_history['Accuracy'], color='green', label='accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f"checkpoints/learning_curves/train_history_{shot}shot{way}way_{len(training_history['Epoch'])}ep.jpg")
        print('learning curve saved')
    elif mode == 'val' and len(training_history) > 0:
        plt.figure()
        plt.title(f'{shot}-shot {way}-way learning curve')
        anyclass = ''
        cmap = get_cmap(len(training_history))
        for idx, clas, history in enumerate(training_history.items()):
            anyclass = clas
            plt.plot(training_history[clas]['Epoch'], training_history[clas]['Loss'], color=cmap(idx), label=f'loss_{clas}')
            plt.plot(training_history[clas]['Epoch'], training_history[clas]['Accuracy'], color=(idx), label=f'accuracy_{clas}', linestyle='--')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(
            f"checkpoints/learning_curves/validation_history_{shot}shot{way}way_{len(training_history[anyclass]['Epoch'])}ep.jpg")
        print('validation curve saved')

