#!/usr/bin/env python3
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

from .DRAEM.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from .models import ProtoNet
from .mvtec_dataset import MVTecDataset, BDDDatasetv2
from .utils import prepare_task_sets, pairwise_distances_logits, fast_adapt, adapt_and_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--train-query', type=int, default=5)

    parser.add_argument('--test-shot', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--test-query', type=int, default=5)

    parser.add_argument("--experiment-name", type=str, default='v1.0_fewshot_learner_default_256x256_fp32')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-epoch', type=int, default=150)
    parser.add_argument('--mini-batch', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--base-width', type=int, default=32)
    parser.add_argument("--embeddings-path", type=str,
                        default='DRAEM/checkpoints/DRAEM_train_0.0001_400_bs16_texture_w32c32_v2.pckl')
    parser.add_argument("--checkpoint-path", type=str,
                        default='./checkpoints/')
    parser.add_argument("--dataset-split", type=str, default='/app/bonseyes_preste-aiassetdemo4/train/configs/dataset_split.json')

    parser.add_argument('--checkpoint-save-freq', type=int, default=25)

    parser.add_argument('--gpu', default=1)

    args = parser.parse_args()
    print(args)
    # run_name = f'{args.experiment_name}_train_seed_{args.random_seed}_{args.base_width}_lr_{args.lr}_e_'
    run_name = f'{args.experiment_name}'

    random.seed(args.random_seed)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(args.random_seed)
        device = torch.device('cuda')

    # Create model
    embeddings = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=args.base_width)
    embeddings.load_state_dict(
        torch.load(args.embeddings_path, map_location='cpu'))

    model = ProtoNet(embeddings.encoder, base_width=args.base_width)
    # Freeze wrights
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    model.to(device)

    # Opening JSON file
    with open(args.dataset_split) as json_file:
        dataset_split = json.load(json_file)

    train_classes_names = list(dataset_split['train'].keys())
    raw_train_ds_list = [MVTecDataset(class_names_dict={class_n: dataset_split['train'][class_n]}, is_train=False, resize=(256, 256))
                         for class_n in train_classes_names]
    #print(raw_train_ds_list)

    val_classes_names = list(dataset_split['val'].keys())
    raw_val_ds_list = [BDDDatasetv2(
                                    class_names_dict={class_n: dataset_split['val'][class_n]},
                                    is_train=False,
                                    resize=(256, 256),
                                    ways=args.test_way,
                                    shots=args.test_shot,
                                    query=args.test_query
                                    ) for class_n in val_classes_names]

    test_classes_names = list(dataset_split['test'].keys())
    raw_test_ds_list = [BDDDatasetv2(
                                    class_names_dict={class_n: dataset_split['test'][class_n]},
                                    is_train=False,
                                    resize=(256, 256),
                                    ways=args.test_way,
                                    shots=args.test_shot,
                                    query=args.test_query
                                    ) for class_n in test_classes_names]


    train_taskset_list = [prepare_task_sets(ds, args.train_query, args.train_way, args.shot) for ds in
                          raw_train_ds_list]  # l2l.data.TaskDataset(dataset_train, transforms_train)
    # val_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_val_ds_list]     # l2l.data.TaskDataset(dataset_val, transforms_val)
    # test_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_test_ds_list]   # l2l.data.TaskDataset(dataset_test, transforms_test)

    train_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in train_taskset_list]
    val_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_val_ds_list]
    test_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_test_ds_list]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in tqdm(range(1, args.max_epoch + 1), desc='epoch: '):

        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        batch_loss = 0
        for i in range(args.mini_batch):
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
            batch_loss += loss
            n_acc += acc
        batch_loss = batch_loss / loss_ctr
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss / loss_ctr, n_acc / loss_ctr))

        if (epoch) % args.checkpoint_save_freq == 0:
            # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + str(epoch) + ".pckl"))
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl"))
        if (epoch) % 10 == 0:
            model.eval()

            # loss_ctr = 0
            # n_loss = 0
            # n_acc = 0
            for v_i, val_loader in enumerate(val_loader_list):
                loss_ctr = 0
                n_loss = 0
                n_acc = 0
                for i, batch in enumerate(val_loader):
                    loss, acc = adapt_and_test(model,
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
            loss, acc = adapt_and_test(model,
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

