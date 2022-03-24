#!/usr/bin/env python3
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import os
from DRAEM.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from models import ProtoNet, SegNetv3
from mvtec_dataset import BDDDataset, BDDDatasetv3
from utils import prepare_task_sets, pairwise_distances_logits, fast_adapt, fast_adaptv3, fast_adapt_train, fast_adaptv3_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--train-query', type=int, default=5)

    parser.add_argument('--test-shot', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--test-query', type=int, default=5)

    parser.add_argument("--experiment-name", type=str, default='fewshot_learner_seg_')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--base-width', type=int, default=32)
    parser.add_argument("--embeddings-path", type=str,
                        default='DRAEM/checkpoints/DRAEM_train_0.0001_400_bs16_texture_w32c32_v2.pckl')
    parser.add_argument("--checkpoint-path", type=str,
                        default='./checkpoints/')

    parser.add_argument('--checkpoint-save-freq', type=int, default=25)

    parser.add_argument('--gpu', default=1)

    args = parser.parse_args()
    print(args)
    run_name = f'{args.experiment_name}_train_seed_{args.random_seed}_{args.base_width}_lr_{args.lr}_e_'

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

    discriminative_pretrained_model = DiscriminativeSubNetwork(in_channels=6, out_channels=2,
                                                               base_channels=args.base_width)
    discriminative_pretrained_model.load_state_dict(
        torch.load("DRAEM/checkpoints/DRAEM_train_0.0001_400_bs16_texture_w32c32_v2_seg.pckl", map_location='cpu'))

    model = SegNetv3(embeddings, discriminative_pretrained_model)
    # Freeze wrights
    for name, param in model.reconstruct.named_parameters():
        param.requires_grad = False

    model.to(device)
    train_classes_names = ['wood', 'pill', 'carpet', 'grid', 'hazelnut', 'zipper']
    # train_classes_names = ['dagm_c1', 'dagm_c2', 'dagm_c4']
    # train_classes_names = ['wood', 'capsule', 'dagm_c2', 'carpet', 'grid',
    #                                 'hazelnut', 'zipper', 'dagm_c1', 'dagm_c3', 'dagm_c5', 'kolectorsdd2_train']

    raw_train_ds_list = [BDDDataset(root_path='data',
                                    bdd_folder_path='mvtech_cleaned',
                                    class_names_list=[class_n],
                                    is_train=False,
                                    resize=(256, 256),
                                    ways=args.test_way,
                                    shots=args.test_shot,
                                    query=args.test_query) for class_n in train_classes_names]

    val_classes_names = ['bottle', 'cable']
    # val_classes_names = ['dagm_c3']

    raw_val_ds_list = [BDDDatasetv3(root_path='data',
                                    bdd_folder_path='mvtech_cleaned',
                                    class_names_list=[class_n],
                                    is_train=False,
                                    resize=(256, 256),
                                    ways=args.test_way,
                                    shots=args.test_shot,
                                    query=args.test_query) for class_n in val_classes_names]

    test_classes_names = ['capsule', 'tile', 'leather']
    # test_classes_names = ['dagm_c5', 'dagm_c6']
    # test_classes_names = ['pill', 'tile', 'leather', 'dagm_c4', 'dagm_c6', 'kolectorsdd2_test']
    raw_test_ds_list = [BDDDatasetv3(root_path='data',
                                     bdd_folder_path='mvtech_cleaned',
                                     class_names_list=[class_n],
                                     is_train=False,
                                     resize=(256, 256),
                                     ways=args.test_way,
                                     shots=args.test_shot,
                                     query=args.test_query) for class_n in test_classes_names]

    # train_taskset_list = [prepare_task_sets(ds, args.train_query, args.train_way, args.shot) for ds in
    #                      raw_train_ds_list]  # l2l.data.TaskDataset(dataset_train, transforms_train)
    # val_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_val_ds_list]     # l2l.data.TaskDataset(dataset_val, transforms_val)
    # test_taskset_list = [prepare_task_sets(ds, args.test_query, args.test_way, args.test_shot) for ds in raw_test_ds_list]   # l2l.data.TaskDataset(dataset_test, transforms_test)

    train_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_train_ds_list]
    val_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_val_ds_list]
    test_loader_list = [DataLoader(task, pin_memory=True, shuffle=True) for task in raw_test_ds_list]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, args.max_epoch + 1):

        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        batch_loss = 0
        for i in range(4):
            batch = next(iter(random.choice(train_loader_list)))
            loss, acc = fast_adapt_train(model,
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
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + str(epoch) + ".pckl"))

        if (epoch) % 1 == 0:
            model.eval()

            # loss_ctr = 0
            # n_loss = 0
            # n_acc = 0
            for v_i, val_loader in enumerate(val_loader_list):
                loss_ctr = 0
                n_loss = 0
                n_acc = 0
                for i, batch in enumerate(val_loader):
                    loss, acc = fast_adaptv3_t(model,
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
            loss, acc = fast_adaptv3_t(model,
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