"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import errno
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.anet_dataset import ANetDataset, get_vocab_and_sentences
from model.bmn import BMN
from model.loss_func import bmn_loss_func, get_mask
from opt import parse_opt

args = parse_opt(train=True)
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(args, args.train_data_folder, text_proc, raw_data, test=False)
    train_sampler = None

    # batch size forced to be 1 here, and batch is implemented in training with periodical zero_grad
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    if args.calc_pos_neg:
        from model.loss_func import add_pem_cls_num, add_pem_reg_num, add_tem_num
        bm_mask = get_mask(args.temporal_scale, args.max_duration)
        num_pos_neg = [0 for i in range(7)]

        for i in range(len(train_dataset)):
            sentence, img_feat, label_confidence, label_start, label_end = train_dataset[i]

            num_pos_neg[0:3] = add_pem_reg_num(label_confidence, bm_mask, num_pos_neg[0:3])
            num_pos_neg[3:5] = add_pem_cls_num(label_confidence, bm_mask, num_pos_neg[3:5])
            num_pos_neg[5:7] = add_tem_num(label_start, label_end, num_pos_neg[5:7])
        np.savetxt('results/num_pos_neg.txt', np.array(num_pos_neg))
    else:
        num_pos_neg = list(np.loadtxt('results/num_pos_neg.txt'))

    valid_dataset = ANetDataset(args, args.val_data_folder, text_proc, raw_data, test=False)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return train_loader, valid_loader, text_proc, train_sampler, num_pos_neg


def get_model(text_proc, args):
    model = BMN(args)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from, map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    return model


### Training the network ###
def train(epoch, model, optimizer, train_loader, bm_mask, num_pos_neg, args):
    model.train()  # training mode
    bm_mask = bm_mask.cuda()
    optimizer.zero_grad()
    train_loss = []
    nbatches = len(train_loader)
    last_time = time.time()
    data_time, model_time = 0, 0

    for train_iter, data in enumerate(train_loader):
        sentence, img_feat, label_confidence, label_start, label_end = data
        data_time += time.time() - last_time
        last_time = time.time()

        if args.cuda:
            sentence, img_feat, label_confidence, label_start, label_end \
                = sentence.cuda(), img_feat.cuda(), label_confidence.cuda(), label_start.cuda(), label_end.cuda()

        confidence_map, start, end = model(img_feat)

        total_loss, tem_loss, pem_reg_loss, pem_cls_loss = \
            bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask, num_pos_neg)

        total_loss.backward()

        if train_iter % args.batch_size == 0:
            train_loss.append(total_loss.data.item())

            total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            print('iter: [{}/{}], training loss: {:.4f}, tem: {:.4f}, pem_reg: {:.4f}, pem_cls: {:.4f}, '
                  'grad norm: {:.4f} data time: {:.4f}s, total time: {:.4f}s'.format(
                train_iter + 1, nbatches, total_loss.data.item(), tem_loss.data.item(),
                pem_reg_loss.data.item(), pem_cls_loss.data.item(), total_grad_norm,
                data_time, data_time + model_time))
            data_time, model_time = 0, 0

        model_time += time.time() - last_time
        last_time = time.time()

    clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_norm)
    optimizer.step()  # update parameters using last data
    optimizer.zero_grad()

    return np.mean(train_loss)


### Validation ##
def valid(model, loader, bm_mask, num_pos_neg):
    model.eval()
    bm_mask = bm_mask.cuda()
    valid_loss = []

    for iter, data in enumerate(loader):
        sentence, img_feat, label_confidence, label_start, label_end = data
        with torch.no_grad():
            if args.cuda:
                sentence, img_feat, label_confidence, label_start, label_end \
                    = map(lambda x: x.cuda(), [sentence, img_feat, label_confidence, label_start, label_end])

            confidence_map, start, end = model(img_feat)

            total_loss, tem_loss, pem_reg_loss, pem_cls_loss = \
                bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask,
                              num_pos_neg)

            valid_loss.append(
                [total_loss.data.item(), tem_loss.data.item(), pem_reg_loss.data.item(), pem_cls_loss.data.item()])

    valid_loss = np.array(valid_loss)
    print(np.mean(valid_loss, axis=0))

    return np.mean(valid_loss, axis=0)


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')
    train_loader, valid_loader, text_proc, train_sampler, num_pos_neg = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    # filter params that don't require gradient
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon, weight_decay=1e-4)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              args.learning_rate, weight_decay=1e-5, momentum=args.alpha, nesterov=True)
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
                                               patience=args.patience_epoch,
                                               verbose=True)

    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_loss = float('inf')

    all_eval_losses = []
    all_tem_losses = []
    all_pem_reg_losses = []
    all_pem_cls_losses = []
    all_training_losses = []
    bm_mask = get_mask(args.temporal_scale, args.max_duration)

    for train_epoch in range(args.max_epochs):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(train_epoch + 1))

        epoch_loss = train(train_epoch, model, optimizer, train_loader, bm_mask, num_pos_neg, args)
        all_training_losses.append(epoch_loss)

        valid_loss, val_tem_loss, val_pem_reg_loss, val_pem_cls_loss = valid(model, valid_loader, bm_mask, num_pos_neg)

        all_eval_losses.append(valid_loss)
        all_tem_losses.append(val_tem_loss)
        all_pem_reg_losses.append(val_pem_reg_loss)
        all_pem_cls_losses.append(val_pem_cls_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.module.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
            print('*' * 5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        # save eval and train losses
        torch.save({'train_loss': all_training_losses,
                    'eval_loss': all_eval_losses,
                    'eval_tem_loss': all_tem_losses,
                    'eval_pem_reg_loss': all_pem_reg_losses,
                    'eval_pem_cls_loss': all_pem_cls_losses,
                    }, os.path.join(args.checkpoint_path, 'model_losses.t7'))

        # learning rate decay
        scheduler.step(valid_loss)

        # validation/save checkpoint every a few epochs
        if train_epoch % args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            torch.save(model.module.state_dict(),
                       os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        print('-' * 80)
        print('Epoch {} summary'.format(train_epoch + 1))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time() - t_epoch_start))
        print('val_tem: {:.4f}, val_pem_reg: {:.4f}, val_pem_cls: {:.4f}'.format(
            val_tem_loss, val_pem_cls_loss, val_pem_cls_loss))
        print('-' * 80)


if __name__ == "__main__":
    main(args)
