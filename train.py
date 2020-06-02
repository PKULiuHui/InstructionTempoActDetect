"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import errno
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
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
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(args, args.train_data_folder, text_proc, raw_data, test=False)

    # dist parallel, optional
    args.distributed = args.world_size > 1
    if args.distributed and args.cuda:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers)

    valid_dataset = ANetDataset(args, args.val_data_folder, text_proc, raw_data, test=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)

    return train_loader, valid_loader, text_proc, train_sampler


def get_model(text_proc, args):
    sent_vocab = text_proc.vocab
    model = BMN(args)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                         map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        if args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')
    train_loader, valid_loader, text_proc, train_sampler = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    # filter params
    # that don't require gradient (credit: PyTorch Forum issue 679)
    # smaller learning rate for the decoder
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon, weight_decay=1e-4)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=1e-5,
            momentum=args.alpha,
            nesterov=True
        )
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
                                               patience=args.patience_epoch,
                                               verbose=True)

    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_loss = float('inf')

    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom()
        vis_window = {'iter': None,
                      'loss': None}
    else:
        vis, vis_window = None, None

    all_eval_losses = []
    all_tem_losses = []
    all_pem_reg_losses = []
    all_pem_cls_losses = []
    all_training_losses = []
    bm_mask = get_mask(args.temporal_scale)

    for train_epoch in range(args.max_epochs):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(train_epoch))

        if args.distributed:
            train_sampler.set_epoch(train_epoch)

        epoch_loss = train(train_epoch, model, optimizer, train_loader, bm_mask, vis, vis_window, args)
        all_training_losses.append(epoch_loss)

        valid_loss, val_tem_loss, val_pem_reg_loss, val_pem_cls_loss = valid(model, valid_loader, bm_mask)

        all_eval_losses.append(valid_loss)
        all_tem_losses.append(val_tem_loss)
        all_pem_reg_losses.append(val_pem_reg_loss)
        all_pem_cls_losses.append(val_pem_cls_loss)

        '''if args.enable_visdom:
            if vis_window['loss'] is None:
                if not args.distributed or (args.distributed and dist.get_rank() == 0):
                    vis_window['loss'] = vis.line(
                        X=np.tile(np.arange(len(all_eval_losses)),
                                  (6, 1)).T,
                        Y=np.column_stack((np.asarray(all_training_losses),
                                           np.asarray(all_eval_losses),
                                           np.asarray(all_cls_losses),
                                           np.asarray(all_reg_losses),
                                           np.asarray(all_sent_losses),
                        opts=dict(title='Loss',
                                  xlabel='Validation Iter',
                                  ylabel='Loss',
                                  legend=['train',
                                          'dev',
                                          'dev_cls',
                                          'dev_reg',
                                          'dev_mask']))
            else:
                if not args.distributed or (
                        args.distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.tile(np.arange(len(all_eval_losses)),
                                  (6, 1)).T,
                        Y=np.column_stack((np.asarray(all_training_losses),
                                           np.asarray(all_eval_losses),
                                           np.asarray(all_cls_losses),
                                           np.asarray(all_reg_losses),
                                           np.asarray(all_sent_losses),
                                           np.asarray(all_mask_losses))),
                        win=vis_window['loss'],
                        opts=dict(title='Loss',
                                  xlabel='Validation Iter',
                                  ylabel='Loss',
                                  legend=['train',
                                          'dev',
                                          'dev_cls',
                                          'dev_reg',
                                          'dev_sentence',
                                          'dev_mask']))'''

        if valid_loss < best_loss:
            best_loss = valid_loss
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
            print('*' * 5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        # save eval and train losses
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
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
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(),
                           os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        print('-' * 80)
        print('Epoch {} summary'.format(train_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time() - t_epoch_start
        ))
        print('val_tem: {:.4f}, val_pem_reg: {:.4f}, val_pem_cls: {:.4f}'.format(
            val_tem_loss, val_pem_cls_loss, val_pem_cls_loss
        ))
        print('-' * 80)


### Training the network ###
def train(epoch, model, optimizer, train_loader, bm_mask, vis, vis_window, args):
    model.train()  # training mode
    train_loss = []
    nbatches = len(train_loader)
    t_iter_start = time.time()

    for train_iter, data in enumerate(train_loader):
        sentence, img_feat, label_confidence, label_start, label_end = data

        if args.cuda:
            sentence, img_feat, label_confidence, label_start, label_end \
                = map(lambda x: x.cuda(), [sentence, img_feat, label_confidence, label_start, label_end])

        t_model_start = time.time()
        confidence_map, start, end = model(img_feat)

        total_loss, tem_loss, pem_reg_loss, pem_cls_loss = \
            bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        optimizer.zero_grad()
        total_loss.backward()
        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_norm)
        optimizer.step()

        train_loss.append(total_loss.data.item())

        t_model_end = time.time()
        print('iter: [{}/{}], training loss: {:.4f}, tem: {:.4f}, pem_reg: {:.4f}, pem_cls: {:.4f}, '
              'grad norm: {:.4f} data time: {:.4f}s, total time: {:.4f}s'.format(
            train_iter, nbatches, total_loss.data.item(), tem_loss.data.item(),
            pem_reg_loss.data.item(), pem_cls_loss.data.item(), total_grad_norm,
            t_model_start - t_iter_start, t_model_end - t_iter_start))

        t_iter_start = time.time()

    return np.mean(train_loss)


### Validation ##
def valid(model, loader, bm_mask):
    model.eval()
    valid_loss = []
    for iter, data in enumerate(loader):
        sentence, img_feat, label_confidence, label_start, label_end = data
        with torch.no_grad():
            if args.cuda:
                sentence, img_feat, label_confidence, label_start, label_end \
                    = map(lambda x: x.cuda(), [sentence, img_feat, label_confidence, label_start, label_end])

            confidence_map, start, end = model(img_feat)

            total_loss, tem_loss, pem_reg_loss, pem_cls_loss = \
                bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

            valid_loss.append(
                [total_loss.data.item(), tem_loss.data.item(), pem_reg_loss.data.item(), pem_cls_loss.data.item()])

    valid_loss = np.array(valid_loss)
    print(np.mean(valid_loss, axis=0))

    return np.mean(valid_loss, axis=0)


if __name__ == "__main__":
    main(args)
