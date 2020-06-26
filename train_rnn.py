import os
import errno
import time
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_

from tools.eval_proposal_anet import ANETproposal
from data.anet_dataset import ANetDataset, get_vocab_and_sentences
from data.utils import iou_with_anchors
from model.bmn import BMN
from model.rnn_ptr import RNNPtr
from opt import parse_opt_rnn

args = parse_opt_rnn()
print(args)
tscale = args.temporal_scale


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(args, args.train_data_folder, text_proc, raw_data, test=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    valid_dataset = ANetDataset(args, args.val_data_folder, text_proc, raw_data, test=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    return train_loader, valid_loader


def get_bmn_model(args):
    model = BMN(args)

    # Initialize the networks and the criterion
    if len(args.bmn_model) > 0:
        print("Initializing weights from {}".format(args.bmn_model))
        model.load_state_dict(torch.load(args.bmn_model, map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        model.cuda()
    model.eval()
    return model


def main():
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')
    train_loader, valid_loader = get_dataset(args)

    print('building models')
    bmn = get_bmn_model(args)
    ptr = RNNPtr(args)
    ptr.cuda()

    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ptr.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon, weight_decay=1e-4)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, ptr.parameters()),
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
    best_loss = float('inf')
    all_training_losses = []
    all_eval_losses = []

    for train_epoch in range(args.max_epochs):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(train_epoch))

        epoch_loss = train(bmn, ptr, optimizer, train_loader, args)
        all_training_losses.append(epoch_loss)

        valid_loss = valid(bmn, ptr, valid_loader, args)
        all_eval_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(ptr.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
            print('*' * 5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        torch.save({'train_loss': all_training_losses,
                    'eval_loss': all_eval_losses,
                    }, os.path.join(args.checkpoint_path, 'model_losses.t7'))

        # learning rate decay
        scheduler.step(valid_loss)

        # validation/save checkpoint every a few epochs
        if train_epoch % args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            torch.save(ptr.state_dict(), os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        print('-' * 80)
        print('Epoch {} summary'.format(train_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time() - t_epoch_start
        ))
        print('-' * 80)


def train(bmn_model, ptr_model, optimizer, train_loader, args):
    ptr_model.train()
    train_loss = []
    nbatches = len(train_loader)
    last_time = time.time()
    data_time, model_time = 0, 0

    for train_iter, data in enumerate(train_loader):
        sentence, img_feat, label_confidence, label_start, label_end, index = data
        if args.cuda:
            sentence, img_feat, label_confidence, label_start, label_end \
                = map(lambda x: x.cuda(), [sentence, img_feat, label_confidence, label_start, label_end])

        confidence_map, start_scores, end_scores, feature_map = map(lambda x: x.detach().cpu().numpy(),
                                                                    bmn_model(img_feat))

        prop_score, prop_loc, prop_feature = generate_prop(confidence_map, start_scores, end_scores, feature_map)
        iou_labels, lens = generate_label(prop_loc, train_loader.dataset.sample_list, index)
        prop_score, prop_loc, prop_feature = torch.FloatTensor(prop_score), torch.FloatTensor(prop_loc), \
                                             torch.FloatTensor(prop_feature)
        iou_labels, lens = torch.FloatTensor(iou_labels), torch.LongTensor(lens)
        prop_score, prop_loc, prop_feature, iou_labels, lens \
            = map(lambda x: x.cuda(), [prop_score, prop_loc, prop_feature, iou_labels, lens])

        data_time += time.time() - last_time
        last_time = time.time()

        pred = ptr_model(prop_score, prop_loc, prop_feature, iou_labels, lens)
        loss = compute_loss_mse(pred, iou_labels, lens)
        loss.backward()
        train_loss.append(loss.data.item())
        clip_grad_norm_(filter(lambda p: p.requires_grad, ptr_model.parameters()), args.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        model_time += time.time() - last_time
        last_time = time.time()

        print('iter: [{}/{}], training loss: {:.4f}, data time: {:.4f}s, total time: {:.4f}s'.format(
            train_iter + 1, nbatches, loss.data.item(), data_time, data_time + model_time))
        data_time, model_time = 0, 0

        model_time += time.time() - last_time
        last_time = time.time()

    return np.mean(train_loss)


def valid(bmn_model, ptr_model, test_loader, args):
    ptr_model.eval()
    valid_loss = []
    nbatches = len(test_loader)
    last_time = time.time()
    data_time, model_time = 0, 0

    for valid_iter, data in enumerate(test_loader):
        sentence, img_feat, label_confidence, label_start, label_end, index = data
        if args.cuda:
            sentence, img_feat, label_confidence, label_start, label_end \
                = map(lambda x: x.cuda(), [sentence, img_feat, label_confidence, label_start, label_end])

        confidence_map, start_scores, end_scores, feature_map = map(lambda x: x.detach().cpu().numpy(),
                                                                    bmn_model(img_feat))

        prop_score, prop_loc, prop_feature = generate_prop(confidence_map, start_scores, end_scores, feature_map)
        iou_labels, lens = generate_label(prop_loc, test_loader.dataset.sample_list, index)
        prop_score, prop_loc, prop_feature = torch.FloatTensor(prop_score), torch.FloatTensor(prop_loc), \
                                             torch.FloatTensor(prop_feature)
        iou_labels, lens = torch.FloatTensor(iou_labels), torch.LongTensor(lens)
        prop_score, prop_loc, prop_feature, iou_labels, lens \
            = map(lambda x: x.cuda(), [prop_score, prop_loc, prop_feature, iou_labels, lens])

        data_time += time.time() - last_time
        last_time = time.time()

        pred = ptr_model(prop_score, prop_loc, prop_feature, iou_labels, lens)
        loss = compute_loss_mse(pred, iou_labels, lens)
        valid_loss.append(loss.data.item())

        model_time += time.time() - last_time
        last_time = time.time()

        print('iter: [{}/{}], valid loss: {:.4f}, data time: {:.4f}s, total time: {:.4f}s'.format(
            valid_iter + 1, nbatches, loss.data.item(), data_time, data_time + model_time))
        data_time, model_time = 0, 0

        model_time += time.time() - last_time
        last_time = time.time()

    return np.mean(valid_loss)


def compute_loss(pred, labels, lens):
    batch_size = pred.size(0)
    loss = .0
    for i in range(batch_size):
        cur_pred = pred[i][:lens[i]]
        cur_label = labels[i][:lens[i]]
        cur_loss = -(cur_label * torch.log(cur_pred + 1e-20) + (1 - cur_label) * torch.log(1 - cur_pred + 1e-20))
        loss += torch.sum(cur_loss)
    loss /= batch_size
    return loss


def compute_loss_mse(pred, labels, lens):
    batch_size = pred.size(0)
    loss = .0
    loss_fn = torch.nn.MSELoss(reduction='none')
    for i in range(batch_size):
        cur_pred = pred[i][:lens[i]]
        cur_label = labels[i][:lens[i]]
        cur_loss = loss_fn(cur_pred, cur_label)
        loss += torch.sum(cur_loss)
    loss /= batch_size
    return loss


def generate_prop(batch_confidence_map, batch_start_scores, batch_end_scores, batch_feature_map):
    batch_size = batch_confidence_map.shape[0]
    prop_score, prop_loc, prop_feature = [], [], []
    for idx in range(batch_size):
        confidence_map, start_scores, end_scores, feature_map = batch_confidence_map[idx], batch_start_scores[idx], \
                                                                batch_end_scores[idx], batch_feature_map[idx]
        reg_confidence, clr_confidence = confidence_map
        max_start = max(start_scores)
        max_end = max(end_scores)

        # generate the set of start points and end points
        start_bins = np.zeros(len(start_scores))
        start_bins[0] = 1  # [1,0,0...,0,1] 首末两帧
        for idx in range(1, tscale - 1):
            if start_scores[idx] > min(max(start_scores[idx + 1], start_scores[idx - 1]), 0.5 * max_start):
                start_bins[idx] = 1
        end_bins = np.zeros(len(end_scores))
        end_bins[-1] = 1
        for idx in range(1, tscale - 1):
            if end_scores[idx] > min(max(end_scores[idx + 1], end_scores[idx - 1]), 0.5 * max_end):
                end_bins[idx] = 1

        # generate proposals
        assert len(start_scores) == len(end_scores)
        id = np.arange(len(end_scores))
        start_set = np.where(start_bins[id] == 1)[0]
        end_set = np.where(end_bins[id] == 1)[0]
        new_props = np.zeros([len(start_set) * len(end_set), 3])
        idx = 0
        for start in start_set:
            criter = lambda x: x > start and x < start + args.max_duration
            end = end_set[np.where(np.vectorize(criter)(end_set))[0]]
            xmin = np.ones_like(end) * start
            xmax = end
            xmin_score = start_scores[start]
            xmax_score = end_scores[end]
            clr_score = clr_confidence[end - start - 1, start]
            reg_score = reg_confidence[end - start - 1, start]
            score = xmin_score * xmax_score * clr_score * reg_score
            new_props[idx:idx + len(score), :] = np.stack([score, xmin, xmax]).transpose()
            idx += len(score)
        new_props = new_props[:idx, :]
        props = video_post_process(args, new_props)
        cur_prop_score, cur_prop_loc, cur_prop_feature = [], [], []
        for prop in props:
            start, end = int(prop[1]), int(prop[2])
            xmin_score = start_scores[start]
            xmax_score = end_scores[end]
            clr_score = clr_confidence[end - start - 1, start]
            reg_score = reg_confidence[end - start - 1, start]
            cur_prop_score.append([prop[0], xmin_score, xmax_score, clr_score, reg_score])
            cur_prop_loc.append([float(start) / tscale, float(end) / tscale])
            cur_prop_feature.append(list(feature_map[:, end - start - 1, start]))
        prop_score.append(cur_prop_score)
        prop_loc.append(cur_prop_loc)
        prop_feature.append(cur_prop_feature)
    return np.array(prop_score), np.array(prop_loc), np.array(prop_feature)


def video_post_process(args, prop):
    #prop: score, xmin, xmax, start, end
    if len(prop) > 1:
        snms_alpha = args.soft_nms_alpha
        snms_t1 = args.soft_nms_low_thres
        snms_t2 = args.soft_nms_high_thres
        prop = soft_nms(prop, snms_alpha, snms_t1, snms_t2)

    idx = np.argsort(prop[:, 1], axis=0)
    prop = prop[idx]

    return prop[: args.prop_num]


def soft_nms(prop, alpha, t1, t2):
    '''
    prop: proposals generated by network, (score, start, end) for each row;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    idx = np.argsort(-prop[:,0], axis=0)
    prop = prop[idx]
    tscore, tstart, tend = map(list, [prop[:, 0], prop[:, 1], prop[:, 2]])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        tmp_iou_list = iou_with_anchors(np.array(tstart), np.array(tend), tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = (tend[max_index] - tstart[max_index]) / tscale
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newprop = np.stack([rscore, rstart, rend]).transpose()
    return newprop


def generate_label(prop_loc, sample_list, index):
    batch_size = prop_loc.shape[0]
    iou_labels = []
    lens = []
    for idx in range(batch_size):
        batch_prop_start = np.append([0.0], prop_loc[idx][:, 0])
        batch_prop_end = np.append([0.0], prop_loc[idx][:, 1])
        gt_start = sample_list[index[idx]][2]
        gt_end = sample_list[index[idx]][3]
        cur_labels = []
        lens.append(len(gt_start) + 1)
        for i in range(len(gt_start)):
            tmp_iou_list = iou_with_anchors(batch_prop_start, batch_prop_end, gt_start[i], gt_end[i])
            cur_labels.append(tmp_iou_list)
        end_prop = np.zeros([len(batch_prop_start), ])
        end_prop[0] = 1.
        cur_labels.append(end_prop)
        for i in range(len(gt_start) + 1, args.max_seq_len):
            cur_labels.append(np.zeros([len(batch_prop_start), ]))
        iou_labels.append(np.array(cur_labels))
    return np.array(iou_labels), np.array(lens)


if __name__ == '__main__':
    main()

