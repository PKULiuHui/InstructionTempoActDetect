import os
import errno
import time
import json

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    test_dataset = ANetDataset(args, args.test_data_folder, text_proc, raw_data, test=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    return test_loader


def get_model(args):
    bmn = BMN(args)
    # Initialize the networks and the criterion
    if len(args.bmn_model) > 0:
        print("Initializing weights from {}".format(args.bmn_model))
        bmn.load_state_dict(torch.load(args.bmn_model, map_location=lambda storage, location: storage))
    ptr = RNNPtr(args)
    if len(args.ptr_model) > 0:
        print("Initializing weights from {}".format(args.ptr_model))
        ptr.load_state_dict(torch.load(args.ptr_model, map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        bmn.cuda()
        ptr.cuda()
    bmn.eval()
    ptr.eval()
    return bmn, ptr


def main():

    print('loading dataset')
    test_loader = get_dataset(args)

    print('building models')
    bmn_model, ptr_model = get_model(args)

    p, r, f = .0, .0, .0
    cnt = .0
    for test_iter, data in enumerate(tqdm(test_loader)):
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

        # pred = ptr_model(prop_score, prop_loc, prop_feature, iou_labels, lens, test=False)

        batch_size = iou_labels.size(0)
        step_num = iou_labels.size(1)
        for i in range(batch_size):
            props = []
            for j in range(step_num):
                idx = torch.argmax(iou_labels[i][j])
                if idx == 0:
                    break
                props.append(prop_loc[i][idx - 1].tolist())
                # pred[i, :, idx] = -1
            avg_P, avg_R, avg_F = evaluate(props, test_loader.dataset.sample_list, index[i])
            p += avg_P
            r += avg_R
            f += avg_F
            cnt += 1
    p, r, f = p / cnt, r / cnt, f / cnt
    print('p: %.4f' % (np.mean(p)))
    print('r: %.4f' % (np.mean(r)))
    print('f: %.4f' % (np.mean(f)))


def evaluate(props, sample_list, idx):
    prop_start, prop_end = [prop[0] for prop in props], [prop[1] for prop in props]
    prop_start, prop_end = np.array(prop_start), np.array(prop_end)
    gt_start = sample_list[idx][2]
    gt_end = sample_list[idx][3]

    prop_gt_iou = []
    for i in range(len(props)):
        tmp_iou_list = iou_with_anchors(gt_start, gt_end, prop_start[i], prop_end[i])
        prop_gt_iou.append(tmp_iou_list)
    prop_gt_iou = np.array(prop_gt_iou)
    p, r, f = [], [], []
    for tiou in np.linspace(0.5, 0.95, 10):
        cur_iou = copy.deepcopy(prop_gt_iou)
        match = .0
        matched = []
        for i in range(len(props)):
            j = np.argmax(cur_iou[i])
            if prop_gt_iou[i][j] > tiou and j not in matched:
                match += 1
                matched.append(j)
                cur_iou[:, j] = -1
        cur_p = match / len(props)
        cur_r = match / len(gt_start)
        p.append(cur_p)
        r.append(cur_r)
        f.append(2 * cur_p * cur_r / (cur_p + cur_r + 1e-10))

    return np.array(p), np.array(r), np.array(f)


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

