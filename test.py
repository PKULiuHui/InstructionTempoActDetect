"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import time
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.eval_proposal_anet import ANETproposal
from data.anet_dataset import ANetDataset, get_vocab_and_sentences
from data.utils import iou_with_anchors
from model.bmn import BMN
from opt import parse_opt

args = parse_opt(train=False)
args.batch_size = 1
print(args)


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    test_dataset = ANetDataset(args, args.test_data_folder, text_proc, raw_data, test=True)

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=args.num_workers)

    return test_loader, text_proc


def get_model(text_proc, args):
    model = BMN(args)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from, map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        model.cuda()

    return model


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
        tmp_iou_list = iou_with_anchors(
            np.array(tstart),
            np.array(tend), tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = tend[max_index] - tstart[max_index]
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


def video_post_process(args, prop, video_info):
    #prop: score, start, end
    if len(prop) > 1:
        snms_alpha = args.soft_nms_alpha
        snms_t1 = args.soft_nms_low_thres
        snms_t2 = args.soft_nms_high_thres
        prop = soft_nms(prop, snms_alpha, snms_t1, snms_t2)

    idx = np.argsort(-prop[:, 0], axis=0)
    prop = prop[idx]
    video_duration = float(video_info["duration_frame"] // 16 * 16) / video_info["duration_frame"] * video_info[
        "duration_second"]
    proposal_list = []

    for j in range(min(100, len(prop))):
        tmp_proposal = {"score": prop[j,0],
                        "segment": [max(0, prop[j,1]) * video_duration, min(1, prop[j,2]) * video_duration]}
        proposal_list.append(tmp_proposal)

    return proposal_list


### Validation ##
def inference(model, loader, args):
    video_dict = dict()
    with open(args.dur_file) as f:
        for line in f:
            name, dur, frame = [l.strip() for l in line.split(',')]
            video_dict[name] = {'duration_second': float(dur), 'duration_frame': float(frame)}

    model.eval()
    result = dict()
    tscale = args.temporal_scale

    t_gen, t_nms = 0, 0
    t0 = time.time()

    with torch.no_grad():
        for epoch, data in enumerate(loader):
            vid, sentence, img_feat = data
            vid = vid[0]
            if args.cuda:
                img_feat = img_feat.cuda()

            confidence_map, start_scores, end_scores, _ = map(lambda x: x[0].detach().cpu().numpy(), model(img_feat))
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
                xmin = np.ones_like(end) * start / tscale
                xmax = end / tscale
                xmin_score = start_scores[start]
                xmax_score = end_scores[end]
                clr_score = clr_confidence[end - start - 1, start]
                reg_score = reg_confidence[end - start - 1, start]
                score = xmin_score * xmax_score * clr_score * reg_score
                new_props[idx:idx + len(score), :] = np.stack([score, xmin, xmax]).transpose()
                idx += len(score)
            new_props = new_props[:idx, :]

            t_gen += time.time() - t0
            t0 = time.time()
            result[vid] = video_post_process(args, new_props, video_dict[vid])
            #result[vid] = pool.apply_async(video_post_process, (args, new_props, video_dict[vid]))
            t_nms += time.time() - t0
            t0 = time.time()

            if epoch % 100 == 0:
                print('Epoch {}: proposal generating time = {}, soft nms time = {}'.format(epoch, t_gen, t_nms))

    return dict(result)


def eval_results(result, args):
    # write proposals to json file for evaluation (proposal)
    prop_all = {'version': 'VERSION 1.0', 'results': result,
                'external_data': {'used': 'false'}}

    resfile = os.path.join('./results/', 'prop_' + args.test_data_folder[0] + '_' + args.id + '.json')
    with open(resfile, 'w') as f:
        json.dump(prop_all, f)

    anet_proposal = ANETproposal(args.dataset_file, resfile,
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset=args.test_data_folder[0], verbose=True, check_status=False)

    anet_proposal.evaluate()
    f_value = anet_proposal.avg_f_value
    print('max f_value: %.4f proposal num: %d' % (f_value.max(), f_value.argmax()))

    return anet_proposal.area


def main():
    global res_before_nms

    print('loading dataset')
    test_loader, text_proc = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    result = inference(model, test_loader, args)

    recall_area = eval_results(result, args)

    print('proposal recall area: {:.6f}'.format(recall_area))


if __name__ == "__main__":
    main()