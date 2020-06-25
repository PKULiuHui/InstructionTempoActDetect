import json
import os

import numpy as np
import scipy.interpolate
import torch
import torchtext
from torch.utils.data import Dataset

from data.utils import ioa_with_anchors, iou_with_anchors


def get_vocab_and_sentences(dataset_file, max_length=20):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                     eos_token='<eos>', tokenize='spacy',
                                     lower=True, batch_first=True,
                                     fix_length=max_length)
    train_val_sentences = []

    with open(dataset_file, 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']

    nsentence = {}
    nsentence['training'] = 0
    nsentence['validation'] = 0
    ntrain_videos = 0
    for vid, val in data.items():
        anns = val['annotations']
        split = val['subset']
        if split == 'training':
            ntrain_videos += 1
        if split in ['training', 'validation']:
            for ind, ann in enumerate(anns):
                ann['sentence'] = ann['sentence'].strip()
                train_val_sentences.append(ann['sentence'])
                nsentence[split] += 1

    sentences_proc = list(map(text_proc.preprocess, train_val_sentences))  # build vocab on train and val
    text_proc.build_vocab(sentences_proc, min_freq=5)
    print('# of words in the vocab: {}'.format(len(text_proc.vocab)))
    print(
        '# of sentences in training: {}, # of sentences in validation: {}'.format(
            nsentence['training'], nsentence['validation']
        ))
    print('# of training videos: {}'.format(ntrain_videos))
    return text_proc, data


# dataloader for training
class ANetDataset(Dataset):
    def __init__(self, args, split, text_proc, raw_data, test=False):
        super(ANetDataset, self).__init__()
        self.is_test = test

        self.tscale = args.temporal_scale  # 100?
        self.maxdur = args.max_duration
        self.temporal_gap = 1. / self.tscale
        match_map = np.zeros([self.maxdur, self.tscale, 2])
        for dur in range(1, self.maxdur + 1):
            for beg in range(self.tscale):
                xmin = self.temporal_gap * beg
                xmax = xmin + self.temporal_gap * dur
                match_map[dur - 1, beg, :] = np.array([xmin, xmax])
        self.match_map = np.reshape(match_map, [-1, 2])  # # [0,1] [1,2] [2,3].....[99,199]   # duration x start
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.tscale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.tscale + 1)]

        image_path = args.feature_root
        dur_file = args.dur_file
        dur_corr = dict()
        sampling_sec = 0.5
        with open(dur_file) as f:
            for line in f:
                name, dur, frame = [l.strip() for l in line.split(',')]
                dur, frame = float(dur), float(frame)
                interval = dur * np.ceil(frame / dur * sampling_sec) / frame  # sampling interval, \approx 0.5s
                dur_corr[name] = ( int(dur / interval) * interval, int((dur + 0.305) / interval))  # corrected duration
                # TODO: Not exactly correspond to the data

        split_paths = []
        for split_dev in split:
            split_paths.append(os.path.join(image_path, split_dev))

        # preprocess sentences
        train_sentences = []
        for vid, val in raw_data.items():
            annotations = val['annotations']
            for split_path in split_paths:
                if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
                    for ind, ann in enumerate(annotations):
                        ann['sentence'] = ann['sentence'].strip()
                        train_sentences.append(ann['sentence'])

        train_sentences = list(map(text_proc.preprocess, train_sentences))
        sentence_idx = text_proc.numericalize(text_proc.pad(train_sentences),
                                              device='cpu')  # put in memory
        if sentence_idx.size(0) != len(train_sentences):
            raise Exception("Error in numericalize sentences")
        print('size of the sentence block variable ({}): {}'.format(split, sentence_idx.size()))

        nvideo = 0
        for vid, val in raw_data.items():
            for split_path in split_paths:
                if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
                    nvideo += 1

        self.sample_list = [None] * nvideo  # list of list for data samples
        idx_out, idx_in, idx_last = 0, 0, 0
        for vid, val in raw_data.items():
            annotations = val['annotations']
            for split_path in split_paths:
                if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
                    start, end = np.zeros(len(annotations)), np.zeros(len(annotations))
                    for ind, ann in enumerate(annotations):
                        start[ind], end[ind] = ann['segment']
                        idx_in += 1
                    start, end = map(lambda x: np.clip( (x - 0.5) / dur_corr[vid][0], 0, 1), [start, end])

                    assert idx_in - idx_last == len(annotations)
                    self.sample_list[idx_out] = (
                        os.path.join(split_path, vid), dur_corr[vid][1], start, end,
                        sentence_idx[idx_last:idx_in, :])
                    idx_out += 1
                    idx_last = idx_in

    def __getitem__(self, index):
        video_prefix, nframe, start, end, sentence = self.sample_list[index]
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        #if bn_feat.shape[0] != nframe:
        #    print(bn_feat.shape, nframe) #not exactly the same

        assert resnet_feat.size(0) == bn_feat.size(0)

        img_feat = torch.cat((resnet_feat, bn_feat), dim=1).float()
        img_feat = self.poolData(img_feat, num_prop=self.tscale).transpose(1, 0)
        x, y = sentence.shape
        sentence = torch.cat([sentence, torch.zeros(20 - x, y).long()], axis=0)

        if not self.is_test:
            start_score, end_score, confidence_score = self._get_train_label(start, end, self.anchor_xmin, self.anchor_xmax)
            return sentence, img_feat, confidence_score, start_score, end_score
        else:
            vid = os.path.split(video_prefix)[-1]
            return vid, sentence, img_feat

    def poolData(self, data, num_prop, num_bin=1, num_sample_bin=3, pool_type="mean"):
        feat_dim = data.shape[1]

        # TODO -- be more exact, and perhaps move this to preprocessing
        if len(data) == 1:
            video_feature = np.stack([data] * num_prop)
            video_feature = np.reshape(video_feature, [num_prop, feat_dim])
            return video_feature

        st = 1 / len(data)
        x = np.arange(len(data)) * st + st / 2
        f = scipy.interpolate.interp1d(x, data, axis=0)

        video_feature = []
        zero_sample = np.zeros(num_bin * feat_dim, dtype=np.float32)
        tmp_anchor_xmin = [1.0 / num_prop * i for i in range(num_prop)]
        tmp_anchor_xmax = [1.0 / num_prop * i for i in range(1, num_prop + 1)]

        num_sample = num_bin * num_sample_bin
        for idx in range(num_prop):
            xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx])
            xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx])
            if xmax < x[0]:
                video_feature.append(zero_sample)
                continue
            if xmin > x[-1]:
                video_feature.append(zero_sample)
                continue

            plen = (xmax - xmin) / (num_sample - 1)
            x_new = [xmin + plen * ii for ii in range(num_sample)]
            y_new = f(x_new).astype(np.float32)
            y_new_pool = []
            for b in range(num_bin):
                tmp_y_new = y_new[num_sample_bin * b:num_sample_bin * (b + 1)]
                if pool_type == "mean":
                    tmp_y_new = np.mean(y_new, axis=0)
                elif pool_type == "max":
                    tmp_y_new = np.max(y_new, axis=0)
                y_new_pool.append(tmp_y_new)
            y_new_pool = np.stack(y_new_pool)
            y_new_pool = np.reshape(y_new_pool, [-1])
            video_feature.append(y_new_pool)
        video_feature = np.stack(video_feature)
        return video_feature

    def _get_train_label(self, start, end, anchor_xmin, anchor_xmax):
        gt_bbox = []
        gt_iou_map = []
        for tmp_start, tmp_end in zip(start, end):
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map, [self.maxdur, self.tscale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)

        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        # calculate the ioa for all timestamp
        match_score_start, match_score_end = [], []
        for i in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[i], anchor_xmax[i], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[i], anchor_xmax[i], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.sample_list)


if __name__ == '__main__':
    import opt
    import time
    from model.bmn import BMN
    from model.loss_func import *

    args = opt.parse_opt()
    args.dataset_file = os.path.join('..', args.dataset_file)
    args.feature_root = os.path.join('..', args.feature_root)
    args.dur_file = os.path.join('..', args.dur_file)
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)
    train_dataset = ANetDataset(args, args.train_data_folder, text_proc, raw_data, test=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    '''model = BMN(args)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                args.learning_rate, weight_decay=1e-5, momentum=args.alpha, nesterov=True)'''
    bm_mask = get_mask(args.temporal_scale, args.max_duration)

    num_pos_neg = [0 for i in range(7)]
    for i in range(len(train_dataset)):
        sentence, img_feat, label_confidence, label_start, label_end = train_dataset[i]

        num_pos_neg[0:3] = add_pem_reg_num(label_confidence, bm_mask, num_pos_neg[0:3])
        num_pos_neg[3:5] = add_pem_cls_num(label_confidence, bm_mask, num_pos_neg[3:5])
        num_pos_neg[5:7] = add_tem_num(label_start, label_end, num_pos_neg[5:7])

    np.savetxt('../results/num_pos_neg.txt', np.array(num_pos_neg))
    exit(0)
    t = time.time()
    for i, x in enumerate(train_dataset):
        if i >= 100: break
    print(time.time() - t)