# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F


def get_mask(tscale, maxdur):
    bm_mask = []
    for idx in range(maxdur):
        mask_vector = [1 for i in range(tscale - idx)] + [0 for i in range(idx)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.float32)
    return torch.Tensor(bm_mask)


def add_pem_reg_num(gt_iou_map, mask, num_pos_neg):
    num_h, num_m, num_l = num_pos_neg
    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h += torch.sum(u_hmask).item()
    num_m += torch.sum(u_mmask).item()
    num_l += torch.sum(u_lmask).item()
    return [num_h, num_m, num_l]


def add_pem_cls_num(gt_iou_map, mask, num_pos_neg):
    num_pos, num_neg = num_pos_neg
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_pos += torch.sum(pmask).item()
    num_neg += torch.sum(nmask).item()
    return [num_pos, num_neg]


def add_tem_num(gt_start, gt_end, num_pos_neg):
    def add(gt_label, num_pos_neg):
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_positive = torch.sum(pmask).item()
        num_pos_neg[0] += num_positive
        num_pos_neg[1] += len(pmask) - num_positive

    # pos and neg number of start and end should be similar
    add(gt_start, num_pos_neg)
    add(gt_end, num_pos_neg)
    return num_pos_neg


def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask, num_pos_neg):
    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    gt_iou_map = gt_iou_map * bm_mask

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask, num_pos_neg[0:3])
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask, num_pos_neg[3:5])
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end, num_pos_neg[5:7])

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask, num_pos_neg):
    num_h, num_m, num_l = num_pos_neg
    assert num_h > 0 and num_m > 0 and num_l > 0

    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask, num_pos_neg):
    npos, nneg = num_pos_neg
    assert npos > 0 and nneg > 0

    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    ratio = (npos + nneg) / npos
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    loss_pos = -coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = -coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = torch.sum(loss_pos + loss_neg) / gt_iou_map.nelement()
    return loss


def tem_loss_func(pred_start, pred_end, gt_start, gt_end, num_pos_neg):
    def bi_loss(pred_score, gt_label, npos, nneg):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        ratio = (nneg + npos) / npos
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 1e-6
        loss_pos = -coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = -coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
        loss = torch.mean(loss_pos + loss_neg)
        return loss

    npos, nneg = num_pos_neg  # pos and neg number of start and end should be similar
    assert npos > 0 and nneg > 0

    loss_start = bi_loss(pred_start, gt_start, npos, nneg)
    loss_end = bi_loss(pred_end, gt_end, npos, nneg)
    loss = loss_start + loss_end
    return loss


if __name__ == '__main__':
    import opt

    args = opt.parse_opt()
    tscale, maxdur = args.temporal_scale, args.max_duration
    bm_mask = get_mask(tscale, maxdur)
    pred_bm = torch.ones([1, maxdur, tscale]) / 2
    pred_start, pred_end = torch.ones([1, tscale]) / 2, torch.ones([1, tscale]) / 2
    gt_iou_map, gt_start, gt_end = map(lambda x: torch.ones_like(x) / 4, [pred_bm, pred_start, pred_end])
    pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask = map(lambda x: x.cuda(),
                                                                               [pred_bm, pred_start, pred_end,
                                                                                gt_iou_map, gt_start, gt_end, bm_mask])
    total_loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map,
                                                                     gt_start, gt_end, bm_mask)
    print(total_loss, tem_loss, pem_cls_loss, pem_reg_loss)
