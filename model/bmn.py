# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class BMN(nn.Module):
    def __init__(self, args):
        super(BMN, self).__init__()
        self.tscale = args.temporal_scale
        self.maxdur = args.max_duration
        self.prop_boundary_ratio = args.prop_boundary_ratio
        self.num_sample = args.num_sample
        self.num_sample_perbin = args.num_sample_perbin
        self.feat_dim = args.feat_dim

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)),  # missing stride?
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        feature_map = self.x_1d_p(base_feature)
        feature_map = self._boundary_matching_layer(feature_map)
        feature_map = self.x_3d_p(feature_map).squeeze(2)
        confidence_map = self.x_2d_p(feature_map)
        return confidence_map, start, end, feature_map

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample, self.maxdur,
                                                        self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = np.arange(num_sample * num_sample_perbin) * plen_sample + seg_xmin
        p_mask = np.zeros([self.tscale, num_sample])
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([self.tscale])
            for sample in bin_samples:
                sample_upper = np.ceil(sample)
                sample_decimal, sample_down = np.modf(sample)
                sample_down, sample_upper = int(sample_down), int(sample_upper)
                if sample_down <= (self.tscale - 1) and sample_down >= 0:
                    bin_vector[sample_down] += 1 - sample_decimal
                if sample_upper <= (self.tscale - 1) and sample_upper >= 0:
                    bin_vector[sample_upper] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask[:,idx] = bin_vector
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map, w_{ij} in the paper
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.maxdur):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.num_sample, self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        print(mask_mat.shape, self.num_sample)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


if __name__ == '__main__':
    import opt

    args = opt.parse_opt()
    model = BMN(args)
    input = torch.randn(2, 3072, args.temporal_scale)
    a, b, c = model(input)
    print(a.shape, b.shape, c.shape)