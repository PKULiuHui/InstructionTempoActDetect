import os
import numpy as np

splits = ['training', 'testing', 'validation']
video_dur_frame = []
with open('./yc2/yc2_duration_frame.csv', 'r') as f:
    for line in f:
        name, dur, frame = [l.strip() for l in line.split(',')]
        dur, frame = float(dur), float(frame)
        interval = dur * np.ceil(frame / dur * 0.5) / frame
        frame_num = dur / interval
        frame_num_1 = dur / 0.5
        feat_num = -1
        for split in splits:
            if os.path.exists('%s/%s_bn.npy' % (split, name)):
                tmp = np.load('%s/%s_bn.npy' % (split, name))
                feat_num = tmp.shape[0]
        # print(feat_num, round(frame_num))
        if feat_num != int(round(frame_num)) and feat_num != int(round(frame_num_1)):
            print(feat_num, frame_num, frame_num_1)
