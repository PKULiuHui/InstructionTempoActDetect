import argparse
import yaml
from data.utils import update_values

def parse_opt(train = True):
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--cfgs_file', default='cfgs/yc2.yml', type=str, help='dataset specific settings. anet | yc2')
    parser.add_argument('--dataset', default='yc2', type=str, help='which dataset to use. two options: anet | yc2')
    parser.add_argument('--dataset_file', default='', type=str)
    parser.add_argument('--feature_root', default='', type=str, help='the feature root')
    parser.add_argument('--dur_file', default='', type=str)
    parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
    if not train:
        parser.set_defaults(start_from='checkpoint/best_model.t7')
    parser.add_argument('--max_sentence_len', default=20, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    if train:
        parser.add_argument('--train_data_folder', default=['training'], type=str, nargs='+', help='training data folder')
        parser.add_argument('--val_data_folder', default=['validation'], help='validation data folder')
    else:
        parser.add_argument('--test_data_folder', default=['validation'], help='testing data folder')

    # Model settings: General
    # waiting for completion
    parser.add_argument('--temporal_scale', default=100, type=int, help='length of observation window')
    parser.add_argument('--prop_boundary_ratio', type=int, default=0.5)
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--num_sample_perbin', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default=3072)

    # Optimization: General
    parser.add_argument('--batch_size', default=4, type=int, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu')
    parser.add_argument('--id', default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')

    if train:
        # Optimization
        parser.add_argument('--max_epochs', default=20, type=int, help='max number of epochs to run for')
        parser.add_argument('--optim',default='sgd', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
        parser.add_argument('--learning_rate', default=0.02, type=float, help='learning rate')
        parser.add_argument('--alpha', default=0.95, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
        parser.add_argument('--beta', default=0.999, type=float, help='beta used for adam')
        parser.add_argument('--epsilon', default=1e-8, help='epsilon that goes into denominator for smoothing')
        parser.add_argument('--loss_alpha_r', default=2, type=int, help='The weight for regression loss')
        parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a pateau')
        parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')
        parser.add_argument('--grad_norm', default=1, type=float, help='Gradient clipping norm')

        # Data parallel
        parser.add_argument('--dist_url', default='file:///home/luozhou/nonexistent_file', type=str, help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
        parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

        # Evaluation/Checkpointing
        parser.add_argument('--save_checkpoint_every', default=5, type=int, help='how many epochs to save a model checkpoint?')
        parser.add_argument('--checkpoint_path', default='./checkpoint', help='folder to save checkpoints into (empty = this folder)')
        parser.add_argument('--losses_log_every', default=1, type=int, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
        parser.add_argument('--seed', default=213, type=int, help='random number generator seed to use')
        parser.add_argument('--enable_visdom', action='store_true', dest='enable_visdom')
        parser.set_defaults(cuda=True, enable_visdom=False)
    else:
        # Model settings: Proposal and mask
        parser.add_argument('--soft_nms_alpha', default=0.4, type=float)
        parser.add_argument('--soft_nms_low_thres', default=0.5, type=float)
        parser.add_argument('--soft_nms_high_thres', default=0.9, type=float)

    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    import os
    path = os.path.dirname(__file__)
    with open(os.path.join(path, args.cfgs_file), 'r') as handle:
        options_yaml = yaml.load(handle)
    update_values(options_yaml, vars(args))

    return args