import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from preprocess import get_loaders, subject_files
from train_region import train_one_fold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='RSM-CoDG cross-subject EEG emotion recognition on SEED')
    parser.add_argument('--data_path', default='../Unified-EEG/data/seed/ExtractedFeatures/',
                        help='folder with one sub-folder of .mat files per session')
    parser.add_argument('--session', default='1')
    parser.add_argument('--targets', type=int, nargs='*', help='target subjects to run (default: all)')
    parser.add_argument('--num_val_subjects', type=int, default=1,
                        help='source subjects held out per fold for early stopping')
    parser.add_argument('--way', default='RSM-CoDG/seed', help='experiment name (log and model folder)')
    parser.add_argument('--index', default='region', help='run name (log and model sub-folder)')
    parser.add_argument('--epoch_training', type=int, default=150)
    parser.add_argument('--patience', type=int, default=40,
                        help='stop after this many epochs without validation improvement')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--samples_per_subject', type=int, default=64,
                        help='windows drawn from each training subject per update')
    parser.add_argument('--iterations', type=int, default=7, help='updates per epoch')
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lambda_mmd', type=float, default=0.15)
    parser.add_argument('--lambda_contrast', type=float, default=0.2)
    parser.add_argument('--lambda_decorr', type=float, default=0.05)
    parser.add_argument('--dg_warmup_epochs', type=int, default=20)
    parser.add_argument('--dg_max_weight', type=float, default=1.0)
    parser.add_argument('--no_calibration', action='store_true', help='remove the calibration layer')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()

    args.num_classes = 3
    args.time_steps = 30
    args.dg_loss_weights = {'decorrelation': args.lambda_decorr, 'mmd': args.lambda_mmd,
                            'contrastive': args.lambda_contrast}
    return args


def main():
    args = parse_args()
    cuda = torch.cuda.is_available()
    writer = SummaryWriter(f"data/session{args.session}/{args.way}/{args.index}")
    targets = args.targets or sorted(subject_files(os.path.join(args.data_path, args.session)))
    print(f"RSM-CoDG | SEED session {args.session} | {args.num_val_subjects} validation subjects per fold | "
          f"calibration {not args.no_calibration} | loss weights {args.dg_loss_weights}")

    test_accs, val_accs = [], []
    for target in targets:
        train_loaders, val_loader, test_loader, train_ids, val_ids = get_loaders(target, args)
        print(f"===== Target {target} | validation subjects {val_ids} | training subjects {train_ids} =====")
        set_seed(args.seed)
        test_acc, val_acc, _ = train_one_fold(train_loaders, val_loader, test_loader, args, cuda, writer,
                                              target, train_ids, val_ids)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        writer.flush()

    mean, std = np.mean(test_accs), np.std(test_accs)
    print(f"Test acc {mean:.4f} +- {std:.4f} | mean best val acc {np.mean(val_accs):.4f}")
    print('Per target:', {t: round(a, 4) for t, a in zip(targets, test_accs)})
    writer.add_text('test acc mean/std', f'{mean:.4f} +- {std:.4f}')
    writer.add_text('test acc per target', ','.join(f'{t}:{a:.4f}' for t, a in zip(targets, test_accs)))
    writer.close()


if __name__ == '__main__':
    main()
