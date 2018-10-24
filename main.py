# coding: UTF-8
import argparse
import os

from experiment import Experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unknown_args = parser.parse_known_args()

    # data
    parser.add_argument('--data_A_dir', type=str, default='data/horse2zebra/trainA')
    parser.add_argument('--data_B_dir', type=str, default='data/horse2zebra/trainB')
    parser.add_argument('--num_workers', type=int, default=4)
    # train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--log_report_freq', type=int, default=50)
    parser.add_argument('--scalar_report_freq', type=int, default=50)
    parser.add_argument('--image_report_freq', type=int, default=100)
    # loss
    parser.add_argument('--use_lsgan', action='store_true', default=False)
    parser.add_argument('--lambda_A', type=float, default=10.)
    parser.add_argument('--lambda_B', type=float, default=10.)
    parser.add_argument('--lambda_idt', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.)
    # misc
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--exp_name', type=str, default='garbage')

    args, unknown_args = parser.parse_known_args()

    args.data_A_dir = './data/horse2zebra/trainA'
    args.data_B_dir = './data/horse2zebra/trainB'
    args.batch_size = 1
    args.train_iters = int(1e+9)
    args.use_lsgan = True
    args.lr = 2e-4
    args.beta1 = 0.5
    args.device = 'cuda:0'
    args.exp_name = 'exp0.4'
    args.log_report_freq = 10
    args.scalar_report_freq = 10
    args.image_report_freq = 10

    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    if os.path.exists(args.log_dir):
        raise OSError('`{}` already exists.'.format(args.log_dir))
    else:
        os.makedirs(args.log_dir)

    Experiment(args).run()
