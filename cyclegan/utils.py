# coding: UTF-8
import os


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is None:
            continue
        for param in net.parameters():
            param.requires_grad = requires_grad


def save_args(log_dir, args):
    fpath = os.path.join(log_dir, 'args.txt')
    with open(fpath, 'w') as f:
        f.writelines(
            ['{}: {}\n'.format(arg, getattr(args, arg))
             for arg in dir(args) if not arg.startswith('_')])
    os.system('cat {}'.format(fpath))
