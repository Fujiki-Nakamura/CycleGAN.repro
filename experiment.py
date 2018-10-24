# coding: UTF-8
import logging
import itertools
import os.path as osp

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from datasets import get_dataloader
from losses import GANLoss
from models import get_model
from utils import set_requires_grad, save_args


class Experiment():

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.start_iter = 1
        self.train_iters = args.train_iters
        # coeffs
        self.lambda_A = args.lambda_A
        self.lambda_B = args.lambda_B
        self.lambda_idt = args.lambda_idt

        self.dataloader_A, self.dataloader_B = get_dataloader(args)

        self.D_B, self.G_AB = get_model(args)
        self.D_A, self.G_BA = get_model(args)

        self.criterion_GAN = GANLoss(use_lsgan=args.use_lsgan).to(args.device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()

        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.D_B.parameters(), self.D_A.parameters()),
            lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

        self.logger = self.get_logger(args)
        self.writer = SummaryWriter(args.log_dir)

        save_args(args.log_dir, args)

    def run(self):
        iter_A = iter(self.dataloader_A)
        iter_B = iter(self.dataloader_B)
        iter_per_epoch = min(len(iter_A), len(iter_B))

        for step in range(self.start_iter, self.start_iter + self.train_iters):
            if step % iter_per_epoch == 0:
                iter_A = iter(self.dataloader_A)
                iter_B = iter(self.dataloader_B)

            real_A = iter_A.next()
            real_B = iter_B.next()
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            # train G
            set_requires_grad([self.D_A, self.D_B], False)
            self.optimizer_G.zero_grad()
            loss_G_AB, loss_G_BA, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B = (  # noqa
                self.backward_G(real_A, real_B, fake_A, fake_B))
            self.optimizer_G.step()

            # train D
            set_requires_grad([self.D_A, self.D_B], True)
            self.optimizer_D.zero_grad()
            loss_D_A = self.backward_D(self.D_A, real_A, fake_A)
            loss_D_B = self.backward_D(self.D_B, real_B, fake_B)
            self.optimizer_D.step()

            # logger
            if step % self.args.log_report_freq == 0:
                self.logger.info(
                    '{} Train: Step {} Loss/G_AB {:.4f}'.format(
                        self.args.exp_name, step, loss_G_AB))

            # writer
            if step % self.args.scalar_report_freq == 0:
                self.writer.add_scalar('Train/Loss/G_AB', loss_G_AB, step)
                self.writer.add_scalar('Train/Loss/cycle_A', loss_cycle_A, step)
                self.writer.add_scalar('Train/Loss/idt_A', loss_idt_A, step)
                self.writer.add_scalar('Train/Loss/G_BA', loss_G_BA, step)
                self.writer.add_scalar('Train/Loss/cycle_B', loss_cycle_B, step)
                self.writer.add_scalar('Train/Loss/idt_B', loss_idt_B, step)
                self.writer.add_scalar('Train/Loss/D_A', loss_D_A, step)
                self.writer.add_scalar('Train/Loss/D_B', loss_D_B, step)

            if step % self.args.image_report_freq == 0:
                real_A = real_A[-1].detach().cpu()
                real_B = real_B[-1].detach().cpu()
                fake_A = fake_A[-1].detach().cpu()
                fake_B = fake_B[-1].detach().cpu()
                self.writer.add_image('Train/real_A', real_A, step)
                self.writer.add_image('Train/fake_A', fake_A, step)
                self.writer.add_image('Train/real_B', real_B, step)
                self.writer.add_image('Train/fake_B', fake_B, step)

    def backward_D(self, D, real, fake):
        loss_D_real = self.criterion_GAN(D(real), target_is_real=True)
        loss_D_fake = self.criterion_GAN(D(fake), target_is_real=False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D

    def backward_G(self, real_A, real_B, fake_A, fake_B):
        loss_G_AB = self.criterion_GAN(self.D_B(fake_B), True)
        loss_G_BA = self.criterion_GAN(self.D_A(fake_A), True)

        # cyclic loss
        self.rec_A = self.G_BA(fake_B)
        self.rec_B = self.G_AB(fake_A)
        loss_cycle_A = self.criterion_cycle(self.rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.criterion_cycle(self.rec_B, real_B) * self.lambda_B
        loss_G = loss_G_AB + loss_G_BA + loss_cycle_A + loss_cycle_B
        loss_G.backward(retain_graph=True)

        # identity loss
        # TODO
        if self.lambda_idt > 0:
            self.idt_A = self.G_AB(real_B)
            loss_idt_A = self.criterion_idt(
                self.idt_A, real_B) * self.lambda_B * self.lambda_idt
            self.idt_B = self.G_BA(real_A)
            loss_idt_B = self.criterion_idt(
                self.idt_B, real_A) * self.lambda_A * self.lambda_idt
        else:
            loss_idt_A = 0.
            loss_idt_B = 0.

        return loss_G_AB, loss_G_BA, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B  # noqa

    def get_logger(self, args):
        logger = logging.getLogger(__name__ + args.exp_name)
        level = logging.DEBUG  # TODO
        logger.setLevel(level)
        for handler in [logging.StreamHandler(),
                        logging.FileHandler(osp.join(args.log_dir, 'log.txt'))]:
            handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
