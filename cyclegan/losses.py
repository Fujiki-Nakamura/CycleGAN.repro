# coding: UTF-8
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, inpt, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(inpt)

    def __call__(self, inpt, target_is_real):
        target_tensor = self.get_target_tensor(inpt, target_is_real)
        return self.loss(inpt, target_tensor)
