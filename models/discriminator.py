# coding: UTF-8
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L314  # noqa
    '''

    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=True),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
