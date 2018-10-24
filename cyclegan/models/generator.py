# coding: UTF-8
import torch.nn as nn


class Generator(nn.Module):
    '''
    originally from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L142  # noqa
    '''

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, args=None):
        super(Generator, self).__init__()

        self.input_nc = None
        self.output_nc = None
        self.ngf = None
        padding_type = 'reflect'

        self.layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        n_downsample = 2
        for i in range(n_downsample):
            mult = 2**i
            self.layers += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2,
                    kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        mult = n_downsample**2
        for i in range(n_blocks):
            self.layers += [
                ResnetBlock(
                    ngf * mult, padding_type=padding_type, norm_layer=nn.InstanceNorm2d,
                    use_dropout=False, use_bias=True)
            ]
        for i in range(n_downsample):
            mult = 2**(n_downsample - i)
            self.layers += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
                ]
        self.layers += [nn.ReflectionPad2d(3)]
        self.layers += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.layers += [nn.Tanh()]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ResnetBlock(nn.Module):
    '''
    from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L191  # noqa
    '''
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
