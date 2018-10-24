from .discriminator import Discriminator
from .generator import Generator


def get_model(args):
    D = Discriminator(input_nc=3, ndf=64, n_layers=3)
    G = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    D = D.to(args.device)
    G = G.to(args.device)
    return D, G
