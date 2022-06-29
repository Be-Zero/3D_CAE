
import torch.nn as nn

class Args():
    cuda = True
    betas = [0.5,0.5]
    start_epoch = 1
    num_epochs = 100
    batch_size = 64
    lr_g = 0.0002
    lr_d = 0.0002
    ndf = 64
    ngf = 64
    latent_dim = 128
    img_size = 64
    channels = 3
    n_critic = 5
    split_rate = 0.8
    loss = nn.BCELoss()

args = Args()