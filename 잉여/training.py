import torch
import torch.nn as nn
from torchsummary import summary
import torch
from torch import nn, optim
import argparse
import Generator

def training(discriminator, generator, data, args):
    # init weight normal distr.
    generator.apply(init_weights_normal)
    discriminator.apply(init_weights_normal)

    if args.cuda:
        discriminator.cuda()
        generator.cuda()
        # 10 test samples
        test_noise = noise(10).cuda()

    # LOSSES
    loss_d = nn.BCELoss()
    loss_g = nn.BCELoss()

    # OPTIMIZERS

    d_optimizer = optim.Adam(discriminator.parameters(), args.lr_d, betas=args.betas)
    g_optimizer = optim.Adam(generator.parameters(), args.lr_g, betas=args.betas)

    # Data loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    num_batches = len(data_loader)

    for epoch in range(args.start_epoch, args.num_epochs):
        for n_batch, sample_subject in enumerate(data_loader):
            iteration = epoch * num_batches + n_batch

            real_data = sample_subject

            # 1. Train Discriminator
            latent_space_data = noise(args.batch_size)
            if args.cuda:
                latent_space_data = latent_space_data.cuda()
                real_data = real_data.cuda()

                fake_data = generator(latent_space_data).detach()

                d_error, d_pred_real, d_pred_fake = train_discriminator(
                    discriminator, d_optimizer, real_data, fake_data,
                    loss_d, epoch, args.cuda)
                print('Discriminator_loss: ',d_error)
                # yield d_error
                print('Discriminator_real_loss: ',d_pred_real)
                print('Discriminator_pred_loss: ',d_pred_fake)

            # 2. Train Generator
            latent_space_data = noise(args.batch_size)
            if args.cuda:
                latent_space_data = latent_space_data.cuda()

            # print(np.shape(latent_space_data))
            fake_data = generator(latent_space_data)
            g_error = train_generator(discriminator, g_optimizer, fake_data,  args.loss)  # Log batch error
            # print('Generator_loss: ', g_error)

    return d_error,d_pred_real, d_pred_fake, g_error
            # save the stats that you want here :)


def init_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


def noise(size, latent_dim=200):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = torch.randn(size, latent_dim)
    return n


# Functions that you ll definately need :)
import os, shutil


def save_checkpoint_gan(state_d, state_g, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name_d = prefix_save + '_D_' + filename
    name_g = prefix_save + '_G_' + filename
    torch.save(state_d, name_d)
    torch.save(state_g, name_g)
    if is_best:
        shutil.copyfile(name_d, prefix_save + '_model_D_best.pth.tar')


def load_checkpoints_gan(model_d, model_g, path_d, path_g):
    checkpoint_d = torch.load(path_d)
    checkpoint_g = torch.load(path_g)
    model_d.load_state_dict(checkpoint_d['state_dict'])
    model_g.load_state_dict(checkpoint_g['state_dict'])
    return model_d, model_g
G = Generator()
