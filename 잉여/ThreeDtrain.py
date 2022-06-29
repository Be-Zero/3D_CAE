import torch
from torch.autograd.variable import Variable
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


def ones_target(size):
    '''
    Real data
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    Synthetic data
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 0))
    return data


def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
    cuda = next(discriminator.parameters()).is_cuda #CUDA확인
    N = real_data.size(0) #training data의 길이를 확인
    # Reset gradients
    optimizer.zero_grad() #Reset gradients tensors
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    target_real = ones_target(N)
    if cuda:
        target_real.cuda()

    error_real = loss(prediction_real, target_real)
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    target_fake = zeros_target(N)
    if cuda:
        target_fake.cuda()
    error_fake = loss(prediction_fake, target_fake)
    error_fake.backward()
    print('D_Loss:',error_fake)

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(generator, optimizer, fake_data, loss):
    cuda = next(generator.parameters()).is_cuda
    N = fake_data.size(0)  # Reset gradients
    optimizer.zero_grad()  # Sample noise and generate fake data
    prediction = generator(fake_data)  # Calculate error and     backpropagate
    target = ones_target(N)
    if cuda:
        target.cuda()
    # print('prediction: ',prediction)
    # print('target: ',target)
    # print('shape of prediction: ',np.shape(prediction))
    # print('shape of target: ',np.shape(target))
    error = loss(prediction, target)
    print('G_Loss:',error)
    error.backward()  # Update weights with gradients
    optimizer.step()  # Return error
    return error

# def DataLoad(dataroot,):
#     # We can use an image folder dataset the way we have it setup.
#     # Create the dataset
#     dataset = dset.ImageFolder(root=dataroot,
#                                transform=transforms.Compose([
#                                    transforms.Resize(image_size),
#                                    transforms.CenterCrop(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))
#     # Create the dataloader
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              shuffle=True, num_workers=workers)
#
#     # Decide which device we want to run on
#     device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#
#     # Plot some training images
#     real_batch = next(iter(dataloader))
#     plt.figure(figsize=(8, 8))
#     plt.axis("off")
#     plt.title("Training Images")
#     plt.imshow(
#         np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

def training(Discriminator,Generator,data, device = 'cuda',num_epochs=1,):
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    size = data[0]
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = ones_target(size)
    fake_label = zeros_target(size)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(data[0], 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            Discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = Discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = Generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = Discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            Generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = Discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = Generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1