import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from Mat_voxelizer import EX_airplane
from Decoder import Decoder
from Encoder import Encoder
from matplotlib import pyplot as plt
from Loss import network_loss

#Data Load
DIR_PATH = r'D:\3DShapeNets\volumetric_data\airplane\30\train'
Airplane_Data = EX_airplane(DIR_PATH,object_ratio=0.1)
print(Airplane_Data)
print(np.shape(Airplane_Data))
print(len(Airplane_Data))
# # Airplane_Data=np.reshape(Airplane_Data,(len(Airplane_Data),64,64,64))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.set_aspect('equal')
# ax.voxels(Airplane_Data[0], edgecolor='red')
# plt.show()
#


#Hyper-Parameters
num_epochs = int(300)
lr_D = 0.0002
lr_G = 0.0002
beta1=0.5
device = torch.device("cuda:0" if (torch.cuda.is_available() and 64 > 0) else "cpu")
batch_size=64

# Initialize BCELoss function
criterion = nn.BCELoss().to(device)
size = len(Airplane_Data)
print(size)

#Creat model_generator and model_discriminator
noise_dim = 200  # latent space vector dim
input_dim = 512  # convolutional channels
dim = 64  # cube volume
# noise = torch.rand(1, noise_dim).to(device)
# print(noise.is_cuda)

print(torch.Tensor(Airplane_Data[0]))

model_Encoder = Encoder(in_channels=1, dim=64, out_conv_channels=512).to(device)
# # sample_z = model_Encoder(torch.Tensor(Airplane_Data[0]))
# print("model_Encoder output", sample_z.item())
#
model_Decoder = Decoder(input_dim=input_dim, out_dim=dim, out_channels=1, noise_dim=noise_dim,activation="sigmoid").to(device)
# generated_volume = model_Decoder(sample_z)
# print("model_Decoder output shape", generated_volume.shape,generated_volume)




# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, device=device)


# Setup Adam optimizers for both G and D
optimizerE = optim.Adam(model_Encoder.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerD = optim.Adam(model_Decoder.parameters(), lr=lr_G, betas=(beta1, 0.999))

shape_list = []
D_losses = []
E_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for idx in range(len(Airplane_Data)):

        ############################
        # 3D-CAE network: maximize L2 norm
        ###########################
        ## Encoder
        model_Encoder.zero_grad()
        # Format batch
        real_cuda = torch.Tensor(Airplane_Data[idx]).to(device)
        b_size = batch_size
        # Forward pass real batch through D
        output_z = model_Encoder(real_cuda)

        ## Decoder
        Reconstruction = model_Decoder(output_z)
        errD = network_loss(Reconstruction, real_cuda)
        errD.backward()
        # Update E & D
        optimizerD.step()
        optimizerE.step()

        if idx % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f'
                  % (epoch, num_epochs, idx, size,
                     errD.item()))

        # Save Losses for plotting later
        # G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (idx == size - 1)):
            with torch.no_grad():
                fake = model_Decoder(output_z).detach().cpu()
            shape_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

#Saving model weight
SAVE_PATH = r'C:\Users\USER\PycharmProjects\Generative_design\pretrained'
torch.save(model_Decoder, SAVE_PATH + '\model_G.pt')
torch.save({
    'model': model_Decoder.state_dict(),
    'optimizer': optimizerD.state_dict()
}, SAVE_PATH + r'\all.tar')
torch.save(model_Encoder, SAVE_PATH + '\model_D.pt')
torch.save({
    'model': model_Encoder.state_dict(),
    'optimizer': optimizerD.state_dict()
}, SAVE_PATH + r'\all.tar')

#Generated Shape plotting
generated_volume = model_Decoder(torch.rand(1, noise_dim).to(device))
generated_volume_cpu = generated_volume.cpu()
volume_list = generated_volume_cpu.detach().numpy()
volume_list=np.reshape(volume_list,(64,64,64))

# 3D Voxel Visualization
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.set_aspect('equal')
ax.voxels(volume_list, edgecolor='red')
plt.show()

