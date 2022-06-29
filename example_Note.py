# # import os
# # shape_dir='D:/3DShapeNets'
# # shape_labels= r"D:/3DShapeNets/volumetric_data"
# # shape_path = os.path.join(shape_labels,'30','train')
# # print(shape_path)
#
#
# import os
# import glob
# import numpy as np
# import scipy.io as io
# import scipy.ndimage as nd
# import matplotlib.pyplot as plt
#
#
# airplanes = glob.glob(r'D:\3DShapeNets\volumetric_data\airplane\30\train'+'/*.mat')
# file_list = airplanes[0:int(1.0 * len(airplanes))]
# print(len(file_list))
# test_image_airplane = file_list[0:2]
#
# for idx, data in enumerate(test_image_airplane):
#
#     voxels = io.loadmat(data)['instance']
#     voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
#     voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
#
#     volumes = np.asarray([voxels for f in range(len(test_image_airplane))], dtype=bool)
#     print(np.shape(volumes))
#     # print(volumes[0][32][32])
# # real=np.array_equal(volumes[0], volumes[2], equal_nan=False)
# # print(real)
# # print(np.shape(volumes))
#
# # import numpy as np
# # import scipy.io as io
# # import scipy.ndimage as nd
# # import matplotlib.pyplot as plt
# #
# # test_image_airplane = airplanes[0]
# # voxels = io.loadmat(test_image_airplane)['instance']
# # voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
# # voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
# # print(voxels)
# #
# # # 3D Voxel 시각화
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # #ax.set_aspect('equal')
# # ax.voxels(voxels, edgecolor='red')
# # plt.show()
#
#
#
#
# import numpy as np
import torch
# from Generator import Generator
# from Discriminator import Discriminator
# import matplotlib.pyplot as plt
#
#
# noise_dim = 200  # latent space vector dim
# input_dim = 512  # convolutional channels
# dim = 64  # cube volume
# noise = torch.rand(1, noise_dim)
#
# model_generator = Generator(input_dim=input_dim, out_dim=dim, out_channels=1, noise_dim=noise_dim)
# generated_volume = model_generator(noise)
# print("model_generator output shape", generated_volume.shape,generated_volume)
# model_discriminator = Discriminator(in_channels=1, dim=64, out_conv_channels=512)
# out = model_discriminator(generated_volume)
# print("model_discriminator output", out.item())
#
#
# volume_list = generated_volume.detach().numpy()
# volume_list=np.reshape(volume_list,(64,64,64))
#
# # 3D Voxel 시각화
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.set_aspect('equal')
# ax.voxels(volume_list, edgecolor='red')
# plt.show()
#
#


device = torch.device("cuda:0" if (torch.cuda.is_available() and 64 > 0) else "cpu")

print(device)






