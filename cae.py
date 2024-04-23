import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class Cae(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, in_channels=1, out_conv_channels=64):
        """_summary_

        Args:
            in_channels (int, optional): number of channels, number of bands. Defaults to 1.
            dim (int, optional): image size. Defaults to 64.
            out_conv_channels (int, optional): _description_. Defaults to 64.
        """
        super(Cae, self).__init__()
        conv1_channels = int(out_conv_channels / 4)
        conv2_channels = int(out_conv_channels / 2)
        dconv1_channels = conv2_channels
        dconv2_channels = conv1_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(  # in channels, out channels, kernel size, stride, padding
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv1_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv2_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=out_conv_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=out_conv_channels, out_channels=out_conv_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2)
        )

        self.dconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(
                in_channels=out_conv_channels, out_channels=out_conv_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_conv_channels, out_channels=out_conv_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.ReLU(),
        )

        self.dconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(
                in_channels=out_conv_channels, out_channels=dconv1_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(dconv1_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=dconv1_channels, out_channels=dconv1_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(dconv1_channels),
            nn.ReLU(),
        )

        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(
                in_channels=dconv1_channels, out_channels=dconv2_channels, kernel_size=(3, 2, 2),
                stride=1, padding=1
            ),
            nn.BatchNorm3d(dconv2_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=dconv2_channels, out_channels=dconv2_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(dconv2_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=dconv2_channels, out_channels=in_channels, kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
        )
        self.to(self.device)

    def forward(self, x):
        # N:batch_size, C:channels, D:bands, H:hight of image, W:width of image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y = self.dconv1(x)
        y = self.dconv2(y)
        y = self.dconv3(y)
        return x, y

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def fit(self, train_loader, valid_loader, lr=0.001, num_epochs=50, loss_type='mse', optimizer_type='Adam', betas=(0.9, 0.999)):
        # Initialize BCELoss function
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "cross-entropy":
            criterion = nn.CrossEntropyLoss()

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        # validate
        self.eval()
        total_loss = 0.0
        total_num = 0
        with torch.no_grad():
            for inputs, _ in valid_loader:
                inputs = inputs.to(self.device)
                _, outputs = self.forward(inputs)
                valid_recon_loss = criterion(outputs, inputs)
                total_loss += valid_recon_loss.data * len(inputs)
                total_num += inputs.size()[0]

        min_valid_loss = 100.0
        valid_loss = total_loss / total_num

        information = []
        print("#Epoch 0: Valid Reconstruct Loss: {:.4f}".format(valid_loss))
        information.append(
            "#Epoch 0: Valid Reconstruct Loss: {:.4f}\n".format(valid_loss))
        best_epoch = 0

        for epoch in range(num_epochs):
            self.train()
            # For each batch in the dataloader
            train_loss = 0.0
            with torch.enable_grad():
                for inputs, _ in train_loader:
                    optimizer.zero_grad()
                    inputs = inputs.to(self.device)
                    _, outputs = self.forward(inputs)
                    recon_loss = criterion(outputs, inputs)
                    train_loss += recon_loss.data * len(inputs)
                    recon_loss.backward()
                    optimizer.step()

            # validate
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, _ in valid_loader:
                    inputs = inputs.to(self.device)
                    _, outputs = self.forward(inputs)
                    valid_recon_loss = criterion(outputs, inputs)
                    valid_loss += valid_recon_loss.data * len(inputs)

            print("#Epoch {:3d}: Reconstruct Loss: {:.4f}, Valid Reconstruct Loss: {:.4f}".format(
                epoch+1, train_loss / len(train_loader.dataset), valid_loss / len(valid_loader.dataset)))
            information.append("#Epoch {:3d}: Reconstruct Loss: {:.4f}, Valid Reconstruct Loss: {:.4f}\n".format(
                epoch+1, train_loss / len(train_loader.dataset), valid_loss / len(valid_loader.dataset)))
            valid_loss = valid_loss / len(valid_loader.dataset)
            if min_valid_loss >= valid_loss:
                min_valid_loss = valid_loss
                best_epoch = epoch + 1
                self.save_model('./model/Indian_pines/cae_current.pt')  
        rt_name = 'cae_ep{}_v{:.4f}'.format(best_epoch, min_valid_loss)
        os.rename('./model/Indian_pines/cae_current.pt', './model/Indian_pines/' + rt_name + '.pt')
        with open('./logs/Indian_pines/' + rt_name + '.log', 'w') as f:
            for str in information:
                f.write(str)
        return rt_name

    def get_summary(self):
        device = torch.device("cuda")  # PyTorch v0.4.0
        model = Cae().to(device)
        summary(model, (1, 200, 9, 9))  # n, c, d, h/w

    def get_encoder(self, dataloader):
        encoder = []
        label = []
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            outputs, _ = self.forward(inputs)
            for j in range(outputs.size(0)):
                encoder.append(outputs[j].data.cpu().numpy())
                label.append(labels[j].data.cpu().numpy())
        encoder = np.array(encoder)
        label = np.array(label)
        return encoder, label

    
    def draw(self, dataloader):
        imgIn = []
        imgOut = []
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            _, outputs = self.forward(inputs)
            for j in range(inputs.size(0)):
                imgIn.append(inputs[j][0,0,:,:].data.cpu().numpy())
                imgOut.append(outputs[j][0,0,:,:].data.cpu().numpy())
        imgIn = np.array(imgIn)
        imgOut = np.array(imgOut)
        
        plt.figure(figsize=(90,90), dpi=5)
        plt.axis('off')
        for i in range(200):
            print('{}th start...'.format(i+1), end='')
            ax = plt.subplot(1, 2, 1)
            plt.gray()
            ax.set_title('autoencoder')
            plt.imshow(imgOut[i])
            ax = plt.subplot(1, 2, 2)
            plt.gray()
            ax.set_title('original')
            plt.imshow(imgIn[i])
            plt.savefig('./picture/Indian_pines/{}.jpg'.format(i+1))
            print(', {}th picture has been done!'.format(i+1))