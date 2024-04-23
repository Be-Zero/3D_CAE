# coding='utf-8'
import numpy as np
import scipy.io as scio
import torch
import torch.utils.data as data
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from bp_net import BpNet
from cae import Cae
from Indian_pines import CaeEncoder, Indian_pines
import torch
from sklearn import metrics
from torch import nn

class Cae_Bp(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cae_name = 'cae_ep46_v0.0009'

    def __init__(self, out_conv_channels=32, base_net=[800, 256, 128, 16], p=0.3):
        super(Cae_Bp, self).__init__()
        self.cae = Cae(in_channels=1, out_conv_channels=out_conv_channels)
        self.bp = BpNet(base_net=base_net, p=p)
        self.cae.get_summary()
        
    def train_cae(self, lr=0.001, num_epochs=50, loss_type='mse', optimizer_type='Adam', betas=(0.9, 0.999)):
        # load datasets
        indian_pines = Indian_pines(train='True', test_size=0.1)
        train_loader = torch.utils.data.DataLoader(indian_pines, batch_size=16, shuffle=True)
        indian_pines.valid_mode()
        valid_loader = torch.utils.data.DataLoader(indian_pines, batch_size=16, shuffle=True)
        self.cae_name = self.cae.fit(train_loader=train_loader, valid_loader=valid_loader, lr=lr, num_epochs=num_epochs, loss_type=loss_type, optimizer_type=optimizer_type, betas=betas)
        self.cae.draw(self.valid_loader)
        
    def train_bp(self, lr=0.001, num_epochs=200, loss_type='cross-entropy', optim_type='Adam'):
        all_loader = torch.utils.data.DataLoader(Indian_pines(
            train='all'), batch_size=16, shuffle=False, num_workers=0)
        self.cae.load_model('./model/Indian_pines/' + self.cae_name + '.pt')
        encoder, label = self.cae.get_encoder(all_loader)
        train_loader = torch.utils.data.DataLoader(CaeEncoder(
            encoder, label, train=True, test_size=0.9), batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(CaeEncoder(
            encoder, label, train=False, test_size=0.9), batch_size=16, shuffle=True)
        self.bp_name = self.bp.fit(train_loader=train_loader, valid_loader=valid_loader, lr=lr, num_epochs=num_epochs, loss_type=loss_type, optim_type=optim_type)
    
    def pseudo_tag_generation(self, inputs, labels, k=10):
        self.bp.train()
        pseudo_tags = []
        with torch.no_grad():
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for i in range(k):
                outputs = self.forward(inputs)
                pseudo_tags.append(outputs)
        pseudo_tags = torch.mean(pseudo_tags, dim=0)
        
    def partition_data(self):
        all_loader = torch.utils.data.DataLoader(Indian_pines(
            train='all'), batch_size=16, shuffle=False, num_workers=0)
        self.cae.load_model('./model/Indian_pines/' + self.cae_name + '.pt')
        encoder, label = self.cae.get_encoder(all_loader)
        cae_encoder = CaeEncoder(encoder, label, train=True, test_size=0.9)
        train_loader = torch.utils.data.DataLoader(cae_encoder, batch_size=16, shuffle=True)
        cae_encoder.valid_mode()
        valid_loader = torch.utils.data.DataLoader(cae_encoder, batch_size=16, shuffle=True)
        
if __name__ == "__main__":
    cae_bp = Cae_Bp()
    # cae_bp.train_cae()
    # cae_bp.train_bp()
    