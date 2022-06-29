import torch
from torch import nn

def network_loss(recon_x, x):
    L2 = torch.mean((recon_x - x) ** 2)
    KLD_func = nn.KLDivLoss() #Kullback-leibler divergence
    KLD= KLD_func(recon_x,x) #Adequetely, using like classes. Call the pre-line, and using
    return L2 + KLD
