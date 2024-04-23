# coding='utf-8'
import numpy as np
import torch

if __name__ == "__main__":
    a = torch.from_numpy(np.random.rand(2, 3, 3))
    print(a)
    b = torch.mean(a, dim=0)
    print(b)
