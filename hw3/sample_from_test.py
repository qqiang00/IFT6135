import os
import numpy as np
import torch.nn as nn
import torch
import random
import math
import scipy.io as sio
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Normalize
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print(cuda)

from classify_svhn import get_data_loader
# build two directories for saving model and submission files.
data_dir = "./data/"

train_loader, valid_loader, test_loader = get_data_loader(data_dir, 32)

n_samples = 1000
enough_images = False
n_produced = 0
save_dir = "./image/test_1000/samples/"
os.makedirs(save_dir, exist_ok=True)
unnormalize = Normalize((-1., -1., -1.),
                          (2, 2, 2))

for idx, data in enumerate(test_loader):
    test_x, y = data
    batch_size = test_x.size(0)
    for i, test_xi in enumerate(test_x):
        print(torch.max(test_xi), torch.min(test_xi))
        
        test_xi = unnormalize(test_xi)
        save_image(test_xi.data, 
                   save_dir + "{}.png".format(n_produced), 
                   nrow = 1, normalize=False)
        n_produced += 1
        if n_produced >= n_samples:
            enough_images = True
            break
    if enough_images:
        break

print("total produced:", n_produced)
    