import numpy as np
import torch.nn as nn
import torch
import random
import os
import math
import scipy.io as sio
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# build two directories for saving model and submission files.
data_dir = "./data/"
model_dir = "./model/"
image_dir = "./image/"
image_dir_gan = "./image/gan/"
image_dir_vae = "./image/vae/"

folders = [data_dir, model_dir, image_dir, image_dir_gan, image_dir_vae]
for i in range(len(folders)):
    os.makedirs(folders[i], exist_ok = True)

class Encoder(nn.Module):
    def __init__(self, z_dim = 100):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = (3,3)),      # 32 -> 30
            #nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size = (3,3)),     # 30 -> 28
            #nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size = (3,3)),     # 28 -> 26
            #nn.BatchNorm2d(64),            
            nn.ELU(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # 26 -> 13
            nn.Conv2d(64, 128, kernel_size = (3,3)),    # 13 -> 11
            #nn.BatchNorm2d(128),            
            nn.ELU(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # 11 -> 5
            nn.Conv2d(128, 256, kernel_size = (5,5)),   # 5  -> 1
            #nn.BatchNorm2d(256),            
            nn.ELU(),
        )
        self.linear1 = nn.Linear(in_features = 256, out_features = 2*z_dim)
        
        
    def forward(self, x):                          # [batch_size, 3, 32, 32]
        q = self.encode(x)                         # [batch_size, 256, 1, 1]
        q = self.linear1(q.view(q.size(0), -1))    # [batch_size, 200]
        mu, log_var = q[:,:z_dim], q[:, z_dim: ]   # [batch_size, 100] 
        return mu, log_var
    
    
class Generator(nn.Module):
    def __init__(self, z_dim = 100):
        """
        params
            z_dim: dimension of latent space
            sampling_generate: if use sampling to produce final image data, bool
                False for default
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        #self.linear2 = nn.Linear(in_features = z_dim, out_features = 256)
        self.decode = nn.Sequential(                       # 256, 1, 1
            #nn.ELU(),
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias = False),
            nn.Tanh() #nn.Sigmoid(), #, #.RELU(), 
            )
        
    def forward(self, z):                            # [batch_size, z_dim]
        x = z.view(z.size(0), -1, 1, 1)              # for conv transpose2d
        x = self.decode(x)
        return x
    
        
'''
class Discriminator(nn.Module):
    def __init__(self, input_dim = 3*32*32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # need a sigmoid to force output be 0 to 1??
        )

    def forward(self, x):    # [batch_size, 3, 32, 32]
        validity = self.model(x.view(x.size(0), -1))
        return validity
'''    
    
# use CNN as a discriminator, will try this later if the first doesn't work       
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(256, 1, 4, 1, 0, bias = False)
        )
        
        
    def forward(self, x):                         # [batch_size, 3, 32, 32]
        q = self.encode(x)                        # [batch_size, 256, 1, 1]
        #validity = self.fc(q.view(q.size(0), -1))        # [batch_size, 1]
        validity = q.view(-1, 1)
        return validity
    
    
class Discriminator3(nn.Module): # not applicable for WGAN, maybe for DCGAN
    def __init__(self):
        super(Discriminator3, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // (2 ** 4)
        self.adv_layer = nn.Sequential(nn.Linear(128 * (ds_size ** 2), 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    
    
def sampling(mu, log_var = None): # reparameterize.
    """this method can be used in two situations: 
    1) generate latent variables by output of an VAE encoder 
    2) generate reconstructed images by output of Generator in VAE  or WGAN
    """
    if log_var is None:
        std = Tensor(np.ones(tuple(mu.size())))
    else:
        std = torch.exp(0.5*log_var) 
    eps = torch.randn_like(std)
    return mu + torch.mul(eps, std)    
    
    
class VAE(nn.Module):
    def __init__(self, z_dim = 100):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim = z_dim)
        self.generator = Generator(z_dim = z_dim)
        self.sampling = sampling
    
    def forward(self, x):                 #[batch_size, 3, 32, 32]
        mu, log_var = self.encoder(x)     #[batch_size, 100] both
        z = self.sampling(mu, log_var)    #[batch_size, 100]
        x_ = self.generator(z)            #[batch_size, 3, 32, 32]
        #x_ = self.sampling(x_, None)      # work well here. :) remove also well.
        return x_, mu, log_var
    
    def kld(self, mu, log_var):
        for_sum = -1 - log_var + torch.pow(mu, 2) + torch.exp(log_var)
        kld = 0.5 * torch.sum(for_sum, dim = 1, keepdim = True)        
        return torch.mean(kld)
    
    def cross_entropy_loss(self, x_, x):
        #cross_entropy loss
        ce = F.binary_cross_entropy(x_, x, reduction = "none")
        #ce = F.binary_cross_entropy_with_logits(x_, x, reduction = "none")
        ce = torch.sum(ce.view(ce.size(0), -1), dim = 1, keepdim = True)
        ce = torch.mean(ce)  
        return ce
        
    def mse_loss(self, x_, x):
        # x_ is reconstructed x
        #
        mse = F.mse_loss(x_, x, reduction = "none")
        mse = torch.sum(mse.view(mse.size(0), -1), dim = 1, keepdim = True)
        mse = torch.mean(mse)  
        return mse        
        
    def loss(self, x_, x, mu, log_var):
        """
        params
            mu, log_var = encoder(x)
            recon_x: sampling(generator(sampling(mu, log_var)))
            x: original x
        returns
            MSE+KLD, MSE, KLD
        """
        mse, kld = self.mse_loss(x_, x), self.kld(mu, log_var)
        #mse, kld = self.cross_entropy_loss(x_, x), self.kld(mu, log_var)
        return mse + kld, mse, kld
    
    
class WGAN(nn.Module):
    def __init__(self, z_dim = 100):
        super(WGAN, self).__init__()
        self.generator = Generator(z_dim = z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):   # z is sampled from N(0, I) [batch_size, z_dim]
        # we don't directly use this method. use
        #x = self.generator(z)
        pass
    
    def generate(self, z):
        x_ = self.generator(z)
        return x_
    
    def discriminate(self, x):
        validity = self.discriminator(x.view(x.size(0), 3, 32, 32))
        return validity
    
    def grad_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        
        #print("shape of real fake samples:", real_samples.shape, fake_samples.shape)
        D = self.discriminator
        batch_size = real_samples.size(0)
        alpha = np.random.random((batch_size, 1, 1, 1))
        alpha = Tensor(alpha)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
        interpolates.requires_grad_(True)
        d_interpolates = D(interpolates)
        grad_outputs = Tensor(batch_size, 1).fill_(1.0)
        # Get gradient w.r.t. interpolates

        gradients = torch.autograd.grad(
            outputs = d_interpolates,
            inputs = interpolates,
            grad_outputs = grad_outputs,
            create_graph = True,
            retain_graph = True,
            only_inputs = True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        #gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
        return gradient_penalty
    
    def wasserstein_distance(self, real_validity, fake_validity):
        return torch.mean(real_validity) - torch.mean(fake_validity)
    
    