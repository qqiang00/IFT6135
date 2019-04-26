from model import VAE, WGAN
from torchvision.utils import save_image
import os 
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print("cuda:", cuda)


#### qualitative evaluation 1
def show_random_sample(generator, img_dir, rows = 10, 
                       name = "gan", epochs = 200):
    z_dim = generator.z_dim
    batch_size = rows * rows
    z = Tensor(np.random.normal(0, 1, (batch_size, z_dim)))
    x_ = generator(z)
    save_image(x_.data, img_dir + "samples_{}.png".format(batch_size), 
               nrow = rows, normalize=True)


#### qualitative evaluation 2
def show_image_variation(generator, img_dir, epsilon = 0.1, dim = 0, 
                         epochs = 200, name = "gan"):
    z_dim = generator.z_dim
    z0 = Tensor(np.random.normal(0, 1, (1, z_dim)))
    #z0 = Tensor(np.zeros((1,z_dim)))
    zs = []
    k = 10
    for dim in range(z_dim):
        for i in range(-k, k+1):
            zi = z0.clone()
            zi[:,dim] += i * epsilon
            zs.append(zi)

    z = torch.cat(zs, 0)
    #print(z.shape)
    x = generator(z)
    save_image(x.data,  img_dir + "eps{}_k{}_var.png".format(epsilon, 2*k+1), 
               nrow = k*2+1, normalize=True)



#### qualitative evaluation 3
def show_space_interpolation(generator, img_dir, name = "gan", epochs = 200, 
                             ex = "0"):
    z_dim = generator.z_dim
    z0 = Tensor(np.random.normal(0, 1, (1, z_dim)))
    z1 = Tensor(np.random.normal(0, 1, (1, z_dim)))
    x0, x1 = generator(z0), generator(z1)
    k = 11
    z_prime = []# = Tensor(np.zeros((k, z_dim)))
    x_hat = [] # = Tensor(np.zeros((k, 3, 32, 32)))
    for i in range(k):
        alpha = i / (k - 1)
        z_prime.append(alpha * z0 + (1-alpha) * z1)
        x_hat.append(alpha * x0 + (1-alpha) * x1)

    z_prime = torch.cat(z_prime, dim = 0)
    x_hat = torch.cat(x_hat, dim = 0)
    x_prime = generator(z_prime)
    total_x = torch.cat([x_prime, x_hat], dim = 0)
    save_image(total_x.data, img_dir + "interpolate_{}.png".format(ex),
               nrow = k, normalize=True)


names = ["gan","vae"]
trained_epochs = {"vae": 220, "gan": 150}
model_paths = {"vae": "./model/best_model{}.vae".format(trained_epochs["vae"]),
               "gan": "./model/best_model{}.gan".format(trained_epochs["gan"])
              }
vae = VAE(z_dim = 100)              
gan = WGAN(z_dim = 100)
if cuda:
    gan.cuda()
    vae.cuda()

models = {"vae": vae, "gan": gan}


for name in names:
    print("For model {}:".format(name))
    model = models[name]
    model_path = model_paths[name]
    epochs = trained_epochs[name]
    model.load_state_dict(torch.load(model_path))   
    img_dir = "./image/problem3_2/{}/{}/".format(name, epochs)
    os.makedirs(img_dir, exist_ok = True)
    generator = model.generator
    show_random_sample(generator, rows = 16, img_dir = img_dir, 
                       name = name, epochs = epochs)
    print("  random sampled images saved.")
    show_image_variation(generator, epsilon = 0.2, img_dir = img_dir, 
                         name = name, epochs = epochs)  
    print("  variation images saved.")    
    for i in range(100):
        show_space_interpolation(generator, img_dir = img_dir, name = name, 
                                 epochs = epochs, ex = i)
    print("  interpolation images saved.")
    
    