from model import VAE, WGAN
from torchvision.utils import save_image
from torchvision.transforms import Normalize
import os 
import torch
import numpy as np

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print(cuda)

# build two directories for saving model and submission files.
data_dir = "./data/"
model_dir = "./model/"
image_dir = "./image/"
image_dir_gan = image_dir +"gan/"
image_dir_vae = image_dir +"vae/"

folders = [data_dir, model_dir, image_dir, image_dir_gan, image_dir_vae]
for i in range(len(folders)):
    os.makedirs(folders[i], exist_ok = True)

def sample_and_save_images(model, n_samples, epoch, name):
    unnormalize = Normalize((-1., -1., -1.),
                          (2, 2, 2))
    z_dim = 100
    save_dir = image_dir + "{}/{}epoch/samples/".format(name, epoch)
    print("create dir: {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    z = Tensor(np.random.normal(0, 1, (n_samples, z_dim)))
    fake_x = model.generator(z)
    print(torch.max(fake_x), torch.min(fake_x))
    
    #t_mean = torch.tensor([.5, .5, .5], dtype=torch.float32)
    #t_std   = torch.tensor([.5, .5, .5], dtype=torch.float32)
    #unnormalize = Normalize((-t_mean / t_std).tolist(), (1.0 / t_std).tolist())

    for i, fake_xi in enumerate(fake_x):
        #img_data = (fake_xi.data.cpu().numpy() * 255).astype('int')
        img_path = save_dir + "{}.png".format(i)
        fake_xi = unnormalize(fake_xi) 
        save_image(fake_xi.data, img_path, nrow = 1, normalize=False)
        
    save_image(fake_x.data, image_dir +"{}/{}_{}.png".format(name, epoch,n_samples), 
               nrow = 32, normalize=False) 
    
        
    
z_dim = 100
gan = WGAN(z_dim = z_dim)
vae = VAE(z_dim = z_dim)
if cuda:
    gan.cuda()
    vae.cuda()

trained_epochs = {"gan": 150, "vae":70}
model_names = ["gan", "vae"] #["gan", "vae"]
models = {"gan": gan, "vae": vae}
model_files = {"gan": model_dir + "best_model{}.gan".format(trained_epochs["gan"]), 
               "vae": model_dir + "best_model{}.vae".format(trained_epochs["vae"])
               }


n_samples, z_dim = 1000, 100

for name in model_names:
    #os.makedirs(sample_dirs[name], exist_ok = True)  
    print("for model: {}".format(name))
    model = models[name]
    print("loading model from file...")
    model.load_state_dict(torch.load(model_files[name]))    
    print("sampling and save images using model ", name)
    epoch = trained_epochs[name]
    sample_and_save_images(model, n_samples, epoch = epoch, name = name)