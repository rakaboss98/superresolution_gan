from dataloader.load import LoadItem
from dataloader.config import data_location
from train_config import training_parameteres
from models import rcan, patchgan, residualgan
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else 'cpu'

train_data = LoadItem(data_location['high_res_train_data'], data_location['low_res_train_data'])

''' The default channels are set to 3 in all generators and discriminators '''

'Initialising the generators'
gen_yd2x = residualgan.ResGan().to(device)
gen_x2yd = rcan.RCAN().to(device)
upsample_net = rcan.RCAN().to(device)

'Initialising the discriminators'
disc_xd = patchgan.PatchGan(image_res='low').to(device)
disc_yd = patchgan.PatchGan(image_res='low').to(device)
disc_xup = patchgan.PatchGan(image_res='high').to(device)

'Initialising the optimizers'

# Calculate the computational graph for all the generators
tensor = torch.randn(4, 3, 256, 256)
gen_yd2x.forward(tensor)
gen_x2yd.rcan_out(tensor, upscale='False')
upsample_net.rcan_out(tensor, upscale='True', scaling_factor=2)

# Optimizer for generators
opti_gen = optim.Adam(
    params = list(gen_yd2x.parameters())+list(gen_x2yd.parameters()),
    lr = training_parameteres['learning_rate'],
    betas=[training_parameteres['beta_1'], training_parameteres['beta_2']],
    # eps= training_parameteres['epsilon']
)

# Optimizer for discriminators
opti_disc = optim.Adam(
    params = list(disc_xd.parameters())+list(disc_yd.parameters())+list(disc_xup.parameters()),
    lr = training_parameteres['learning_rate'],
    betas=[training_parameteres['beta_1'], training_parameteres['beta_2']],
    # eps= training_parameteres['epsilon']
)

# Optimizer for upsampling network
opti_upsamp_net = optim.Adam(
    params = list(upsample_net.parameters()),
    lr = training_parameteres['learning_rate'],
    betas=[training_parameteres['beta_1'], training_parameteres['beta_2']],
    # eps= training_parameteres['epsilon']
)

for epoch in range(training_parameteres['epochs']):
    
    batch = DataLoader(train_data, shuffle=True, batch_size=training_parameteres['batch_size'])

    print("Strating epoch number {}".format(epoch+1))


    try: 
        for idx, sample in enumerate(batch):
            'calculating discriminator loss'
            real_x = sample[0].to(device) # low resolution image (source)
            real_y = sample[1].to(device) # high resolution image (target)
            print(real_x.shape, real_y.shape)
            real_yd = F.interpolate(real_y, size=(256,256), mode='bicubic') # downsample the high resolution image
            print(real_yd.shape)
            fake_x = gen_yd2x(real_yd)
            fake_x.detach()
            
            disc_xd_loss = torch.log(disc_xd(real_x))+torch.log(1-disc_xd(fake_x))

            print(disc_xd_loss.shape)
    except:
        pass

    'backpropogation and updating the weights'

