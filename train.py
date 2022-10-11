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
gen_x2yd.forward(tensor, upscale='False')
upsample_net.forward(tensor, upscale='True', scaling_factor=2)

# Optimizer for generators
opti_gen = optim.Adam(
    params = list(gen_yd2x.parameters())+list(gen_x2yd.parameters())+list(upsample_net.parameters()),
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

for epoch in range(training_parameteres['epochs']):
    
    loader = DataLoader(train_data, shuffle=True, batch_size=training_parameteres['batch_size'])

    print("Strating epoch number {}".format(epoch+1))


    for idx, batch in enumerate(loader):
        with torch.autograd.set_detect_anomaly(True):
            'calculating discriminator loss'
            real_x = batch[0].to(device) # low resolution image (source)
            real_y = batch[1].to(device) # high resolution image (target)
            real_yd = F.interpolate(real_y, size=(256,256), mode='bicubic') # downsample the high resolution image
            fake_x = gen_yd2x.forward(real_yd)
            fake_x
            
            disc_xd_loss = -torch.log(disc_xd.forward(real_x))-torch.log(1-disc_xd.forward(fake_x.detach()))
            real_xd = gen_x2yd.forward(real_x)
            fake_yd = gen_x2yd.forward(fake_x)

            disc_yd_loss = -torch.log(disc_yd.forward(real_yd))-torch.log(1-disc_yd.forward(real_xd.detach()))

            real_xup = upsample_net.forward(real_xd, upscale='True', scaling_factor=2)
            fake_yup = upsample_net.forward(fake_yd, upscale='True', scaling_factor=2)

            disc_xup_loss = -torch.log(disc_xup.forward(real_xup.detach()))-torch.log(1-disc_xup.forward(fake_yup.detach()))

            disc_loss = torch.mean(disc_xd_loss)+torch.mean(disc_yd_loss)+torch.mean(disc_xup_loss)
            opti_disc.zero_grad()
            disc_loss.backward()
            opti_disc.step()

        print('backpropagated for this batch', 'disc loss is {}'.format(disc_loss))

        with torch.autograd.set_detect_anomaly(True):
            'write script for training generator and saving model'
            gen_yd2x_loss = torch.log(1-disc_xd.forward(fake_x)) # generator y2xd adversarial loss
            gen_x2yd_loss = torch.log(1-disc_yd.forward(real_xd)) # generator x2yd adversarial loss
            gen_yd2x_x2yd_loss = torch.log(1-disc_xup.forward(fake_yup)) # adversarial loss for both generators
            cycle_loss = torch.abs(fake_yd-real_yd) # cycle consistency loss
            idty_loss = torch.abs(gen_x2yd.forward(real_yd)-real_yd) # Identity loss
            sr_loss = torch.abs(fake_yup-real_y) # super resolution loss

            gen_loss = torch.mean(gen_yd2x_loss) + torch.mean(gen_x2yd_loss) + torch.mean(gen_yd2x_x2yd_loss) + torch.mean(cycle_loss) + torch.mean(idty_loss) + torch.mean(sr_loss)

            gen_loss.backward()
            opti_gen.zero_grad()
            opti_gen.step()
        
        print('backpropagated for this batch', 'gen loss is {}'.format(gen_loss))
        break

            




