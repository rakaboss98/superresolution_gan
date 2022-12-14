from dataloader.load import LoadItem
from train_config import training_parameteres
from models import rcan, patchgan, residualgan
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
from utils import save_checkpoint, load_checkpoint, downloadDataFroms3, downloadCheckpointsFroms3, uploadCheckpointsTos3, downloadSamples, uploadSamples
import imageio as io 
from torchvision.transforms import transforms

'Read saving and loading tags from training_config'

save_model = training_parameteres['save_model']
load_model = training_parameteres['load_model']

'Specify the device for computation'

device = "cuda" if torch.cuda.is_available() else 'cpu'

try: 
    train_data = LoadItem(training_parameteres['high_res_optical'], training_parameteres['low_res_optical'])
    loader = DataLoader(train_data, shuffle=True, batch_size=training_parameteres['batch_size'])
    print('no data download happening from s3')
except:
    downloadDataFroms3()
    train_data = LoadItem(training_parameteres['high_res_optical'], training_parameteres['low_res_optical'])
    print('data download happening from s3')

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

'Load any existing checkpoints'

if load_model:
    downloadCheckpointsFroms3('gen_yd2x.pt')
    downloadCheckpointsFroms3('gen_x2yd.pt')
    downloadCheckpointsFroms3('upsample_net.pt')
    downloadCheckpointsFroms3('disc_xd.pt')
    downloadCheckpointsFroms3('disc_yd.pt')
    downloadCheckpointsFroms3('disc_xup.pt')
    load_checkpoint('checkpoints/gen_yd2x.pt', gen_yd2x, optimizer=opti_gen, device=device)
    load_checkpoint('checkpoints/gen_x2yd.pt', gen_x2yd,optimizer=opti_gen, device=device)
    load_checkpoint('checkpoints/upsample_net.pt', upsample_net, optimizer=opti_gen, device=device)
    load_checkpoint('checkpoints/disc_xd.pt', disc_xd, optimizer=opti_disc,device=device)
    load_checkpoint('checkpoints/disc_yd.pt', disc_yd, optimizer=opti_disc,device=device)
    load_checkpoint('checkpoints/disc_xup.pt', disc_xup, optimizer=opti_disc,device=device)


for epoch in range(training_parameteres['epochs']):
    
    loader = DataLoader(train_data, shuffle=True, batch_size=training_parameteres['batch_size'])

    genLoss = 0
    discLoss = 0 

    print("=========Starting epoch number========={}".format(epoch+1))


    for idx, batch in enumerate(loader):

        try: 
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

                discLoss = discLoss+disc_loss

            print('backpropagated for this batch', 'disc loss is {}'.format(disc_loss))

            with torch.autograd.set_detect_anomaly(True):
                'write script for training generator and saving model'
                gen_yd2x_loss = torch.log(1-disc_xd.forward(fake_x)) # generator y2xd adversarial loss
                gen_x2yd_loss = torch.log(1-disc_yd.forward(real_xd)) # generator x2yd adversarial loss
                gen_yd2x_x2yd_loss = torch.log(1-disc_xup.forward(fake_yup)) # adversarial loss for both generators
                cycle_loss = torch.abs(fake_yd-real_yd) # cycle consistency loss
                idty_loss = torch.abs(gen_x2yd.forward(real_yd)-real_yd) # Identity loss
                sr_loss = torch.abs(fake_yup-real_y) # super resolution loss

                gen_loss = torch.mean(gen_yd2x_loss) + torch.mean(gen_x2yd_loss) + training_parameteres['lambda_up']*torch.mean(gen_yd2x_x2yd_loss) + \
                training_parameteres['lambda_cyc']*torch.mean(cycle_loss) + training_parameteres['lambda_idt']*torch.mean(idty_loss) + torch.mean(sr_loss)

                gen_loss.backward()
                opti_gen.zero_grad()
                opti_gen.step()

                genLoss = genLoss + gen_loss
            break
            print('backpropagated for this batch', 'gen loss is {}'.format(gen_loss))
        except:
            pass
    if save_model==True:
        save_checkpoint(gen_yd2x, optimizer=opti_gen,filename='checkpoints/gen_yd2x.pt')
        save_checkpoint(gen_x2yd, optimizer=opti_gen,filename='checkpoints/gen_x2yd.pt')
        save_checkpoint(upsample_net, optimizer=opti_gen,filename='checkpoints/upsample_net.pt')
        save_checkpoint(disc_xd, optimizer=opti_disc,filename='checkpoints/disc_xd.pt')
        save_checkpoint(disc_yd, optimizer=opti_disc,filename='checkpoints/disc_yd.pt')
        save_checkpoint(disc_xup, optimizer=opti_disc,filename='checkpoints/disc_xup.pt')
        uploadCheckpointsTos3('gen_yd2x')
        uploadCheckpointsTos3('gen_x2yd')
        uploadCheckpointsTos3('upsample_net')
        uploadCheckpointsTos3('disc_xd')
        uploadCheckpointsTos3('disc_yd')
        uploadCheckpointsTos3('disc_xup')
    
    print('====The cumulative generator loss after epoch {} is {}===='.format(epoch, genLoss))
    print('====The cumulative discriminator loss after epoch {} is {}===='.format(epoch, discLoss))

    'check the results on an image after every epoch'
    downloadSamples('raw.tif')
    image = io.imread('samples/raw.tif')
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(dim=0)
    image = gen_x2yd(image)
    image = upsample_net(image)
    image = image.squeeze()
    io.volsave('samples/'+'_'+str(epoch)+'.tif', image.detach().numpy())
    uploadSamples('_'+str(epoch)+'.tif')







    



    

            




