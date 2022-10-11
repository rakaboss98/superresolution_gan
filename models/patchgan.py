import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PatchGan(nn.Module):
    def __init__(self, in_channels = 3, image_res = 'low'):
        super(PatchGan, self).__init__()
        self.channel_in = in_channels
        self.image_res = image_res
        self.patch_gan_down = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_in, out_channels=64, stride=1, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=5 ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=16, stride=1, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=4, stride=1, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=4, out_channels=1, stride=1, kernel_size=5),
            nn.Sigmoid()
        )
        self.patch_gan_up = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_in, out_channels=64, stride=2, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=32, stride=2, kernel_size=5 ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=16, stride=1, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=4, stride=1, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=4, out_channels=1, stride=1, kernel_size=5),
            nn.Sigmoid()
        )
    def forward(self, tensor):
        if self.image_res=='low':
            return self.patch_gan_down(tensor)

        elif self.image_res == 'high':
            return self.patch_gan_up(tensor)
            
        else:
            raise Exception('Only two values possible for image_res, high and low')

if __name__ == "__main__":
    tensor = torch.randn(1, 8, 256, 256)
    _, in_channels, _, _ = tensor.shape
    disc = PatchGan(in_channels, image_res='low')
    print('The shape of tensor after low res discriminator is {}'.format(disc.forward(tensor).shape))

    disc = PatchGan(in_channels, image_res='high')
    print('The shape of tensor after low res discriminator is {}'.format(disc.forward(tensor).shape))