from statistics import mean, variance
import torch
import torch.nn as nn 
import torch.nn.functional as F

class ResGan(nn.Module):
    def __init__(self):
        super(ResGan, self).__init__()
    
    # Combination of a residual block and a gaussian layer
    def input_block(self, tensor):
        batch_size, channel_in, height, width = tensor.shape
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        noise_layer = torch.randn(batch_size, 1, height, width)
        tensor = self.conv_layer_1(tensor)
        tensor = torch.cat(tensors=(tensor, noise_layer), dim=1)
        return tensor 
    
    # Creating a residual block with skip connections
    def residual_block(self, tensor):
        _, channel_in, _, _ = tensor.shape
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=channel_in, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(channel_in)
        )
        tensor = self.conv_layer_2(tensor) + tensor
        tensor = F.leaky_relu(tensor, negative_slope=0.01)
        return tensor

    # Creating a fusion block to fuse features from residual and parallel cnn layers
    def fusion_block(self, tensor_1, tensor_2):
        _, channel_in_1, _, _ = tensor_1.shape
        _, channel_in_2, _, _ = tensor_2.shape
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in_1, out_channels=channel_in_1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(channel_in_1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        tensor_1 = self.conv_layer_3(tensor_1)
        out_tensor = torch.cat(tensors=(tensor_1, tensor_2), dim=1)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(in_channels=channel_in_2+channel_in_1, out_channels=channel_in_1, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_in_1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        out_tensor = self.fuse_layer(out_tensor)
        return(out_tensor)
    
    # Combining all the layers
    def forward(self, tensor):
        _, channel_in, _, _ = tensor.shape
        tensor = self.input_block(tensor)
        tensor_2 = self.residual_block(tensor)
        tensor_1 = self.fusion_block(tensor, tensor_2)
        tensor_2 = self.residual_block(tensor_1)
        tensor_1 = self.fusion_block(tensor_1, tensor_2)
        tensor_2 = self.residual_block(tensor_1)
        tensor_1 = self.fusion_block(tensor_1, tensor_2)

        _, channel_in_1, _, _ = tensor_1.shape
        self.final_out = nn.Sequential(
            nn.Conv2d(in_channels=channel_in_1, out_channels=channel_in, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
        tensor_1 = self.final_out(tensor_1)
        return tensor_1
        
if __name__ == '__main__':

    'testing the residual block'
    tensor = torch.rand(1, 64, 256, 256)
    resgan = ResGan()
    print('The output of residual gan is {}'.format(resgan.residual_block(tensor).shape))

    'testing the fusion block'
    tensor_1 = torch.rand(1, 3, 256, 256)
    tensor_2 = torch.rand(1, 3, 256, 256)
    print('The output of the fusion block is {}'.format(resgan.fusion_block(tensor_1, tensor_2).shape))

    'testing the forward block'
    tensor = torch.rand(2, 8, 256, 256)
    print('The output of the forward block is {}'.format(resgan.forward(tensor).shape))

    'testing for model parameters'
    print('the model summary is', resgan)

    
    

