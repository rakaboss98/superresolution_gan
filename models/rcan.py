import torch 
import torch.nn as nn 
import torch.nn.functional as F

class RCAB(nn.Module):
    def __init__(self, in_channels):
        super(RCAB, self).__init__()
        self.in_channels = in_channels
    
    def initial_conv(self, im_height, im_width, tensor):
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        )
        return conv_layers(tensor)

    def channel_attention(self, im_height, im_width, tensor):
        attention_conv_layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=(im_height, im_width)),
            nn.Conv2d(in_channels=self.in_channels, kernel_size=1, out_channels=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, kernel_size=1, out_channels=self.in_channels),
            nn.Sigmoid()
        )
        return attention_conv_layers(tensor)
    
    def block_output(self, tensor):
        _, in_height, in_width = tensor.shape
        conv_tensor = self.initial_conv(in_height, in_width, tensor) # passing throught the initial conv layer
        _, in_height, in_width = conv_tensor.shape
        scaling_stat = self.channel_attention(in_height, in_width, conv_tensor) # passing through channel attention layer
        conv_tensor = scaling_stat*conv_tensor # scaling the conv tensor
        tensor += conv_tensor # adding the initial input tensor
        return tensor 


class ResidualGroup():
    def __init__(self, num_blocks, in_channels):
        self.num_blocks = num_blocks
        self.blocks = []
        self.in_channels = in_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels= self.in_channels, kernel_size=3, padding=1)
        )
        for i in range(self.num_blocks):
            temp_rcab = RCAB(self.in_channels)
            self.blocks.append(temp_rcab)
    
    def group_out(self, tensor):
        temp_tensor = tensor
        for i in range(self.num_blocks):
            temp_tensor = self.blocks[i].block_output(temp_tensor)
        
        temp_tensor = self.conv_block(temp_tensor)
        tensor = temp_tensor+tensor
        return tensor

class RCAN():
    def __init__(self, in_channels, num_groups):
        self.in_channels = in_channels
        self.conv_start = nn.Conv2d(in_channels=in_channels, kernel_size=3, padding=1, out_channels=in_channels)
        self.conv_mid = nn.Conv2d(in_channels=in_channels, kernel_size=3, padding=1, out_channels=in_channels)
        self.conv_end = nn.Conv2d(in_channels=in_channels, kernel_size=3, padding=1, out_channels=3)
        self.num_groups = num_groups
        self.groups = []
        for i in range(self.num_groups):
            temp_group = ResidualGroup(num_blocks=10, in_channels=self.in_channels)
            self.groups.append(temp_group)
    
    def rcan_out(self, tensor):
        temp_tensor = tensor
        temp_tensor = self.conv_start(temp_tensor)
        for i in range(self.num_groups):
            temp_tensor = self.groups[i].group_out(temp_tensor)
        
        temp_tensor = self.conv_mid(temp_tensor)
        tensor = tensor + temp_tensor
        tensor = self.conv_end(tensor)
        return tensor

'Unit testing for different Classes'
if __name__ == '__main__':
    'Unit testing of the Residual Channel Attention Block'
    in_tensor = torch.rand(64,256,256)
    num_channels, _, _ = in_tensor.shape
    rcab = RCAB(in_channels=num_channels)
    out_tensor = rcab.block_output(in_tensor)
    print("The shape of the output tensor from rcab is {}".format(out_tensor.shape))

    'Unit testing of the Residual Group'
    in_tensor = torch.rand(64,256,256)
    num_channels, _, _ = in_tensor.shape
    residual_group = ResidualGroup(in_channels=num_channels, num_blocks=10)
    out_tensor = residual_group.group_out(in_tensor)
    print("The shape of the output tensor from rcab is {}".format(out_tensor.shape))

    'Unit testing of the Residual Channel attention network (rcan)'
    in_tensor = torch.rand(64, 256, 256)
    num_channels, _, _ = in_tensor.shape
    rcan = RCAN(in_channels=num_channels, num_groups=5)
    out_tensor = rcan.rcan_out(in_tensor)
    print("The shape of the output tensor from rcab is {}".format(out_tensor.shape))