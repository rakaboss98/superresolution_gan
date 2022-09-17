from email.mime import image
import torch
from residualgan import ResGan
from rcan import RCAN
from patchgan import PatchGan

"""Collective testing of all the function calls from this package"""

if __name__ == '__main__':
    tensor = torch.randn(4, 3, 256, 256)

    # Testing residual gan architecture
    resgan = ResGan()
    resgan.forward(tensor)
    print('ResGan call successful')

    # Testing rcan architecture
    rcan = RCAN(in_channels=tensor.shape[1], num_groups=5)
    rcan.rcan_out(tensor, upscale='True', scaling_factor=2)
    print('Rcan call sucessful')

    # Testing the patchgan architecture
    pgan = PatchGan(tensor, image_res='high')
    pgan.forward()
    print('PatchGan call successful')
