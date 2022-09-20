"""Create 256x256 patches of Sentinel-2 data for training"""
import os 
import re
import rasterio as rio 
from config import file_location
import numpy as np
import matplotlib.pyplot as plt
import imageio as io 
import itertools

data_folders = os.listdir(file_location['sen2'])

tile_size = 256
tiling_stride = int(tile_size/2)

def norm(band):
    band = band.read(1).astype(np.float32)
    band = (band-band.min())/(band.max()-band.min())
    return band

for index, folder in enumerate(data_folders):
    if folder!='.DS_Store':
        img_granule = file_location['sen2'] + folder + '/GRANULE/'
        granule_name = os.listdir(img_granule)[1]
        img_granule= img_granule + granule_name + '/IMG_DATA/' + 'R10m/'
        bands = os.listdir(img_granule)
        for band in bands:
            if re.search('_B02_10m.jp2',band):
                blue_band = img_granule+band 
            if re.search('_B03_10m.jp2', band):
                green_band = img_granule+band
            if re.search('_B04_10m.jp2', band):
                red_band = img_granule+band
        red = rio.open(red_band)
        blue = rio.open(blue_band)
        green = rio.open(green_band)
        buffer_x = (red.bounds.right-red.bounds.left)*0.2
        buffer_y = (red.bounds.top-red.bounds.bottom)*0.2
        x_left, y_top = red.index(red.bounds.left +buffer_x, red.bounds.top-buffer_y)
        x_right, y_bott = red.index(red.bounds.right-buffer_x, red.bounds.bottom+buffer_y)

        red = norm(red)[x_left:x_right, y_top:y_bott]
        blue = norm(blue)[x_left:x_right, y_top:y_bott]
        green = norm(green)[x_left:x_right, y_top:y_bott]
        img = np.dstack((red, green, blue))

        height, width, channel = img.shape

        num_x_tiles = int((width-tile_size)/tiling_stride)+1
        num_y_tiles = int((height-tile_size)/tiling_stride)+1

        for pointer_x, pointer_y in itertools.product(range(num_x_tiles), range(num_y_tiles)):
            origin_x, origin_y = 0+pointer_x*tiling_stride, 0+pointer_y*tiling_stride
            cropped_image = img[origin_x:origin_x+tile_size,origin_y:origin_y+tile_size]
            io.volsave(file_location['low_res_optical']+granule_name+'__'+str(pointer_x)+str(pointer_y)+'.tif', cropped_image)
