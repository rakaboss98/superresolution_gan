"""Create 256x256 patches of airbus plaedis data"""
import os 
import re 
import rasterio as rio 
import itertools
import numpy as np
from config import file_location
import imageio as io
import matplotlib.pyplot as plt

data_location = file_location['airbus']
folders = os.listdir(data_location)
 
for data_folder in folders:
    if data_folder!='.DS_Store':
        base_name = data_folder
        bands = os.listdir(data_location+data_folder)
        for band in bands:
            if band!='.DS_Store' and re.search('MS', band):
                ms_band_location = data_location + data_folder + '/' + band
                images = os.listdir(ms_band_location)
                for image in images:
                    if image!='.DS_Store' and re.search('.tif', image):
                        if re.search('Preview.tif', image):
                            pass
                        else:
                            ms_band_location+='/'+image
        # print(ms_band_location)
        data = rio.open(ms_band_location)
        red = data.read(3)
        red = (red-red.min())/(red.max()-red.min())
        green = data.read(2)
        green = (green-green.min())/(green.max()-green.min())
        blue = data.read(1)
        blue = (blue-blue.min())/(blue.max()-blue.min())
        img = np.dstack((red, green, blue))
        height, width, channels = img.shape

        buffer_y = int(height*0.2)
        buffer_x = int(width*0.2)

        img = img[buffer_x:width-buffer_x, buffer_y:height-buffer_y]
        img = img.astype(np.float32)

        tile_size = 256
        tiling_stride = 128

        height, width, channels = img.shape # Update the height, width and channels

        # # counting the number of tiles that can be generated with the given stride 
        num_x_tiles = int((width-tile_size)/tiling_stride)+1
        num_y_tiles = int((height-tile_size)/tiling_stride)+1

        # # loop through the data and generate tiles
        for pointer_x, pointer_y in itertools.product(range(num_x_tiles), range(num_y_tiles)):
            origin_x, origin_y = 0+pointer_x*tiling_stride, 0+pointer_y*tiling_stride
            cropped_image = img[origin_x:origin_x+tile_size,origin_y:origin_y+tile_size]
            io.volsave(file_location['high_res_optical']+base_name+'__'+str(pointer_x)+str(pointer_y)+'.tif', cropped_image)











