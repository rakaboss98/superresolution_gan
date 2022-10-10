"""Create 256x256 patches of Sentinel-2 data for training"""
import os 
import re
import rasterio as rio 
from config import file_location
import numpy as np
import matplotlib.pyplot as plt
import imageio as io 
import itertools

tile_size = 256
tiling_stride = 128

'generate the data tiles from sentinel-2 images'
'currently supports the tiling of rgb channels'
def generate_data(data_location=file_location['sen2'],save_location=file_location['low_res_optical'], tile_size=256, tiling_stride=128):
    data_folders = os.listdir(data_location) # Identify all the sen-2 data folders at data_location
    # Normalise the band
    def norm(band):
        band = band.read(1).astype(np.float32)
        band = (band-band.min())/(band.max()-band.min())
        return band
    for index, folder in enumerate(data_folders): # loop in through each folder in data_location
        if folder!='.DS_Store':
            img_granule = data_location + folder + '/GRANULE/'
            for granule in os.listdir(img_granule):
                if granule != '.DS_Store':
                    granule_name = granule
            img_granule= img_granule + granule_name + '/IMG_DATA/' + 'R10m/'
            bands = os.listdir(img_granule)

            # identify the bandnames for rgb band
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
            # the buffer is to avoid the black stripes in the generated tiles
            # the buffer is set to 20% reduction from each side of the original image
            buffer_x = (red.bounds.right-red.bounds.left)*0.2
            buffer_y = (red.bounds.top-red.bounds.bottom)*0.2
            x_left, y_top = red.index(red.bounds.left +buffer_x, red.bounds.top-buffer_y)
            x_right, y_bott = red.index(red.bounds.right-buffer_x, red.bounds.bottom+buffer_y)

            red = norm(red)[x_left:x_right, y_top:y_bott]
            blue = norm(blue)[x_left:x_right, y_top:y_bott]
            green = norm(green)[x_left:x_right, y_top:y_bott]
            img = np.dstack((red, green, blue))

            height, width, channel = img.shape

            # counting the number of tiles that can be generated with the given stride 
            num_x_tiles = int((width-tile_size)/tiling_stride)+1
            num_y_tiles = int((height-tile_size)/tiling_stride)+1

            # loop through the data and generate tiles
            for pointer_x, pointer_y in itertools.product(range(num_x_tiles), range(num_y_tiles)):
                origin_x, origin_y = 0+pointer_x*tiling_stride, 0+pointer_y*tiling_stride
                cropped_image = img[origin_x:origin_x+tile_size,origin_y:origin_y+tile_size]
                io.volsave(save_location+granule_name+'__'+str(pointer_x)+str(pointer_y)+'.tif', cropped_image)

if __name__ == "__main__":
    generate_data()

