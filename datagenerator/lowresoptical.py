"""Create 256x256 patches of Sentinel-2 data for training"""
import os 
import re
import rasterio as rio 
from config import file_location

data_folders = os.listdir(file_location['sen2'])

for index, folder in enumerate(data_folders):
    if folder!='.DS_Store':
        img_granule = file_location['sen2'] + folder + '/GRANULE/'
        temp_file = os.listdir(img_granule)[1]
        img_granule= img_granule + temp_file + '/IMG_DATA/' + 'R10m/'
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
        

