import os 
import imageio.v2 as io
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class LoadItem(Dataset):
    def __init__(self, root_dir_high_res, root_dir_low_res):
        self.root_dir_high_res = root_dir_high_res
        self.root_dir_low_res = root_dir_low_res
        self.transforms_up = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.6),
            transforms.Resize(size=(512, 512))
        ])
        self.transforms_down = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.6)
        ])

        self.high_res_files = []
        self.low_res_files = []

    def __len__(self):
        self.low_res_files = os.listdir(self.root_dir_low_res)
        try:
            self.low_res_files.remove('.DS_Store')
        except:
            pass

        self.high_res_files = os.listdir(self.root_dir_high_res)
        try:
            self.high_res_files.remove('.DS_Store')
        except:
            pass

        self.len_high, self.len_low = len(self.high_res_files), len(self.low_res_files)

        if self.len_high<=self.len_low:
            return self.len_high
        else:
            return self.len_low
    
    def __getitem__(self, idx):

        upsamp_image_name = self.high_res_files[idx]
        id_temp = random.randint(0, self.len_high)
        downsamp_image_name = self.low_res_files[id_temp]

        downsamp_image = io.imread(self.root_dir_low_res+downsamp_image_name)
        downsamp_image = self.transforms_down(downsamp_image)
        upsamp_image = io.imread(self.root_dir_high_res+upsamp_image_name)
        upsamp_image = self.transforms_up(upsamp_image)

        return [downsamp_image, upsamp_image]

if __name__ == "__main__":
    from config import data_location
    train_data = LoadItem(data_location['high_res_train_data'], data_location['low_res_train_data'])
    train_iterator = DataLoader(train_data, shuffle=True, batch_size=2)
    images = next(iter(train_iterator))
    print(images[0].shape, images[1].shape)