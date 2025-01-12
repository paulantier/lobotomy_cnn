import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class PawpularityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        pawpularity_score = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return image, pawpularity_score