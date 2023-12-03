from __future__ import print_function, division
from data.loaders.DataLoader import RedWebDataset, Rescale, RandomCrop, ToTensor
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class OnlineRedWeb(RedWebDataset):
    """
    By default it would not make online samples
    """
    def __init__(self, root_dir, transform=None,
                 N: int =0, sigma: int = 0.02):
        super().__init__(root_dir, transform)
        self.N = N
        self.sigma = sigma

    def __getitem__(self, index):
        # Call the __getitem__ method of the original dataset
        original_item = super(OnlineRedWeb, self).__getitem__(index)

        # Extract necessary information from the original_item
        mono = original_item['mono']
        heat = original_item['heat']
        print(self.N)
        if self.N !=0:
            # Implement online sampling logic here
            _ , height, width = heat.shape
            point_a = []
            point_b = []
            labels = []

            for _ in range(self.N):
                i, j = random.randint(0, height-1), random.randint(0, width-1)
                k, l = random.randint(0, height-1), random.randint(0, width-1)

                ga, gb = heat[i, j], heat[k, l]  # Assuming heat is a 2D tensor

                if ga / gb > 1 + self.sigma:
                    label = 1
                elif ga / gb < 1 - self.sigma:
                    label = -1
                else:
                    label = 0

                point_a.append((i, j))
                point_b.append((k, l))
                labels.append(label)

            # Update the original_item or create a new dictionary to return
            online_item = {'mono': mono,
                           'heat': heat,
                           'point_a': torch.tensor(point_a), "point_b": torch.tensor(point_b),
                           'labels': torch.tensor(labels)}

            online_item = {"mono":mono,
                           "heat":heat}

        return online_item


def online_collater(batch):
    mono = torch.stack([item['mono'] for item in batch])
    heat = torch.stack([item['heat'] for item in batch])
    point_a = [item['point_a'] for item in batch]
    point_b = [item['point_b'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'mono': mono, 'heat': heat,
            'point_a': point_a, "point_b": point_b,
            'labels': labels}
