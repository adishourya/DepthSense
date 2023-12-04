from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


class RedWebDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Root Dir is REDWEB_V1
            Imgs -> Monocular Images
            RDs -> Corresponding Response (Heatmap)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.monocular_folder = os.path.join(root_dir, 'Imgs')
        self.heatmap_folder = os.path.join(root_dir, 'RDs')
        # Monocular images are in jpgs and the heatmap images are in pngs
        self.monocular_images = sorted(os.listdir(self.monocular_folder))
        self.heatmap_images = sorted(os.listdir(self.heatmap_folder))

    def __len__(self):
        assert len(self.monocular_images) == len(self.heatmap_images), "Hein?"
        return len(self.monocular_images)

    def _show_sample(self, name=None):
        if name:
            name = name if "." not in name else name.spit(".")[0]
        else:
            # show random index
            rand_index = np.random.randint(
                low=0, high=len(self), size=1).item()
            name = self.monocular_images[rand_index].split(".")[0]

        mono_img = io.imread(self.monocular_folder + "/" + name + ".jpg")
        heat_img = io.imread(self.heatmap_folder + "/" + name + ".png")

        fig, ax = plt.subplot_mosaic("AB")
        ax["A"].set_title(f"Monocular Image{name}", fontsize=8)
        ax["A"].imshow(mono_img)
        ax["B"].set_title(f"Heatmap Image{name}", fontsize=8)
        ax["B"].imshow(heat_img, cmap="inferno")
        ax["A"].grid(False)
        ax["A"].axis('off')
        ax["B"].grid(False)
        ax["B"].axis('off')
        plt.show()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        monocular_image = read_image(
            self.monocular_folder+"/"+self.monocular_images[idx])
        heatmap_image = read_image(
            self.heatmap_folder + "/" + self.heatmap_images[idx])

        sample = {'mono': monocular_image, 'heat': heatmap_image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """
    simple Rescaler for my monocular and heatmap images
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        mono, heat = sample['mono'], sample['heat']
        _, h, w = mono.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        resize_transform = transforms.Resize((new_h, new_w))
        mono = resize_transform(mono)
        heat = resize_transform(heat)


        return {'mono': mono, 'heat': heat}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        mono, heat = sample['mono'], sample['heat']

        _, h, w = mono.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        mono = mono[top: top + new_h, left: left + new_w]
        heat = heat[top: top + new_h, left: left + new_w]

        return {'mono': mono, 'heat': heat}


# a simple image to tensor class
# most likely not going to use it now
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        mono, heat = sample['mono'], sample['heat']

        # for monocular image :
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(mono.shape , heat.shape)
        mono = mono.transpose((2, 0, 1))
        heat = heat.transpose((2, 0, 1))

        return {'mono': torch.from_numpy(mono),
                'heat': torch.from_numpy(heat)}
