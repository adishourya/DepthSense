import os
import torch


class ImgDataset:
    """
    Load RedWeb Dataset
    The  file struncture is :
        The input data comes in a pair
        data/RedWeb_V1/Imgs -> RGB Monocular Image
        data/RedWeb_V1/RDs -> Depth Map
    """

    def __init__(self, rgb_path: str, depth_map: str):
        pass

    def parse_pairs(self):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        pass
        # return len(self.pairs)
