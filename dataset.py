import os

import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt


class MovingGIFInterpolationDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, skip_frames=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.skip_frames = skip_frames
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.root_dir, self.files[idx]),
                        formats=["gif"]) as gif:
            frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
            frames = [transforms.ToTensor()(frame) for frame in frames[1:]]
            frames = [(frame * -1) + 1 for frame in frames] # invert to black bg

            if len(frames) < 3 + self.skip_frames * 2:
                print(
                    f"{self.files[idx]} cant be used for skip_frames={self.skip_frames}"
                )
                return (frames[0], frames[0]), frames[0]

            random_idx = np.random.randint(
                0,
                len(frames) - (2 + self.skip_frames * 2))
            x = frames[random_idx]
            y = frames[random_idx + 1 + self.skip_frames]
            z = frames[random_idx + 2 * (1 + self.skip_frames)]
            return (x, z), y
