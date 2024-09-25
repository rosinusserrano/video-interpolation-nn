import os
from random import choice

import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageSequence


class MovingGIFLinearInterpolationDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.root_dir, self.files[idx]),
                        formats=["gif"]) as gif:
            frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
            frames = [transforms.ToTensor()(frame) for frame in frames[1:]]
            frames = [(frame * -1) + 1
                      for frame in frames]  # invert to black bg

            random_idx = np.random.randint(0, len(frames), 2)
            first_idx, last_idx = np.sort(random_idx)
            next_idx = first_idx + 1 if last_idx > first_idx else choice(
                [first_idx, last_idx])

            # linear interpolation
            p = (next_idx - first_idx) / (last_idx - first_idx)
            if np.isnan(p):
                p = 0
            p = torch.tensor(p).float()

            x = frames[first_idx]
            y = frames[next_idx]
            z = frames[last_idx]
            return (x, z, p), y
