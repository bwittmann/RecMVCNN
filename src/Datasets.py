import json

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import read_as_3d_array


class ShapeNetDataset(Dataset):

    def __init__(self, rendering_dir, voxel_dir, metadata_path, split='train'):
        assert split in ['train', 'val', 'test']

        self.voxel_dir = voxel_dir
        self.rendering_dir = rendering_dir
        self.taxonomy_ids = []
        self.data_ids = []
        
        with open(metadata_path) as json_file:
            metadata = json.load(json_file)
            for i in metadata:
                self.taxonomy_ids.append(i['taxonomy_id'])
                for j in i[split]:
                    self.data_ids.append(i['taxonomy_id'] + '/' + j)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        shapenet_id = self.data_ids[idx]
        renderings_path = os.path.join(self.rendering_dir, shapenet_id, 'rendering')
        png_files = []

        with open(renderings_path + '/renderings.txt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                png_files.append(line.strip())

        renderings = None

        # Load images into 4D tensors
        for i in png_files:
            
            image = read_image(renderings_path + '/' + i)
            if renderings is None:
                renderings = image.unsqueeze(0)
            else:
                renderings = torch.cat((renderings, image.unsqueeze(0)), 0)

        # Set class label
        class_label = shapenet_id.split('/')[0]       
        
        # Load voxel
        voxel = None
        with open(self.voxel_dir + '/' + shapenet_id + '/model.binvox', "rb") as fptr:
            voxel = read_as_3d_array(fptr).astype(np.float32)

        return renderings, class_label, voxel


if __name__ == '__main__':
    dataset = ShapeNetDataset('/media/andrew/Storage HDD/data/ShapeNet/ShapeNetRendering', '/media/andrew/Storage HDD/data/ShapeNet/ShapeNetVox32', 'data/ShapeNet.json')
    renderings, class_label, voxel = dataset[0]

    npimg = renderings[0, :3, :, :].numpy()

    
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')