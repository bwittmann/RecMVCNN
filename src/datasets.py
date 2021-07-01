import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import cv2
import numpy as np
from random import randrange

from utils import read_as_3d_array, env_vars


class ShapeNetDataset(Dataset):

    def __init__(self, rendering_dir, voxel_dir, split='train', num_views=24):
        assert split in ['train', 'val', 'test', 'overfit']
        assert 1 <= num_views <= 24, "num_views must be between 1 and 24"

        self.voxel_dir = voxel_dir
        self.rendering_dir = rendering_dir
        self.data_ids = []
        self.num_views = num_views

        with open(f'{env_vars["PROJECT_DIR_PATH"]}/data/shapenet_info.json') as json_file:
            self.class_name_mapping = json.load(json_file)

        self.classes = sorted(self.class_name_mapping.keys())

        with open(f'{env_vars["PROJECT_DIR_PATH"]}/data/{split}.txt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[-1] == '\n':
                    line = line[:-1]
                self.data_ids.append(line)

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
        selected_png_files = [png_files[i] for i in (np.linspace(0, 23, self.num_views, dtype=np.int_) + randrange(24)) % 24]
        # Load images into 4D tensors
        for i in selected_png_files:
            image = cv2.imread(renderings_path + '/' + i)
            image = torch.from_numpy(image).permute(2,1,0)
            if renderings is None:
                renderings = image.unsqueeze(0)
            else:
                renderings = torch.cat((renderings, image.unsqueeze(0)), 0)

        # Set class label
        class_label = self.classes.index(shapenet_id.split('/')[0])
        
        # Load voxel
        voxel = None
        with open(self.voxel_dir + '/' + shapenet_id + '/model.binvox', "rb") as fptr:
            voxel = read_as_3d_array(fptr).astype(np.float32)

        return shapenet_id, renderings, class_label, voxel


if __name__ == '__main__':
    dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'])
    dataloader = DataLoader(dataset)
    shapenet_id, renderings, class_label, voxel = next(iter(dataloader))
    tensor_image = renderings[0, 0, :, :, :]

    cv2.imshow('Test image',tensor_image.permute(2,1,0).numpy())
    cv2.waitKey(0)