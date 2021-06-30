import json
import os
from sys import meta_path
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import cv2
import numpy as np
import random

from utils import read_as_3d_array, dotenv_values
from dotenv import load_dotenv

class ShapeNetDataset(Dataset):
    def __init__(self, voxel_dir, rendering_dir, split='train', num_views=24., project_path=None):
        assert split in ['train', 'val', 'test', 'overfit']
        assert 1 <= num_views <= 24, "num_views must be between 1 and 24"

        self.voxel_dir = voxel_dir
        self.rendering_dir = rendering_dir
        self.data_ids = []
        self.num_views = num_views
        
        # Needed for raytune
        if project_path:
            metadata_path = project_path + "/data"
        else:
            metadata_path = "data"

        with open(f'{metadata_path}/shapenet_info.json') as json_file:
            self.class_name_mapping = json.load(json_file)

        self.classes = sorted(self.class_name_mapping.keys())

        if split == 'overfit':
            with open(f'{metadata_path}/overfit.txt') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line[-1] == '\n':
                        line = line[:-1]
                    self.data_ids.append(line)
        else:
            with open(f'{metadata_path}/ShapeNet.json') as json_file:
                metadata = json.load(json_file)
                for i in metadata:
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
        selected_png_files = random.sample(png_files, k=self.num_views)

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
    dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], 'test')
    dataloader = DataLoader(dataset)
    shapenet_id, renderings, class_label, voxel = next(iter(dataloader))
    tensor_image = renderings[0, 0, :, :, :]

    cv2.imshow('Test image',tensor_image.permute(2,1,0).numpy())
    cv2.waitKey(0)

    # j = 0
    # try:
    #     for i in dataloader:
    #         j += 1
    # except Exception as e:
    #     print(e)
    #     print("index:", j)
