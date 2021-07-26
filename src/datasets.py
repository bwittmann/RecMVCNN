"""Dataset for the different ShapeNet versions."""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import cv2
import numpy as np
from random import randrange
from utils import read_as_3d_array, env_vars


class ShapeNetDataset(Dataset):
    """Dataset of ShapeNet.

    This class can be used for different multi-view representations of the ShapeNet dataset 
    like incomplete point clouds, point clouds, or colored meshed.
    """
    def __init__(self, voxel_dir, rendering_dir, split='train', num_views=24, pointcloud_renderings=False):
        """Initialization method.
        
        Args:
            voxel_dir: absolute path to ground truth voxel grid
            rendering_dir: absolute path to rendered multi-view images
            split: either train, val, test, or overfit
            num_views: int corresponding to the number of utilized multi-view images
            pointcloud_renderings: boolean indicating if point cloud renderings 
                should be used
        """
        assert split in ['train', 'val', 'test', 'overfit']
        assert 1 <= num_views <= 24, "num_views must be between 1 and 24"

        self.voxel_dir = voxel_dir
        self.rendering_dir = rendering_dir
        self.data_ids = []
        self.num_views = num_views
        self.pointcloud_renderings = pointcloud_renderings

        with open(f'{env_vars["PROJECT_DIR_PATH"]}/data/shapenet_info.json') as json_file:
            self.class_name_mapping = json.load(json_file)

        self.classes = sorted(self.class_name_mapping.keys())

        self.split = split
        split_path = f'{env_vars["PROJECT_DIR_PATH"]}/data/{split}.txt'
        split_path = split_path if not pointcloud_renderings else f'{env_vars["PROJECT_DIR_PATH"]}/data/shapenet_pc/{split}.txt'
        with open(split_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[-1] == '\n':
                    line = line[:-1]
                self.data_ids.append(line)

    def __len__(self):
        """Returns size of dataset."""
        return len(self.data_ids)

    def __getitem__(self, idx):
        """Returns sample of dataset corresponding to idx."""
        shapenet_id = self.data_ids[idx]
        renderings_path = os.path.join(self.rendering_dir, shapenet_id, 'rendering')
        png_files = []

        renderings = None

        if self.pointcloud_renderings == True:
            imgs = ["00.png", "01.png", "02.png"]
            for i in imgs:
                image = cv2.imread(renderings_path + '/' + i)
                image = torch.from_numpy(image).permute(2,1,0)
                if renderings is None:
                    renderings = image.unsqueeze(0)
                else:
                    renderings = torch.cat((renderings, image.unsqueeze(0)), 0)
        else:
            with open(renderings_path + '/renderings.txt') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    png_files.append(line.strip())
            if self.split == "test":
                selected_png_files = [png_files[i] for i in np.linspace(0, 23, self.num_views, dtype=np.int_)]
            else:
                selected_png_files = [png_files[i] for i in (np.linspace(0, 23, self.num_views, dtype=np.int_) + randrange(24)) % 24]
            # Load images into 4D tensors
            for i in selected_png_files:
                try:
                    image = cv2.imread(renderings_path + '/' + i)
                    image = torch.from_numpy(image).permute(2,1,0)
                except:
                    print('Rendering could not be found in the dataset.')
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
    dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_PC_RENDERING_DATASET_PATH'], 'val', pointcloud_renderings=True)
    dataloader = DataLoader(dataset)
    shapenet_id, renderings, class_label, voxel = next(iter(dataloader))
    tensor_image = renderings[0, 0, :, :, :]

    print(len(dataset))
    # cv2.imshow('Test image',tensor_image.permute(2,1,0).numpy())
    # cv2.waitKey(0)