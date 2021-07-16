from logging import raiseExceptions
import sys
sys.path.append('./src')

from datasets import ShapeNetDataset
from utils import env_vars
import numpy as np
import random


if __name__ == '__main__':
    train_dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_PC_RENDERING_DATASET_PATH'], 'train', pointcloud_renderings=True)
    val_dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_PC_RENDERING_DATASET_PATH'], 'val', pointcloud_renderings=True)
    test_dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_PC_RENDERING_DATASET_PATH'], 'test', pointcloud_renderings=True)

    valid = []
    invalid = []
    for i in range(len(train_dataset)):
        try:
            shapenet_id, renderings, class_label, voxel = train_dataset.__getitem__(i)
            if (list(renderings.shape) == [3,3,137,137]) and (type(voxel) is np.ndarray):
                valid.append(shapenet_id)
            else:
                invalid.append(shapenet_id)
        except:
            invalid.append(train_dataset.data_ids[i])

    for i in range(len(val_dataset)):
        try:
            shapenet_id, renderings, class_label, voxel = val_dataset.__getitem__(i)
            if (list(renderings.shape) == [3,3,137,137]) and (type(voxel) is np.ndarray):
                valid.append(shapenet_id)
            else:
                invalid.append(shapenet_id)
        except:
            invalid.append(val_dataset.data_ids[i])

    for i in range(len(test_dataset)):
        try:
            shapenet_id, renderings, class_label, voxel = test_dataset.__getitem__(i)
            if (list(renderings.shape) == [3,3,137,137]) and (type(voxel) is np.ndarray):
                valid.append(shapenet_id)
            else:
                invalid.append(shapenet_id)
        except:
            invalid.append(test_dataset.data_ids[i])

    random.shuffle(valid)
    train = valid[:int(len(valid)*.7)]
    valid = valid[int(len(valid)*.7):]
    val = valid[int(len(valid)*.333):]
    test = valid[:int(len(valid)*.333)]


    with open("data/shapenet_pc/train.txt", "w") as outfile:
        outfile.write("\n".join(train))

    with open("data/shapenet_pc/val.txt", "w") as outfile:
        outfile.write("\n".join(val))

    with open("data/shapenet_pc/test.txt", "w") as outfile:
        outfile.write("\n".join(test))

    print("invalid len: " +  str(len(invalid)))
    print("train/val/test lens:" + str(len(train)) + "/" + str(len(val)) + "/" + str(len(test)))