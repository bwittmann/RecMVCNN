import argparse
import torch
import numpy as np

from torch.utils.data import  DataLoader
from dotenv import dotenv_values

from train import train
from mvcnn import MVCNN
from datasets import ShapeNetDataset


def main(args):
    # Get env variables
    env_vars = dotenv_values('.env')

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get model
    model = get_model(args)

    # Get data loaders
    train_dataloader = get_dataloader(args, env_vars, 'overfit') # TODO: dont hardcode split
    val_dataloader = get_dataloader(args, env_vars, 'val')

    # Train
    train(device, model, args, train_dataloader, val_dataloader)


def get_dataloader(args, env_vars, split):
    if args.dataset == 'scannet':
        dataset = dataset = ShapeNetDataset(
            env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], 'data/ShapeNet.json', split
            )
    else:
        raise NotImplementedError
            
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

def get_model(args):
    return MVCNN(args.mvcnn_num_classes, args.mvcnn_num_views)


if __name__ == "__main__":
    # Init parser to receive arguments from the terminal
    parser = argparse.ArgumentParser()

    # Standard arguments
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--seed", type=int, help="random seed", default=42)

    # Arguments related training
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--lr_decay_factor", type=float, help="decay factor of the lr scheduler", default=0.1)
    parser.add_argument("--lr_decay_patience", type=float, help="patience of the lr scheduler", default=10)
    parser.add_argument("--lr_decay_cooldown", type=float, help="cooldown of the lr scheduler", default=0)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)

    # Arguments related to MVCNN model
    parser.add_argument("--mvcnn_num_classes", type=int, help="number of classes", default=40)
    parser.add_argument("--mvcnn_num_views", type=int, help="number of views per object", default=12)

    # Arguments related to datasets
    # TODO: add more choices
    parser.add_argument("--dataset", type=str, choices=['scannet'], help="used dataset", default='scannet')

    args = parser.parse_args()

    # For reproducability # TODO: check functionalilty
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    main(args)
