import argparse
import torch
import numpy as np
import torch.optim as optim

from torch.utils.data import  DataLoader
from dotenv import dotenv_values

from train import train
from test import test
from mvcnn_rec import ReconstructionMVCNN
from datasets import ShapeNetDataset


def main(args):
    # Get env variables
    env_vars = dotenv_values('.env')

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get model
    model = get_model(args)
    model.to(device)

    # Get optim
    backbone_params = []
    rec_params = []
    cls_params = []

    for param_name, param in model.named_parameters():
        print(param_name)
        if not param.requires_grad:
            continue
        if "decoder" in param_name:
            rec_params.append(param)
        elif "classifier" in param_name:
            cls_params.append(param)
        else:
            backbone_params.append(param)

    param_dicts = [
        {"params" : backbone_params},
        {"params" : rec_params,
         "lr" : args.lr_rec_head,
         "weight_decay" : args.wd_rec_head},
        {"params" : cls_params,
        "lr" : args.lr_cls_head,
        "weight_decay" : args.wd_cls_head},
    ]

    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.wd)

    # Get scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, cooldown=args.lr_decay_cooldown
    )

    if args.use_checkpoint:
        checkpoint = torch.load(args.use_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if args.debug:
        # Print number of model parameters
        params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        params_general = sum(param.numel() for param in model.parameters())
        backbone_params_trainable = sum(param.numel() for param in model.features.parameters() if param.requires_grad)
        backbone_params_general = sum(param.numel() for param in model.features.parameters())
        decoder_params_trainable = sum(param.numel() for param in model.decoder.parameters() if param.requires_grad)
        decoder_params_general = sum(param.numel() for param in model.decoder.parameters())
        classifier_params_trainable = sum(param.numel() for param in model.classifier.parameters() if param.requires_grad)
        classifier_params_general = sum(param.numel() for param in model.classifier.parameters())
        # TODO: implement
        #fusion_module_params_trainable = sum(param.numel() for param in model.fusion_module.parameters() if param.requires_grad)
        #fusion_module_params_general = sum(param.numel() for param in model.fusion_module.parameters())

        print('model parameters: [{:,}/{:,}]'.format(params_trainable, params_general))
        print('backbone parameters: [{:,}/{:,}]'.format(backbone_params_trainable, backbone_params_general))
        print('classifier parameters: [{:,}/{:,}]'.format(classifier_params_trainable, classifier_params_general))
        print('decoder parameters: [{:,}/{:,}]'.format(decoder_params_trainable, decoder_params_general))
        #print('fusion module parameters: [{:,}/{:,}]'.format(fusion_module_params_trainable, fusion_module_params_general))

    # Get data loaders
    if args.overfit:
        train_dataloader = get_dataloader(args, env_vars, 'overfit')
    else:
        train_dataloader = get_dataloader(args, env_vars, 'train')

    val_dataloader = get_dataloader(args, env_vars, 'val')
    test_dataloader = get_dataloader(args, env_vars, 'test')

    # TODO: Add test data split
    # Train
    if args.test:
        test(device, model, args, test_dataloader, args.num_running_visualizations)
    elif args.val:
        test(device, model, args, val_dataloader, args.num_running_visualizations)
    else:
        train(device, model, optimizer, scheduler, args, train_dataloader, val_dataloader)


def get_dataloader(args, env_vars, split):
    if args.dataset == 'scannet_pc':
        dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_PC_RENDERING_DATASET_PATH'], split, pointcloud_renderings=True)
    elif args.dataset == 'scannet_mesh':
        dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], split, num_views=args.num_views)
    else:
        raise NotImplementedError
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

def get_model(args):
    model = ReconstructionMVCNN(args.num_classes, args.backbone, args.no_reconstruction, args.use_fusion_module, args.cat_cls_res, args.dropout_prob)
    return model


if __name__ == "__main__":
    # Init parser to receive arguments from the terminal
    parser = argparse.ArgumentParser()

    # Standard arguments
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=500)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--overfit", action="store_true", help="use reduced dataset for overfitting")
    parser.add_argument("--tag", type=str, required=True, help="experiment tag for tensorboard logger", default='')
    parser.add_argument("--no_validation", action="store_true", help="do not validate")
    parser.add_argument("--debug", action="store_true", help="switches to debug mode")
    parser.add_argument("--use_checkpoint", type=str, help="specify the checkpoint root", default="")

    # Arguments related training
    parser.add_argument("--lr", type=float, help="learning rate", default=5e-5)
    parser.add_argument("--lr_rec_head", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--lr_cls_head", type=float, help="learning rate", default=5e-5)

    parser.add_argument("--lr_decay_factor", type=float, help="decay factor of the lr scheduler", default=0.5)
    parser.add_argument("--lr_decay_patience", type=float, help="patience of the lr scheduler", default=10)
    parser.add_argument("--lr_decay_cooldown", type=float, help="cooldown of the lr scheduler", default=0)

    parser.add_argument("--wd", type=float, help="weight decay", default=0)
    parser.add_argument("--wd_rec_head", type=float, help="weight decay", default=0)
    parser.add_argument("--wd_cls_head", type=float, help="weight decay", default=0)

    parser.add_argument("--loss_coef_cls", type=float, help="loss coefficient of the classification task", default=0.5)
    parser.add_argument("--loss_coef_rec", type=float, help="loss coefficient of the reconstruction task", default=0.5)

    # Arguments related to MVCNN model
    parser.add_argument("--no_reconstruction", action="store_true", help="no reconstruction, only classification")
    parser.add_argument("--use_fusion_module", action="store_true", help="use fusion module for reconstruction")
    parser.add_argument("--num_classes", type=int, help="number of classes", default=13)
    parser.add_argument("--backbone", type=str, choices=['resnet18_1x1conv', 'resnet18_stdconv', 'mobilenetv3l_1x1conv', 'mobilenetv3s_1x1conv', 'vgg16_1x1conv'], 
                        help="feature extraction backbone", default='resnet18_1x1conv')
    parser.add_argument("--cat_cls_res", action="store_true", help="concatenate classification results to reconstruction feature map")
    parser.add_argument("--dropout_prob", type=float, help="dropout probability in linear layers of classification head", default=0.5)

    # Arguments related to datasets
    # TODO: add more choices
    parser.add_argument("--dataset", type=str, choices=['scannet_mesh', 'scannet_pc'], help="used dataset", default='scannet_mesh')
    parser.add_argument("--num_views", type=int, help="number of views, between 1 and 24", default=4)
    parser.add_argument("--num_workers", type=int, help="multi-process data loading", default=4)
    # TODO: implement
    #parser.add_argument("--augment", action="store_true", help="use data augmentation")
    parser.add_argument("--test", action="store_true", help="test model on test split")
    parser.add_argument("--val", action="store_true", help="test model on val split")
    parser.add_argument("--num_running_visualizations", type=int, help="visualizations for test script", default=3)

    args = parser.parse_args()

    # For reproducability # TODO: check functionalilty
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    main(args)
