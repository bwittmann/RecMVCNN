import os

import torch 
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from train import evaluate_classification, evaluate_reconstruction
from utils import env_vars, save_voxel_grid
from utils import visualize_voxel_grid

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def test(device, model, args, dataloader, num_running_visualizations, sort_iou=True):
    visualizations_path = os.path.join(env_vars['PROJECT_DIR_PATH'], 'visualizations/{}'.format(args.tag))
    
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path)

    criterion_classification = nn.CrossEntropyLoss()
    criterion_classification.to(device)
    criterion_reconstruction = nn.BCELoss()
    criterion_reconstruction.to(device)

    loss_running_classification = 0.
    loss_running_reconstruction = 0.
    loss_running_total = 0.

    correct_classification = 0
    total_classification = 0
    reconstruction_iou = 0.
    model.eval()

    confusion_matrix = torch.zeros(model.num_classes, model.num_classes, dtype=torch.int)
    viz_count = 0

    iou_dict = {}

    for batch in tqdm(dataloader):
        shapenet_ids, renderings, class_labels, voxels = batch
        renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

        with torch.no_grad():
            predictions_classification, predictions_reconstruction = model(renderings.float())
            loss_classification = criterion_classification(predictions_classification, class_labels)
            loss_running_classification += loss_classification

            if predictions_reconstruction is not None:
                if len(list(predictions_reconstruction.size())) == 3:
                    predictions_reconstruction = torch.unsqueeze(predictions_reconstruction, 0)

                loss_reconstruction = criterion_reconstruction(predictions_reconstruction, voxels)
                loss_running_reconstruction += loss_reconstruction.item()
                loss = args.loss_coef_cls * loss_classification + args.loss_coef_rec * loss_reconstruction
                loss_running_total += loss.item()
                iou = evaluate_reconstruction(predictions_reconstruction, voxels)
                reconstruction_iou += iou

                if sort_iou:
                    for i in range(predictions_reconstruction.shape[0]):
                        single_iou = evaluate_reconstruction(predictions_reconstruction[i], voxels[i])
                        iou_dict[shapenet_ids[i]] = single_iou

            else:
                loss = loss_classification
                loss_running_total += loss_classification.item()

            for recon, id in zip(predictions_reconstruction, shapenet_ids):
                save_voxel_grid(visualizations_path + f'/{id.replace("/", "_", 1)}.ply', recon.cpu().numpy())

                if num_running_visualizations > viz_count:
                    visualize_voxel_grid(recon.cpu().numpy())
                    viz_count += 1

            pred_labels = torch.argmax(predictions_classification, dim=1)

            for i in range(len(pred_labels)):
                confusion_matrix[pred_labels[i].item(), class_labels[i].item()] += 1       

            correct_pred = evaluate_classification(predictions_classification, class_labels)
            total_classification += predictions_classification.shape[0]
            correct_classification += correct_pred 

    loss_classification = loss_running_classification / len(dataloader)

    if sort_iou:
        sorted_iou = sorted(iou_dict.items(), key=lambda x: x[1])
        with open(visualizations_path + '/iou_sorted.txt', "w") as f:
            for i in sorted_iou:
                f.write("%s\n" % (i[0] + " " + str(i[1])))

    print("\n----------")
    print("Evaluation results:")
    print(f"Classifcation accuracy: {correct_classification / total_classification}")
    print(f"Reconstruction IoU: {reconstruction_iou / len(dataloader)}")
    print("----------\n")

    df_cm = pd.DataFrame(confusion_matrix.numpy(), index = [dataloader.dataset.class_name_mapping[i] for i in dataloader.dataset.classes],
                  columns = [dataloader.dataset.class_name_mapping[i] for i in dataloader.dataset.classes])
    plt.figure(figsize = (model.num_classes, model.num_classes))
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.show()