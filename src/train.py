import os

import torch 
import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dotenv import dotenv_values


def train(device, model, optimizer, scheduler, args, train_dataloader, val_dataloader):
    """
    Tried to keep the structure similar to the exercises.
    """
    # Init tensorboard logger
    tb_logger = SummaryWriter(comment=args.tag)

    # Init criteria
    criterion_classification = nn.CrossEntropyLoss()
    criterion_classification.to(device)
    criterion_reconstruction = nn.BCELoss()
    criterion_reconstruction.to(device)

    
    # Init loss and acc related values
    train_loss_running_classification = 0.
    train_loss_running_reconstruction = 0.
    train_loss_running = 0.
    train_correct_classification = 0
    train_total_classification = 0
    train_reconstruction_iou = 0.

    val_loss_running_classification = 0.
    val_loss_running_reconstruction = 0.
    val_loss_running = 0.
    val_correct_classification = 0
    val_total_classification = 0
    val_reconstruction_iou = 0.

    best_accuracy_classification = -np.inf

    model.train()
    try:
        # Main loop
        for epoch in range(1, args.epoch + 1):
            print('Starting epoch:', epoch)
            # Train on train set
            for batch in tqdm(train_dataloader):
                _, renderings, class_labels, voxels = batch
                renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

                # Predict and estimate loss
                predictions_classification, predictions_reconstruction = model(renderings.float())

                train_loss_classification = criterion_classification(predictions_classification, class_labels)
                train_loss_running_classification += train_loss_classification.item()

                if predictions_reconstruction != None:
                    train_loss_reconstruction = criterion_reconstruction(predictions_reconstruction, voxels)
                    train_loss_running_reconstruction += train_loss_reconstruction.item()
                    train_loss = args.loss_coef_cls * train_loss_classification + args.loss_coef_rec * train_loss_reconstruction
                    train_loss_running += train_loss.item()
                    iou = evaluate_reconstruction(predictions_reconstruction.detach().clone(), voxels)
                    train_reconstruction_iou += iou
                else:
                    train_loss = train_loss_classification
                    train_loss_running += train_loss.item()
                
                # Backprob and make a step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                correct_pred = evaluate_classification(predictions_classification, class_labels)

                train_total_classification += predictions_classification.shape[0]
                train_correct_classification += correct_pred

            # Print and log train loss and train acc
            train_accuracy_classificaton = 100 * train_correct_classification / train_total_classification

            print('[epoch:{}] train_loss: {}'.format(epoch, train_loss_running / len(train_dataloader)))
            tb_logger.add_scalar('loss/train_cls', train_loss_running_classification / len(train_dataloader), epoch)
            if predictions_reconstruction != None:
                tb_logger.add_scalar('loss/train', train_loss_running / len(train_dataloader), epoch)
                tb_logger.add_scalar('loss/train_rec', train_loss_running_reconstruction / len(train_dataloader), epoch)
                tb_logger.add_scalar('acc/train_iou', train_reconstruction_iou / len(train_dataloader), epoch)
            tb_logger.add_scalar('acc/train_cls', train_accuracy_classificaton, epoch)

            for param_group, info in zip(optimizer.param_groups, ['_cls', '_rec_head']):
                tb_logger.add_scalar('lr/' + info, param_group['lr'], epoch)

            train_loss_running = 0.
            train_loss_running_classification = 0.
            train_loss_running_reconstruction = 0.
            train_correct_classification = 0
            train_total_classification = 0
            train_reconstruction_iou = 0.

            # Validate on val set
            if not args.no_validation:
                print('Starting validation')
                model.eval()

                for batch in tqdm(val_dataloader):
                    _, renderings, class_labels, voxels = batch
                    renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

                    with torch.no_grad():
                        predictions_classification, predictions_reconstruction = model(renderings.float())

                        val_loss_classification = criterion_classification(predictions_classification, class_labels)
                        val_loss_running_classification += val_loss_classification.item()

                        if predictions_reconstruction != None:
                            val_loss_reconstruction = criterion_reconstruction(predictions_reconstruction, voxels)
                            val_loss_running_reconstruction += val_loss_reconstruction.item()
                            val_loss = args.loss_coef_cls * val_loss_classification + args.loss_coef_rec * val_loss_reconstruction
                            val_loss_running += val_loss.item()
                            iou = evaluate_reconstruction(predictions_reconstruction, voxels)
                            val_reconstruction_iou += iou
                        else:
                            val_loss = val_loss_classification
                            val_loss_running += val_loss_classification.item()

                        correct_pred = evaluate_classification(predictions_classification, class_labels)
                        val_total_classification += predictions_classification.shape[0]
                        val_correct_classification += correct_pred 

                # Estimate val loss and acc
                val_accuracy_classificaton = 100 * val_correct_classification / val_total_classification
                val_loss = val_loss_running / len(val_dataloader)

                # Logging 
                print('[epoch:{}] val_loss: {}, val_acc_cls: {}'.format(epoch, val_loss, val_accuracy_classificaton))
                tb_logger.add_scalar('loss/val_cls', val_loss_running_classification / len(val_dataloader), epoch)
                if predictions_reconstruction != None:
                    tb_logger.add_scalar('loss/val', val_loss, epoch)
                    tb_logger.add_scalar('loss/val_rec', val_loss_running_reconstruction / len(val_dataloader), epoch)
                    tb_logger.add_scalar('acc/val_iou', val_reconstruction_iou / len(val_dataloader), epoch)
                tb_logger.add_scalar('acc/val_cls', val_accuracy_classificaton, epoch)

                # Make a step based on the validation loss
                scheduler.step(val_loss)

                # Save model is accuracy is better that all time best
                if val_accuracy_classificaton >= best_accuracy_classification:
                    best_accuracy_classification = val_accuracy_classificaton
                    print('best classification accuracy -> save model')
                    save_model(model, epoch, optimizer, args, True)

                # Reset loss and acc related values
                val_loss_running = 0.
                val_loss_running_classification = 0.
                val_loss_running_reconstruction = 0.
                val_correct_classification = 0
                val_total_classification = 0
                val_reconstruction_iou = 0.

                model.train()

        # Save checkpoint after epochs ended
        print('training finished -> saving checkpoint')
        save_model(model, epoch, optimizer, args)
    except KeyboardInterrupt:
        # Save checkpoint if interrupted
        print('keyboard interrupt -> saving checkpoint')
        save_model(model, epoch, optimizer, args)


def save_model(model, epoch=None, optimizer=None, args=None, best=False):
    # TODO: test, not sure if scheduler also part of checkpoint
    checkpoint_path = os.path.join(dotenv_values('.env')['PROJECT_DIR_PATH'], 'outputs/{}'.format(args.tag))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if best:
        name = 'model_best.tar'
    else:
        name = 'model_last.tar'

    save_dict = {
        'epoch': epoch, 
        'model_state_dict' : model.state_dict(),
        'optim_state_dict' : optimizer.state_dict() 
    }
    torch.save(save_dict, os.path.join(checkpoint_path, name))



def evaluate_classification(predictions, labels):
    pred_labels = torch.argmax(predictions, dim=1)
    correct_pred = (pred_labels == labels).sum().item()
    return correct_pred


def evaluate_reconstruction(reconstruction, voxel):
    # Convert probabilities to occupancy grid
    reconstruction[reconstruction >= 0.5] = 1
    reconstruction[reconstruction < 0.5] = 0

    # Estimate intersection and union
    intersection = (reconstruction + voxel)
    intersection = torch.count_nonzero(intersection == 2).item()
    union = reconstruction.sum().item() + voxel.sum().item()
    return (intersection/union) * 100
