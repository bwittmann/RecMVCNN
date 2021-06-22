import os

import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dotenv import dotenv_values


def train(device, model, args, train_dataloader, val_dataloader):
    """
    Tried to keep the structure similar to the exercises.
    """
    # Init tensorboard logger
    tb_logger = SummaryWriter(comment=args.tag)

    # Init criterion
    criterion_classification = nn.CrossEntropyLoss()
    criterion_classification.to(device)

    # Init optim and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, cooldown=args.lr_decay_cooldown
        )

    # Init loss and acc related values
    train_loss_running_classification = 0.
    val_loss_running_classification = 0.
    best_accuracy_classification = np.inf
    correct_classification = 0
    total_classification = 0
    train_correct_classification = 0
    train_total_classification = 0

    model.train()
    try:
        # Training loop
        for epoch in range(1, args.epoch + 1):
            print('Starting epoch:', epoch)
            for batch_idx, batch in enumerate(train_dataloader):

                _, renderings, class_labels, voxels = batch
                renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

                # Predict and estimate loss
                predictions = model(renderings.float())
                train_loss_classification = criterion_classification(predictions, class_labels)

                # Backprob and make a step
                optimizer.zero_grad()
                train_loss_classification.backward()
                optimizer.step()

                correct_pred = evaluate_classification(predictions, class_labels)
                train_total_classification += predictions.shape[0]
                train_correct_classification += correct_pred

                train_loss_running_classification += train_loss_classification.item()
                iteration = (epoch - 1) * len(train_dataloader) + batch_idx

                # Print and log train loss and train acc
                if iteration % args.verbose == 0:
                    train_accuracy_classificaton = 100 * train_correct_classification / train_total_classification
                    # Logging
                    print('[{}/{}] train_loss: {}'.format(epoch, iteration, train_loss_running_classification / args.verbose))
                    tb_logger.add_scalar('loss/train_cls', train_loss_running_classification / args.verbose, iteration)
                    tb_logger.add_scalar('acc/train_cls', train_accuracy_classificaton, iteration)
                    tb_logger.add_scalar('epoch', epoch, iteration)

                    train_loss_running_classification = 0.
                    train_correct_classification = 0
                    train_total_classification = 0

                # Validate on validation set
                if iteration % args.val_step == 0 and not args.no_validation:
                    print('Starting validation')
                    model.eval()

                    for batch in tqdm(val_dataloader):
                        _, renderings, class_labels, voxels = batch
                        renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

                        with torch.no_grad():
                            predictions = model(renderings.float())
                            correct_pred = evaluate_classification(predictions, class_labels)

                            val_loss_classification = criterion_classification(predictions, class_labels)

                            # Update loss and acc related values
                            val_loss_running_classification += val_loss_classification
                            total_classification += predictions.shape[0]
                            correct_classification += correct_pred 

                    # Estimate val loss and acc
                    accuracy_classificaton = 100 * correct_classification / total_classification
                    val_loss_classification = val_loss_running_classification / len(val_dataloader)

                    # Make a step based on the validation loss
                    scheduler.step(val_loss_classification)

                    # Save model is accuracy is better that all time best
                    if accuracy_classificaton >= best_accuracy_classification:
                        best_accuracy_classification = accuracy_classificaton
                        print('best classification accuracy -> save model')
                        save_model(model)

                    # Logging 
                    print('[{}/{}] val_loss: {}, val_acc_cls: {}'.format(epoch, iteration, val_loss_classification, accuracy_classificaton))
                    tb_logger.add_scalar('loss/val_cls', val_loss_running_classification / len(val_dataloader), iteration)
                    tb_logger.add_scalar('acc/val_cls', accuracy_classificaton, iteration)

                    # Reset loss and acc related values
                    val_loss_running_classification = 0.
                    correct_classification = 0
                    total_classification = 0

                    model.train()

        # Save checkpoint
        print('training finished -> saving checkpoint')
        save_model(model, epoch, optimizer, args, True)
    except KeyboardInterrupt:
        # Save checkpoint
        print('keyboard interrupt -> saving checkpoint')
        save_model(model, epoch, optimizer, args, True)


def save_model(model, epoch=None, optimizer=None, args=None, checkpoint=False):
    # TODO: test, not sure if scheduler also part of checkpoint
    checkpoint_path = os.path.join(dotenv_values('.env')['PROJECT_DIR_PATH'], 'outputs/{}'.format(args.tag))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if checkpoint:
        save_dict = {
            'epoch': epoch, 
            'model_state_dict' : model.state_dict(),
            'optim_state_dict' : optimizer.state_dict() 
        }
        torch.save(save_dict, os.path.join(checkpoint_path, 'checkpoint.tar'))
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_best.pth'))


def evaluate_classification(predictions, labels):
    pred_labels = torch.argmax(predictions, dim=1)
    correct_pred = (pred_labels == labels).sum().item()
    return correct_pred


def evaluate_reconstruction():
    pass
