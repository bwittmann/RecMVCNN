
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from datasets import ShapeNetDataset

# TODO: add tensorboard for logging

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

    model.train()

    # Init loss values
    best_loss_val = np.inf
    train_loss_running = 0.

    # Training loop
    for epoch in range(args.epoch):
        print('Starting epoch:', epoch)
        for batch_idx, batch in enumerate(train_dataloader):

            # TODO: move to device
            shapenet_id, renderings, class_labels, voxels = batch

            # Predict and estimate loss
            predictions = model(renderings.to(device).float())
            loss_classification = criterion_classification(predictions, class_labels.to(device))

            # Make a step
            optimizer.zero_grad()
            loss_classification.backward()
            optimizer.step()

            # Logging
            train_loss_running += loss_classification.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % args.verbose == 0 and iteration != 0:
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / args.verbose:.6f}')

                # Tensorboard logging
                tb_logger.add_scalar('loss/train', train_loss_running / args.verbose, iteration)
                tb_logger.add_scalar('epoch', epoch, iteration)

                train_loss_running = 0.

            if iteration % args.val_step == 0 and iteration != 0:
                pass
                # TODO: Add validation, make lr scheduler step and implement model save


def evaluate():
    # Evaluate the classification accuracy
    pass
