import os
import argparse
import torch
from torch import nn, optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from torch.utils.data import random_split

from mvcnn import MVCNN
from utils import env_vars
from datasets import ShapeNetDataset


def load_data():
    trainset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], 'train')
    testset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], 'val')
    return trainset, testset


def hyperparameter_search(config, device, model, checkpoint_dir):
    criterion_classification = nn.CrossEntropyLoss()
    criterion_classification.to(device)

    optimizer = optim.Adam(model.parameters(), config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, valset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]))

    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]))

    model.train()
    for epoch in range(1, 10):
        print('Starting epoch:', epoch)
        running_loss = 0.0
        epoch_steps = 0
        for batch_idx, batch in enumerate(trainloader):
            _, renderings, class_labels, voxels = batch
            renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

            # Predict and estimate loss
            predictions = model(renderings.float())
            train_loss_classification = criterion_classification(predictions, class_labels)

            # Backprop and make a step
            optimizer.zero_grad()
            train_loss_classification.backward()
            optimizer.step()

            running_loss += train_loss_classification.item()
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                _, renderings, class_labels, voxels = batch
                renderings, class_labels, voxels = renderings.to(device), class_labels.to(device), voxels.to(device)

                outputs = model(renderings.float())
                _, predicted = torch.max(outputs.data, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()

                loss = model(outputs, class_labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--num_workers", type=int, help="number of epochs", default=1)
    args = parser.parse_args()

    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MVCNN(13, 'vgg16')
    model.to(device)
    
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(hyperparameter_search, device=device, model=model, checkpoint_dir='models/hyperparameter_search'),
        resources_per_trial={"cpu": 4, "gpu": 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))