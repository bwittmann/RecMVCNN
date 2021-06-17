import argparse
import torch
import numpy as np

def main(args):
    # Training, etc.
    pass


if __name__ == "__main__":
    # Init parser to receive arguments from the terminal
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--lr_decay_factor", type=float, help="decay factor of the lr scheduler", default=0.1)
    parser.add_argument("--lr_decay_patience", type=float, help="patience of the lr scheduler", default=10)
    parser.add_argument("--lr_decay_cooldown", type=float, help="cooldown of the lr scheduler", default=0)

    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)

    args = parser.parse_args()

    # For reproducability # TODO: check functionalilty
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    main(args)
