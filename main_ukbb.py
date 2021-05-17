import argparse

from train_ukbb import main

EPOCH = 120
IS_SEX = True #comment: declare it only here
THRESHOLD = 0.02
LR = 0.01
HIDDEN_CHANNELS = 64
COMMENT = 'TEST'
DELETE = False

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=EPOCH,
                    help='Number of epochs to train.')
parser.add_argument('--threshold', type=float, default=THRESHOLD,
                    help='Threshold')
parser.add_argument('--lr', type=float, default=LR,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=HIDDEN_CHANNELS,
                    help='Number of hidden units.')
parser.add_argument('--is_sex', type=bool, default=IS_SEX,
                    help='Whether the task is sex classification (True) or age (False)')
parser.add_argument('--comment', type=str, default=COMMENT,
                    help='Additional specifier for the Tensorboard folder')
parser.add_argument('--reset_dataset', type=bool, default=DELETE,
                    help='Whether to reset the dataset or not')

parser.add_argument('--log_tensorboard', type=bool, default=True,
                    help='When put false when tuning because the tuner uses tensorboard internally')
parser.add_argument('--verbose', type=bool, default=True)

config = parser.parse_args()



if __name__ == '__main__':
    main(config)