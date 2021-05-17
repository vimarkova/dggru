import time
from os import path as osp

import torch
from sklearn.model_selection import KFold
import numpy as np

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import f1_score

from dataset import HCPDataset_Timepoints, ABIDE
from model_hcp import GGRU
from utils import count_parameters, save_code, draw_curves

EPOCHS = 20
NUM_NODES = 50
IS_SEX = False # comment: declare it only here
THRESHOLD = 0.5
LR = 0.001 if IS_SEX else 0.005
COMMENT = 'Time_test'


def train(train_loader, model, criterion, optimizer):
    model.train()
    correct = 0
    running_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model.forward(data)  # Perform a single forward pass.
        loss = criterion(out, data[0].y)  # Compute the loss.

        running_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data[0].y).sum())  # Check against ground-truth labels.
    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def val(loader, model, criterion):
    model.eval()
    correct = 0
    running_loss = 0

    target = torch.empty((0,), device='cuda')
    predict = torch.empty((0,), device='cuda')

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model.forward(data)
            # out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data[0].y)  # Compute the loss.
            running_loss += loss
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data[0].y).sum())  # Check against ground-truth labels.
            target = torch.cat((target, data[0].y), dim=0)
            predict = torch.cat((predict, pred), dim=0)
    f1 = f1_score(predict, target, num_classes=2 if IS_SEX else 4)

    return running_loss / len(loader.dataset), correct / len(loader.dataset), f1  # Derive ratio of correct predictions.


def main_timecuts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HCPDataset_Timepoints(root='data/hcp', is_sex=IS_SEX, num_cuts=4, threshold=THRESHOLD, num_nodes=NUM_NODES)
    #dataset = ABIDE(root='data/', is_sex=IS_SEX, num_cuts=4, threshold=THRESHOLD, num_nodes=NUM_NODES)
    print(f'num_cuts: {dataset.num_cuts}')

    dataset = dataset.shuffle()

    model = GGRU(feature_size=dataset[0][0].num_node_features, num_classes=2 if IS_SEX else 4, num_nodes=dataset.num_nodes).to(device)

    count_parameters(model)

    tb = SummaryWriter(comment='HCP_{}_{}_{}'.format(model.__class__.__name__, 'sex' if IS_SEX else 'age', dataset.num_nodes))
    save_code(tb.get_logdir())

    val_accuracies = np.array([])
    kfold = KFold(n_splits=10)  # StratifiedKFold -> preserves the class

    try:
        for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold+1}:')
            model.reset_parameters()

            train_dataset = dataset[torch.tensor(train_index)]
            val_dataset = dataset[torch.tensor(test_index)]

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            train_dist = [(train_dataset.labels[train_dataset.indices()] == i).sum() for i in ([0, 1] if IS_SEX else [0, 1, 2, 3])]
            train_str = f'Train 0: {train_dist[0]}; Train 1: {train_dist[1]} ' if IS_SEX else f'Train 0: {train_dist[0]}; Train 1: {train_dist[1]}; Train 2: {train_dist[2]}; Train 3: {train_dist[3]};'
            print(train_str)
            print(f'Val 0: {(val_dataset.labels[val_dataset.indices()] == 0).sum()}; Val 1: {(val_dataset.labels[val_dataset.indices()] == 1).sum()} ')
            train_dist /= (train_dist[0] + train_dist[1]) if IS_SEX else (train_dist[0] + train_dist[1] + train_dist[2] + train_dist[3])
            weight = torch.FloatTensor(1 / train_dist).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6) # weight_decay original 1e-6, try 1e-5
            decayRate = 0.90
            lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            criterion_train = torch.nn.L1Loss()
            criterion_val = torch.nn.L1Loss()

            accuracies_per_epoch = ([], [])
            losses_per_epoch = ([], [])
            max_validation_accuracy = 0
            for epoch in range(1, EPOCHS):
                t = time.time()
                val_loss, val_acc, val_f1 = val(loader=val_loader, model=model, criterion=criterion_val)
                train_loss, train_acc = train(train_loader=train_loader,
                                              model=model,
                                              criterion=criterion_train,
                                              optimizer=optimizer)
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1[0]:.2f} | {val_f1[1]:.2f}, time: {(time.time() - t):.4f}s')
                lr_decay.step()

                max_validation_accuracy = max(max_validation_accuracy, val_acc)

                if train_loss < 0.0002:
                    break  # early stopping

                tb.add_scalars(f'accuracies_{fold}', {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, epoch)
                accuracies_per_epoch[0].append(train_acc)
                accuracies_per_epoch[1].append(val_acc)

                tb.add_scalars(f'losses_{fold}', {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, epoch)
                losses_per_epoch[0].append(float(train_loss.detach()))
                losses_per_epoch[1].append(float(val_loss.detach()))


            val_accuracies = np.append(val_accuracies, max_validation_accuracy)
            print('Max validation accuracy: ', max_validation_accuracy)
            draw_curves(list(range(len(accuracies_per_epoch[0]))), losses_per_epoch, accuracies_per_epoch, f'Fold {fold + 1}')

    finally:
        mean_acc, std_acc = np.mean(val_accuracies), np.std(val_accuracies)
        print(f'Mean: {mean_acc}    Std: {std_acc}')
        # save the validation as a file
        f = open(osp.join(tb.get_logdir(), f'val_{mean_acc:.4f}_std_{std_acc:.4f}'), "x")
        f.close()


if __name__ == "__main__":
    print(f'LR: {LR}')
    print(f'Threshold: {THRESHOLD}')
    main_timecuts()

