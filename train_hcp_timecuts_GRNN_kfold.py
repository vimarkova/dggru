import time
from os import path as osp

import torch
from sklearn.model_selection import KFold
import numpy as np

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HCPDataset_Timepoints, HCPDataset_Raw
from model_hcp import GGRU, GRNN, GRNN_reduced
from utils import count_parameters, save_code, draw_curves

EPOCHS = 20
NUM_NODES = 200
IS_SEX = True # comment: declare it only here
THRESHOLD = 0.5
LR = 0.001 if IS_SEX else 0.005
COMMENT = 'Time_test'



def train(train_loader, model, criterion, optimizer):
    model.train()
    correct = 0
    running_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model.forward(data[0])
        loss = criterion(out, data[1])

        running_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data[1]).sum())  # Check against ground-truth labels.
    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def val(loader, model, criterion):
    model.eval()
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model.forward(data[0])
            # out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data[1])  # Compute the loss.
            running_loss += loss
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data[1]).sum())  # Check against ground-truth labels.
    return running_loss / len(loader.dataset), correct / len(loader.dataset)  # Derive ratio of correct predictions.



def main_timecuts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HCPDataset_Raw(root='data/', is_sex=IS_SEX, num_cuts=32, num_nodes=50)
    print(f'num_cuts: {dataset.num_cuts}')

    dataset = dataset.shuffle()

    model = GRNN(feature_size=dataset[0][0][0][1].shape, num_classes=2 if IS_SEX else 4, num_cuts=dataset.num_cuts, num_nodes=dataset.num_nodes).to(device)

    count_parameters(model)

    tb = SummaryWriter(comment='{}_{}_{}'.format(model.__class__.__name__, 'sex' if IS_SEX else 'age', dataset.num_nodes))
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
            train_dist = np.array([float((train_dataset.labels[train_dataset.indices()] == i).sum()) for i in ([0, 1] if IS_SEX else [0, 1, 2, 3])])
            train_str = f'Train 0: {train_dist[0]}; Train 1: {train_dist[1]} ' if IS_SEX else f'Train 0: {train_dist[0]}; Train 1: {train_dist[1]}; Train 2: {train_dist[2]}; Train 3: {train_dist[3]};'
            print(train_str)
            print(f'Val 0: {(val_dataset.labels[val_dataset.indices()] == 0).sum()}; Val 1: {(val_dataset.labels[val_dataset.indices()] == 1).sum()} ')
            train_dist /= (train_dist[0] + train_dist[1])
            weight = torch.FloatTensor(1 / train_dist).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6) # weight_decay original 1e-6, try 1e-5
            decayRate = 0.90
            lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            criterion_train = torch.nn.CrossEntropyLoss(weight)
            criterion_val = torch.nn.CrossEntropyLoss()

            accuracies_per_epoch = ([], [])
            losses_per_epoch = ([], [])
            max_validation_accuracy = 0
            for epoch in range(1, EPOCHS):
                t = time.time()
                val_loss, val_acc = val(loader=val_loader, model=model, criterion=criterion_val)
                train_loss, train_acc = train(train_loader=train_loader,
                                              model=model,
                                              criterion=criterion_train,
                                              optimizer=optimizer)
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, time: {(time.time() - t):.4f}s')
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

