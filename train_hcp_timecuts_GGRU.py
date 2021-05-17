import time
from os import path as osp

import torch

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HCPDataset_Timepoints
from model_hcp import GGRU
from utils import count_parameters, save_code, FocalLoss

EPOCHS = 30
IS_SEX = True # comment: declare it only here
THRESHOLD = 0.5
LR = 0.001 if IS_SEX else 0.005
COMMENT = 'Time_test'


def train_GGRU(train_loader, model, criterion, optimizer):
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


def val_GGRU(loader, model, criterion):
    model.eval()
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model.forward(data)
            # out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data[0].y)  # Compute the loss.
            running_loss += loss
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data[0].y).sum())  # Check against ground-truth labels.
    return running_loss / len(loader.dataset), correct / len(loader.dataset)  # Derive ratio of correct predictions.



def main_timecuts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HCPDataset_Timepoints(root='data/', is_sex=IS_SEX, num_cuts=4, threshold=THRESHOLD, num_nodes=50)
    print(f'num_cuts: {dataset.num_cuts}')
    print(f'num_nodes: {dataset.num_nodes}')

    dataset = dataset.shuffle()

    graphs_training = 802 # 600
    train_dataset = dataset[:graphs_training]
    val_dataset = dataset[graphs_training:]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_dist = [(train_dataset.labels[train_dataset.indices()] == i).sum() for i in [0, 1]]
    train_dist /= (train_dist[0] + train_dist[1])
    weight = torch.FloatTensor(1 / train_dist).to(device)
    print(f'Train 0: {train_dist[0]}; Train 1: {train_dist[1]} ')
    print(f'Val 0: {(val_dataset.labels[val_dataset.indices()] == 0).sum()}; Val 1: {(val_dataset.labels[val_dataset.indices()] == 1).sum()} ')

    val = val_GGRU
    train = train_GGRU
    model = GGRU(feature_size=dataset[0][0].num_node_features, num_classes=2 if IS_SEX else 4, num_nodes=dataset.num_nodes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    decayRate = 0.90
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    criterion_train = torch.nn.CrossEntropyLoss(weight)
    criterion_val = torch.nn.CrossEntropyLoss()
    count_parameters(model)

    tb = SummaryWriter(comment='fmrifeatures_{}_{}'.format(model.__class__.__name__, 'sex' if IS_SEX else 'age'))
    save_code(tb.get_logdir())

    max_validation_accuracy = 0

    try:
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

            if train_loss < 0.0009:
                break  # early stopping

            tb.add_scalars(f'accuracies', {
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, epoch)

            tb.add_scalars(f'losses', {
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, epoch)

            for name, param in model.named_parameters():
                tb.add_histogram(name, param)
    finally:
        print('Max validation accuracy: ', max_validation_accuracy)
        # save the max validation as a file
        f = open(osp.join(tb.get_logdir(), f'max_val_{max_validation_accuracy:.4f}'), "x")
        f.close()


if __name__ == "__main__":
    print(f'LR: {LR}')
    print(f'Threshold: {THRESHOLD}')
    main_timecuts()

