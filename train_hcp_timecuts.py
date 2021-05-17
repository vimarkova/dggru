import time
from os import path as osp

import torch

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HCPDataset_Timepoints, HCPDataset_Raw
from model_hcp import GRNN, GRNN_reduced, GRNN_kTop
from utils import count_parameters, save_code


EPOCHS = 120
IS_SEX = True # comment: declare it only here
LR = 0.01 if IS_SEX else 0.0005
COMMENT = 'Time_test'
HIDDEN_CHANNELS = 64


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

    graphs_training = 802 # 802
    train_dataset = dataset[:graphs_training]
    val_dataset = dataset[graphs_training:]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f'Train 0: {(train_dataset.labels[train_dataset.indices()] == 0).sum()}; Train 1: {(train_dataset.labels[train_dataset.indices()] == 1).sum()} ')
    print(f'Val 0: {(val_dataset.labels[val_dataset.indices()] == 0).sum()}; Val 1: {(val_dataset.labels[val_dataset.indices()] == 1).sum()} ')

    #model = GRNN_reduced(feature_size=dataset[0][0][0][1].shape, num_classes=2, num_cuts=dataset.num_cuts).to(device)
    model = GRNN_kTop(feature_size=dataset[0][0][0][1].shape, num_classes=2, num_cuts=dataset.num_cuts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    count_parameters(model)

    tb = SummaryWriter(comment='{}_{}'.format(model.__class__.__name__, 'sex' if IS_SEX else 'age'))
    save_code(tb.get_logdir())

    max_validation_accuracy = 0
    try:
        for epoch in range(1, EPOCHS):
            t = time.time()
            train_loss, train_acc = train(train_loader=train_loader,
                                          model=model,
                                          criterion=criterion,
                                          optimizer=optimizer)
            val_loss, val_acc = val(loader=val_loader, model=model, criterion=criterion)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, time: {(time.time() - t):.4f}s')

            if train_loss < 0.0002:
                break  # early stopping

            max_validation_accuracy = max(max_validation_accuracy, val_acc)

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
    main_timecuts()
