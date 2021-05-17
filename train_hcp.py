import torch

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HCPDataset, HCPDataset_Timepoints, HCPDataset_Age_Reduced
from model_hcp import GCN, GCN_skip_connections, GAT


EPOCHS = 120
IS_SEX = False # comment: declare it only here
THRESHOLD = 0.3
LR = 0.01
HIDDEN_CHANNELS = 64
BATCH=1
COMMENT = 'TEST'

def train(train_loader, model, criterion, optimizer):
    """Function to perform training step.

    Args:
        train_loader (DataLoader): Loads `train_set` data as mini-batches.
        model (nn.Module): Neural network model used.
        criterion (nn.loss): Loss function used in for optimization.
        optimizer (torch.optim.Optimizer): Optimization method of choice.
        device (str, optional): Flag for `cpu` or `gpu`. Defaults to None.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.long())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def val(loader, model, device=None):
    """Function to compute and return accuracy (`age` or `sex`)

    Args:
        loader (DataLoader): Loads either `train_set` or `val_set` as mini-batches
        model (nn.Module): Neural network model used.
        device (str, optional): Flag for `cpu` or `gpu`. Defaults to None.

    Returns:
        Classification accuracy (`age` or `sex`) on `train_set` or `val_set`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HCPDataset_Age_Reduced(root='data/', is_sex=IS_SEX, threshold=THRESHOLD)
    dataset = dataset.shuffle()

    graphs_training = 650 # 802
    train_dataset = dataset[:graphs_training]
    val_dataset = dataset[graphs_training:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    #model = GCN(hidden_channels=HIDDEN_CHANNELS, num_node_features=dataset.num_node_features, num_classes=2 if IS_SEX else 4)
    model = GCN(hidden_channels=HIDDEN_CHANNELS, num_node_features=dataset.num_node_features, num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    max_validation_accuracy = 0

    tb = SummaryWriter(comment='{}_{}_th={}_lr={}_comment={}'.format(
        model.__class__.__name__, 'sex' if IS_SEX else 'age', THRESHOLD, LR, COMMENT))


    for epoch in range(1, EPOCHS):
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer)
        train_acc = val(loader=train_loader, model=model)
        val_acc = val(loader=val_loader,  model=model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        max_validation_accuracy = max(max_validation_accuracy, val_acc)

        tb.add_scalars(f'accuracies', {
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, epoch)

    print('Max validation accuracy: ', max_validation_accuracy)


if __name__ == "__main__":
    main()
