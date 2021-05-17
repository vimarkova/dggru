import torch
import time
import os.path as osp

from dataset import UKBBDataset
from model_ukbb import GCN, GAT, DGCNN, MLP

from torch.utils.tensorboard import SummaryWriter

def train(model, optimizer, data, criterion, config):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    if config.is_sex:
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
        train_acc = int(correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
    # else:
        # regression return the loss
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
    return train_acc if config.is_sex else loss


def test(model, criterion, data, config):
    model.eval()
    out = model(data.x, data.edge_index)
    # classification for the sex, regression for the age
    if config.is_sex:
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    else:
        # age regression, return loss
        loss = criterion(out[data.test_mask], data.y[data.test_mask])
    return test_acc if config.is_sex else loss


def main(config, epoch_func=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    dataset = UKBBDataset(root=osp.join(osp.dirname(osp.realpath(__file__)), 'data/ukbb'),
                          is_sex=config.is_sex, delete=config.reset_dataset, threshold=config.threshold)
    data = dataset[0]
    data.to(device)

    model = DGCNN(dataset.num_features, hidden_channels=config.hidden, num_classes=2 if config.is_sex else 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.log_tensorboard:
        tb = SummaryWriter(comment='UKBB_{}_{}_th={}_lr{}_comment+{}'.format(
            model.__class__.__name__, 'sex' if config.is_sex else 'age', config.threshold, config.lr, config.comment))
        tb.add_text('configuration', str(config))

    criterion = torch.nn.CrossEntropyLoss() if config.is_sex else torch.nn.L1Loss()

    for epoch in range(1, config.epochs):
        t = time.time()
        train_loss_acc = train(model, optimizer, data, criterion, config)
        val_loss_acc = test(model, criterion, data, config)
        metric = "accuracy" if config.is_sex else "loss"
        if config.verbose:
            print(f'Epoch: {epoch:03d}, Train {metric}: {train_loss_acc:.4f}, Val {metric}: {val_loss_acc:.4f}, time: {(time.time() - t):.4f}s')

        if config.log_tensorboard:
            tb.add_scalars(metric, {
                f'train_{metric}': train_loss_acc,
                f'val_{metric}': val_loss_acc,
            }, epoch)

        # Run the passed function if it exists
        # This is for the tuner
        if (epoch_func):
            epoch_func(train_loss_acc,val_loss_acc)
