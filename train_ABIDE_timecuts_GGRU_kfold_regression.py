import time
from os import path as osp

import torch
from sklearn.model_selection import KFold
import numpy as np

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import f1_score
from scipy.stats import pearsonr

from dataset import ABIDE
from model_hcp import GGRU_ABIDE
from utils import count_parameters, save_code, draw_curves, style

EPOCHS = 115
NUM_NODES = 111
LABEL = 'age' # comment: declare it only here
THRESHOLD = 0.87
LR = 0.001

def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model.forward(data)  # Perform a single forward pass.
        gt = data[0].y.type(torch.FloatTensor).to('cuda')
        loss = criterion(out.squeeze(0), gt)  # Compute the loss.

        running_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    return running_loss / len(train_loader.dataset)


def val(loader, model, criterion):
    model.eval()
    running_loss = 0

    predicted = []
    ground_truth = []

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model.forward(data)
            # out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.squeeze(0), data[0].y)  # Compute the loss.
            running_loss += loss

            predicted.append(float(out.detach()))
            ground_truth.append(float(data[0].y))
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    prediction_correlaton, _ = pearsonr(predicted, ground_truth)
    return running_loss / len(loader.dataset), prediction_correlaton # Derive ratio of correct predictions.


def main_timecuts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ABIDE(root='data/', label=LABEL, num_cuts=4, threshold=THRESHOLD, num_nodes=NUM_NODES)
    print(f'num_cuts: {dataset.num_cuts}')

    dataset = dataset.shuffle()

    model = GGRU_ABIDE(feature_size=dataset[0][0].num_node_features, num_nodes=dataset.num_nodes,
                       num_classes=1).to(device)

    count_parameters(model)

    tb = SummaryWriter(comment='ABIDE_{}_{}_{}'.format(model.__class__.__name__, 'age', dataset.num_nodes))
    save_code(tb.get_logdir())

    val_losses = np.array([])
    pred_correlations = np.array([])
    kfold = KFold(n_splits=10)  # StratifiedKFold -> preserves the class

    try:
        for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold+1}:')
            model.reset_parameters()

            train_dataset = dataset[torch.tensor(train_index)]
            val_dataset = dataset[torch.tensor(test_index)]

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6) # weight_decay original 1e-6, try 1e-5
            decayRate = 0.90
            lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            ## TODO: try MSE
            criterion_train = torch.nn.MSELoss()
            criterion_val = torch.nn.L1Loss()

            losses_per_epoch = ([], [])

            max_pred_correlation = -1
            min_loss = 100000
            for epoch in range(1, EPOCHS):
                t = time.time()
                val_loss, prediction_correlation = val(loader=val_loader, model=model, criterion=criterion_val)
                train_loss = train(train_loader=train_loader,
                                              model=model,
                                              criterion=criterion_train,
                                              optimizer=optimizer)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {style.GREEN_BOLD if val_loss < min_loss else style.CBOLD}{val_loss:.4f}{style.RESET}, '
                      f'Pred Corr: {style.GREEN_BOLD if prediction_correlation > max_pred_correlation else style.CBOLD}{prediction_correlation:.4f}{style.RESET}, '
                      f'time: {(time.time() - t):.4f}s')

                if (epoch < 25):
                    lr_decay.step()

                min_loss = min(min_loss, float(val_loss))
                max_pred_correlation = max(max_pred_correlation, prediction_correlation)

                if train_loss < 0.0002:
                    break  # early stopping

                tb.add_scalars(f'losses_{fold}', {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, epoch)
                losses_per_epoch[0].append(float(train_loss.detach()))
                losses_per_epoch[1].append(float(val_loss.detach()))

            pred_correlations = np.append(pred_correlations, max_pred_correlation)
            val_losses = np.append(val_losses, min_loss)
            print('Min validation loss: ', min_loss)
            print('Max prediction correlation: ', max_pred_correlation)
            draw_curves(list(range(len(losses_per_epoch[0]))), losses_per_epoch, None, f'Fold {fold + 1}')

    finally:
        mean_loss, std_loss = np.mean(val_losses), np.std(val_losses)
        mean_pred, std_pred = np.mean(pred_correlations), np.std(pred_correlations)
        print(f'Loss Mean: {mean_loss}    Std: {std_loss}')
        print(f'Prediction Correlation Mean: {mean_pred}    Std: {std_pred}')
        # save the validation as a file
        f = open(osp.join(tb.get_logdir(), f'loss_{mean_loss}_std_{std_loss}___pred_{mean_pred:.4f}_std_{std_pred:.4f}'), "x")
        f.close()


if __name__ == "__main__":
    print(f'LR: {LR}')
    print(f'Threshold: {THRESHOLD}')
    main_timecuts()

