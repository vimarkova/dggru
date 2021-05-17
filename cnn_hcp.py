import torch
import torch.nn as nn
from torchvision import models, transforms
from dataset import HCPDataset
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

EPOCHS = 120
IS_SEX = True # comment: declare it only here
THRESHOLD = 0.3
LR = 0.003
HIDDEN_CHANNELS = 64
COMMENT = 'TEST'


class CNN_HCP(nn.Module):
    def __init__(self, num_classes):
        super(CNN_HCP, self).__init__()
        self.conv1 = nn.Conv2d(1, 3,  kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=6*7*7, out_features=64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 6*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        return x


def get_data_hcp(batch_size=64):
    # load the hcp data and do the normalization required for pretrained cnn models, the value is in range of [0,1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # use unweighted binary graph
    dataset = HCPDataset(root='data/hcp', is_sex=IS_SEX, threshold=THRESHOLD, bin=True)
    # split dataset for training and test
    dataset_for_training = dataset[:802]
    dataset_for_training = dataset_for_training.shuffle()
    test_dataset = dataset[802:]

    # split the training dataset for validation
    graphs_training = 702
    train_dataset = dataset_for_training[:graphs_training]
    val_dataset = dataset_for_training[graphs_training:]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(train_loader, model, criterion, optimizer, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        if data.x.dim() < 3:
            # change the dimension of node features into [batch_size, 15, 15]
            data.x = torch.reshape(data.x, (-1, 15, 15))
            # dimension required for conv2d [batch_size, channel_in, h, w]
            data.x = torch.unsqueeze(data.x, dim=1)
            # uncomment only for vgg16
            # data.x = transforms.Resize((224, 224))(data.x)
        # dimension of edge to [batch_size, channel_in, h, w]
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(out, data.y.long())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def val(loader, model, device=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        if data.x.dim() < 3:
            # change the dimension into [batch_size, 15, 15]
            data.x = torch.reshape(data.x, (-1, 15, 15))
            data.x = torch.unsqueeze(data.x, dim=1)
            # uncomment only for vgg16
            # data.x = transforms.Resize((224, 224))(data.x)
        out = model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_hcp(batch_size=16)

    model = CNN_HCP(num_classes=2 if IS_SEX else 4).to(device)
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
        val_acc = val(loader=val_loader, model=model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        max_validation_accuracy = max(max_validation_accuracy, val_acc)

        tb.add_scalars(f'accuracies', {
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, epoch)

    print('Max validation accuracy: ', max_validation_accuracy)


if __name__ == "__main__":
    main()