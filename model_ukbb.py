import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, DynamicEdgeConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x



class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class DGCNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes=2):
        super(DGCNN, self).__init__()
        # use MLP as h_theta in the paper
        self.h_theta1 = torch.nn.Linear(num_features*2, hidden_channels)
        # set the k_nn to 10, default aggregation function is max, get a dynamic graph
        self.dgc1 = DynamicEdgeConv(self.h_theta1, 10)
        self.h_theta2 = torch.nn.Linear(hidden_channels*2, num_classes)
        # set the k_nn to 8
        self.dgc2 = DynamicEdgeConv(self.h_theta2, 8)

    def forward(self, x, edge_index):
        x = self.dgc1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.dgc2(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes=2):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(num_features, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_indices):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x



if __name__ == "__main__":
    model = GCN(hidden_channels=64)
    print(model)