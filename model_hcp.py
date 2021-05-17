import torch

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, GlobalAttention
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import DynamicEdgeConv


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes=2):
        super(GCN, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


#   As multiple papers pointed out (Xu et al. (2018), Morris et al. (2018)), applying neighborhood normalization decreases
#   the expressivity of GNNs in distinguishing certain graph structures.  An alternative formulation (Morris et al. (2018))
#   omits neighborhood normalization completely and adds a simple skip-connection to the
#   GNN layer in order to preserve central node information:
class GCN_skip_connections(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes=2):
        super(GCN_skip_connections, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes=2):
        super(GAT, self).__init__()
        self.name = "GAT"

        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


from torch import Tensor
from torch_geometric.nn import MessagePassing, knn
from torch_geometric.nn.inits import reset
class DGCN(MessagePassing):
    def __init__(self, nn: torch.nn.Module, k: int, aggr: str = 'max', num_workers: int = 1, **kwargs):
        super(DGCN, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, batch=None, edge_index_original=None):
        """"""
        if isinstance(x, Tensor):
            x = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `DGCN`.'

        b = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1], cosine=False,
                         num_workers=self.num_workers)

        if edge_index_original is not None:
            batch_size = batch.max().item() + 1
            num_nodes = x[0].shape[0] // batch_size

            #edge_index = find_same_edges(num_nodes, edge_index_original, edge_index, batch)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None), edge_index

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


# https://github.com/zhiyongc/Graph_Convolutional_LSTM/blob/master/Code_V2/HGC_LSTM%20%26%20Experiments.ipynb

from torch_geometric.data import Data
class GGRU(torch.nn.Module):

### IDEA: we could also just update the features with the correlation matrix but use the netmaps for the edges
    def __init__(self, feature_size, num_classes=2, num_nodes=200, regression=False):
        super(GGRU, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_regression = regression

        self.feature_size = int(feature_size)
        self.hidden_size = int(feature_size)
        self.num_nodes = num_nodes #int(feature_size)

        self.first_hidden = self.initHidden()
        self.hidden = [self.first_hidden]
        self.hidden_light = [(self.first_hidden.x, self.first_hidden.edge_index)]

        self.lin = Linear(feature_size, 1 if self.is_regression else num_classes)

        self.leakyrelu = torch.nn.LeakyReLU(0.1)


        hidden_size = self.feature_size
        input_size = self.feature_size

        self.conv_z_input = GCNConv(input_size, hidden_size)
        self.conv_z_hidden = GCNConv(input_size, hidden_size)

        self.conv_r_input = GCNConv(input_size, hidden_size)
        self.conv_r_hidden = GCNConv(input_size, hidden_size)

        self.conv_h_input = GCNConv(input_size, hidden_size)
        self.conv_h_hidden = GCNConv(input_size, hidden_size)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.conv_z_input.reset_parameters()
        self.conv_z_hidden.reset_parameters()

        self.conv_r_input.reset_parameters()
        self.conv_r_hidden.reset_parameters()

        self.conv_h_input.reset_parameters()
        self.conv_h_hidden.reset_parameters()

    def weigh_graph(self, r, graph2: Data):
        graph2.x = r * graph2.x
        return graph2

    def add_graphs(self, prev, next: Data):
        next.x = prev.x + next.x
        return next

    def weigh_graph_light(self, r, graph2_features):
        return r * graph2_features # + edges graph2

    def add_graphs_light(self, prev_features, next_features):
        return prev_features + next_features # + edges next2


    def step_light(self, data_features, data_edges,
                   hidden_features, hidden_edges):

        z: torch.Tensor = torch.sigmoid(self.conv_z_input(data_features, data_edges) +
                      self.conv_z_hidden(hidden_features, hidden_edges))
        r: torch.Tensor = torch.sigmoid(self.conv_r_input(data_features, data_edges) +
                      self.conv_r_hidden(hidden_features, hidden_edges))

        weighted = self.weigh_graph_light(r, hidden_features)
        candadate_activations = torch.tanh(self.conv_h_input(data_features,data_edges) +
                              self.conv_h_hidden(weighted, hidden_edges))

        return self.add_graphs_light(self.weigh_graph_light(torch.ones(z.shape).to(self.device) - z, hidden_features),
                                 self.weigh_graph_light(z, candadate_activations))

    def step(self, data: [Data], output):


        z: torch.Tensor = torch.sigmoid(self.conv_z_input(data.x, data.edge_index, data.edge_attr) +
                      self.conv_z_hidden(output.x, output.edge_index, output.edge_attr))
        r: torch.Tensor = torch.sigmoid(self.conv_r_input(data.x, data.edge_index, data.edge_attr) +
                      self.conv_r_hidden(output.x, output.edge_index, output.edge_attr))

        weighted = self.weigh_graph(r, output)
        candadateAct : torch.Tensor = torch.tanh(self.conv_h_input(data.x, data.edge_index, data.edge_attr) +
                              self.conv_h_hidden(weighted.x, weighted.edge_index, weighted.edge_attr))
        candadateX = Data(x=candadateAct, edge_index=data.edge_index)

        return self.add_graphs(self.weigh_graph(torch.ones(z.shape).to(self.device) - z, output),
                                 self.weigh_graph(z, candadateX))


    def forward(self, data_list):
        time_step = len(data_list)
        Hidden_State = self.initHidden()
        outputs = [Hidden_State]

        for i in range(time_step):
            outputs.append(self.step(data_list[i], Hidden_State))

        # edges are not relevant beyond this point so it doesn't matter if we begin with the outputs[1] or some other

        x = torch.cat((outputs[1].x.unsqueeze(2), outputs[2].x.unsqueeze(2),
                       outputs[3].x.unsqueeze(2), outputs[4].x.unsqueeze(2)), 2)

        x = x.sum(dim=2)

        # 2. Readout layer
        x = global_mean_pool(x, data_list[0].batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        self.hidden = [self.first_hidden]
        return x

    def initHidden(self):
        output = Data(
            x=torch.autograd.Variable(torch.eye(self.num_nodes, self.hidden_size)),
                                      edge_index=torch.tensor([[], []], dtype=torch.long))
        output.to(self.device)
        return output




class GGRU_ABIDE(torch.nn.Module):
    def __init__(self, feature_size, num_classes=2, num_nodes=111):
        super(GGRU_ABIDE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_size = int(feature_size)
        self.hidden_size = int(feature_size)
        self.num_nodes = num_nodes #int(feature_size)

        self.first_hidden = self.initHidden()
        self.hidden = [self.first_hidden]
        self.hidden_light = [(self.first_hidden.x, self.first_hidden.edge_index)]


        self.lin = Linear(feature_size, num_classes)
        # self.lin = Linear(feature_size, int(feature_size / 4))
        self.lin2 = Linear(int(feature_size / 4), num_classes)

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

        hidden_size = self.feature_size
        input_size = self.feature_size

        self.conv_z_input = GCNConv(input_size, hidden_size)
        self.conv_z_hidden = GCNConv(input_size, hidden_size)

        self.conv_r_input = GCNConv(input_size, hidden_size)
        self.conv_r_hidden = GCNConv(input_size, hidden_size)

        self.conv_h_input = GCNConv(input_size, hidden_size)
        self.conv_h_hidden = GCNConv(input_size, hidden_size)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        self.conv_z_input.reset_parameters()
        self.conv_z_hidden.reset_parameters()

        self.conv_r_input.reset_parameters()
        self.conv_r_hidden.reset_parameters()

        self.conv_h_input.reset_parameters()
        self.conv_h_hidden.reset_parameters()

    def weigh_graph(self, r, graph2: Data):
        graph2.x = r * graph2.x
        return graph2

    def add_graphs(self, prev, next: Data):
        next.x = prev.x + next.x
        return next

    def weigh_graph_light(self, r, graph2_features):
        return r * graph2_features # + edges graph2

    def add_graphs_light(self, prev_features, next_features):
        return prev_features + next_features # + edges next2


    def step(self, data: [Data], output):


        z: torch.Tensor = torch.sigmoid(self.conv_z_input(data.x, data.edge_index, data.edge_attr) +
                      self.conv_z_hidden(output.x, output.edge_index, output.edge_attr))
        r: torch.Tensor = torch.sigmoid(self.conv_r_input(data.x, data.edge_index, data.edge_attr) +
                      self.conv_r_hidden(output.x, output.edge_index, output.edge_attr))

        weighted = self.weigh_graph(r, output)
        candadateAct : torch.Tensor = torch.tanh(self.conv_h_input(data.x, data.edge_index, data.edge_attr) +
                              self.conv_h_hidden(weighted.x, weighted.edge_index, weighted.edge_attr))
        candadateX = Data(x=candadateAct, edge_index=data.edge_index)

        return self.add_graphs(self.weigh_graph(torch.ones(z.shape).to(self.device) - z, output),
                                 self.weigh_graph(z, candadateX))


    def forward(self, data_list):
        time_step = len(data_list)
        Hidden_State = self.initHidden()
        outputs = [Hidden_State]

        for i in range(time_step):
            outputs.append(self.step(data_list[i], Hidden_State))

        # edges are not relevant beyond this point so it doesn't matter if we begin with the outputs[1] or some other

        x = torch.cat((outputs[1].x.unsqueeze(2), outputs[2].x.unsqueeze(2),
                       outputs[3].x.unsqueeze(2), outputs[4].x.unsqueeze(2)), 2)

        x = x.sum(dim=2)

        # 2. Readout layer
        x = global_mean_pool(x, data_list[0].batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        #x = self.leakyrelu(x)
        #x = self.lin2(x)

        self.hidden = [self.first_hidden]
        return x

    def initHidden(self):
        output = Data(
            x=torch.autograd.Variable(torch.eye(self.num_nodes, self.hidden_size)),
                                      edge_index=torch.tensor([[], []], dtype=torch.long))
        output.to(self.device)
        return output



from utils_dataset import graph_from_series_torch
class GRNN(torch.nn.Module):

    def __init__(self, feature_size, num_classes=2, num_cuts=32, num_nodes=200):
        super(GRNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size

        self.initial_size = feature_size[0]  # 150
        self.hidden_size_1 = int(100)
        self.hidden_size_2 = int(50)

        self.num_cuts = num_cuts
        self.num_nodes = num_nodes

        self.knn1 = 6  # 6

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gru = torch.nn.GRU(self.initial_size, self.hidden_size_1)  # Input dim is 3, output dim is 3
        # tune the 400

        # use MLP as h_theta in the paper

        self.h_theta1 = torch.nn.Linear(self.hidden_size_1 * self.num_cuts * 2, self.hidden_size_2)

        self.dgcn1 = DynamicEdgeConv(self.h_theta1, k=self.knn1)

        self.classifier = torch.nn.Linear(self.hidden_size_2, num_classes)

        self.conv = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)

        self.first_hidden = self.initHidden()
        self.hidden = [self.first_hidden]
        self.batch = torch.zeros(self.num_nodes, dtype=torch.int64).to(self.device)

    def step(self, data):
        data = self.conv(data)
        data = self.leakyrelu(data)
        data = self.conv2(data)
        data = self.leakyrelu(data)
        data = self.conv3(data)
        data = self.leakyrelu(data)

        x, hidden = self.gru(data, self.hidden[-1])  # remove unsqueeze for gru, put for lstm

        self.hidden.append(hidden)
        return x

    def forward(self, timeseries_list):
        time_step = len(timeseries_list)

        timeseries = []
        for i in range(time_step):
            timeseries.append(self.step(timeseries_list[i]))

        t = torch.cat(timeseries, 2)

        t = self.leakyrelu(t)

        x = self.dgcn1(t.squeeze(0), 1)
        x = self.leakyrelu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, self.batch)

        x = self.classifier(x)

        self.hidden = [
            self.first_hidden]  # https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
        return x

    def initHidden(self):
        output = torch.autograd.Variable(torch.zeros(1, self.num_nodes, self.hidden_size_1)).to(self.device)
        output.detach()
        return output

    def reset_parameters(self):
        self.conv3.reset_parameters()
        self.conv.reset_parameters()
        self.conv2.reset_parameters()
        self.gru.reset_parameters()
        self.dgcn1.reset_parameters()
        self.classifier.reset_parameters()



class GRNN_reduced(torch.nn.Module):

    def __init__(self, feature_size ,num_classes=2, num_cuts=32, num_nodes = 200):
        super(GRNN_reduced, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size

        self.initial_size = feature_size[0] # 150
        self.hidden_size_1 = int(100)
        self.hidden_size_2 = int(50)

        self.num_cuts = num_cuts
        self.num_nodes = num_nodes

        self.knn1 = 6 #6

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gru = torch.nn.GRU(self.initial_size, self.hidden_size_1)  # Input dim is 3, output dim is 3
        # tune the 400

        # use MLP as h_theta in the paper

        self.h_theta1 = torch.nn.Linear(self.hidden_size_1 * self.num_cuts * 2, self.hidden_size_2)

        #self.dgc1 = DGCN(self.mlp1, k=k)
        self.dgc1 = DynamicEdgeConv(self.h_theta1, self.knn1)

        self.classifier = torch.nn.Linear(self.hidden_size_2, num_classes)

        self.conv = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)


        self.first_hidden = self.initHidden()
        self.hidden = [self.first_hidden]
        self.batch = torch.zeros(self.num_nodes, dtype=torch.int64).to(self.device)

    def reset_parameters(self):
        self.conv3.reset_parameters()
        self.conv.reset_parameters()
        self.conv2.reset_parameters()
        self.gru.reset_parameters()
        self.dgc1.reset_parameters()
        self.classifier.reset_parameters()

    def step(self, data):
        data = self.conv(data)
        data = self.leakyrelu(data)
        data = self.conv2(data)
        data = self.leakyrelu(data)
        data = self.conv3(data)
        data = self.leakyrelu(data)

        x, hidden = self.gru(data, self.hidden[-1]) # remove unsqueeze for gru, put for lstm

        self.hidden.append(hidden)
        return x

    def forward(self, timeseries_list):
        time_step = len(timeseries_list)

        timeseries = []
        for i in range(time_step):
            timeseries.append(self.step(timeseries_list[i]))

        t = torch.cat(timeseries, 2)

        t = self.leakyrelu(t)

        x = self.dgc1(t.squeeze(0), 1)
        x = self.leakyrelu(x)
        x = F.dropout(x, p=0.5, training=self.training)


        x = global_mean_pool(x, self.batch)

        x = self.classifier(x)

        self.hidden = [self.first_hidden] # https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
        return x

    def initHidden(self):
        output = torch.autograd.Variable(torch.zeros(1, self.num_nodes, self.hidden_size_1)).to(self.device)
        output.detach()
        return output



class GRNN_kTop(torch.nn.Module):

    def __init__(self, feature_size, num_classes=2, num_cuts=32, num_nodes=50):
        super(GRNN_kTop, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size

        self.initial_size = feature_size[0]  # 150
        self.hidden_size_1 = int(100)
        self.hidden_size_2 = int(50)

        self.num_cuts = num_cuts
        self.num_nodes = num_nodes

        self.knn1 = 6  # 6

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gru = torch.nn.GRU(self.initial_size, self.hidden_size_1)  # Input dim is 3, output dim is 3
        # tune the 400

        self.topk_classifier = DGCN_TopK_2(16, self.hidden_size_1*self.num_cuts)

        self.conv = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(self.num_nodes, self.num_nodes, 5, stride=1, padding=2)

        self.first_hidden = self.initHidden()
        self.hidden = [self.first_hidden]
        self.batch = torch.zeros(self.num_nodes, dtype=torch.int64).to(self.device)

    def step(self, data):
        data = self.conv(data)
        data = self.leakyrelu(data)
        data = self.conv2(data)
        data = self.leakyrelu(data)
        data = self.conv3(data)
        data = self.leakyrelu(data)

        x, hidden = self.gru(data, self.hidden[-1])  # remove unsqueeze for gru, put for lstm

        self.hidden.append(hidden)
        return x

    def forward(self, timeseries_list):
        time_step = len(timeseries_list)

        timeseries = []
        for i in range(time_step):
            timeseries.append(self.step(timeseries_list[i]))

        t = torch.cat(timeseries, 2)
        t = t.squeeze(0)

        t = self.leakyrelu(t)

        x = self.topk_classifier(t, None, None, self.batch)

        self.hidden = [
            self.first_hidden]  # https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
        return x

    def initHidden(self):
        output = torch.autograd.Variable(torch.zeros(1, self.num_nodes, self.hidden_size_1)).to(self.device)
        output.detach()
        return output

    def reset_parameters(self):
        self.conv3.reset_parameters()
        self.conv.reset_parameters()
        self.conv2.reset_parameters()
        self.gru.reset_parameters()
        self.topk_classifier.reset_parameters()
        self.classifier.reset_parameters()


class DGCN_TopK_2(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, k=20, num_classes=2):
        super(DGCN_TopK_2, self).__init__()
        self.name = 'DGCN_TopK_2'
        self.mlp1 = torch.nn.Sequential(Linear(2*num_node_features, hidden_channels), torch.nn.ReLU(),
                                  Linear(hidden_channels, hidden_channels))
        self.dgcn1 = DGCN(self.mlp1, k=k)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        # self.gcn1 = GCNConv(hidden_channels, hidden_channels)

        self.mlp2 = torch.nn.Sequential(Linear(hidden_channels*2, hidden_channels), torch.nn.ReLU(),
                                  Linear(hidden_channels, hidden_channels))
        self.dgcn2 = DGCN(self.mlp2, k=k)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        # self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        """
        self.mlp3 = Linear(hidden_channels*2, hidden_channels)
        self.dgcn3 = DGCN(self.mlp3, k=int(k*0.8*0.8))
        self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
        """

        # try global attention pooling layer
        self.gate1 = Linear(hidden_channels, 1)
        self.gap1 = GlobalAttention(self.gate1)
        self.gate2 = Linear(hidden_channels, 1)
        self.gap2 = GlobalAttention(self.gate2)
        # self.gate3 = Linear(hidden_channels, 1)
        # self.gap3 = GlobalAttention(self.gate3)

        # self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def reset_parameters(self):
        for module in self.mlp1.children():
            if isinstance(module, Linear):
                module.reset_parameters()

        self.dgcn1.reset_parameters()
        self.pool1.reset_parameters()

        for module in self.mlp2.children():
            if isinstance(module, Linear):
                module.reset_parameters()
        self.dgcn2.reset_parameters()
        self.pool2.reset_parameters()


        self.gate1.reset_parameters()
        self.gap1.reset_parameters()
        self.gate2.reset_parameters()
        self.gap2.reset_parameters()

        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight, batch):
        x, edge_index = self.dgcn1(x, batch)
        x = F.relu(x)
        x, edge_index, edge_weight, batch1, perm1, score1 = self.pool1(x, edge_index, batch=batch)
        # detect whether isolated nodes exist

        x1 = self.gap1(x, batch1)

        x, edge_index = self.dgcn2(x, batch1)
        x, edge_index, edge_weight, batch2, perm2, score2 = self.pool2(x, edge_index, batch=batch1)


        x2 = self.gap2(x, batch2)

        x = x1 + x2#  + x3
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

