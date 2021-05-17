import torch
import numpy as np
from torch_geometric.data import Data


def graph_from_series_ABIDE(label, ts, threshold, full_matrix):
    #conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    #correl_mat = conn_measure.fit_transform([ts])[0]

    ts = np.swapaxes(ts, axis1=0, axis2=1)
    correl_mat = np.corrcoef(ts)

    node_features = torch.tensor(full_matrix, dtype=torch.float)

    # keep the weight if the absolute value of correlation is larger than the threshold, else discard
    correl_mat -= np.eye(correl_mat.shape[0])
    # correl_mat = np.abs(correl_mat)
    adjacency_mat = np.where((correl_mat > threshold), correl_mat, 0)

    # indices of edges in the graph
    src_nodes, tgt_nodes = np.nonzero(adjacency_mat)
    assert src_nodes.shape == tgt_nodes.shape
    edge_indices = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    # print(f'Num of edges: {edge_indices.shape[1]}')
    # for 2. method, need edge_attr in Data(), [num_edges, dim_edge_features]
    # convert to tensor
    adjacency_mat = torch.tensor(adjacency_mat, dtype=torch.float)
    edge_attr = adjacency_mat[adjacency_mat.nonzero(as_tuple=True)]
    assert edge_indices.shape[1] == edge_attr.shape[0]

    # edge_attr = torch.unsqueeze(edge_attr, dim=1).to(torch.float)
    # creating a data object
    data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attr, y=torch.tensor(int(label), dtype=torch.long))
    return data


def graph_from_series(label, ts, threshold, netmap):
    # converting timeseries str => float

    ts = np.array(ts).astype(np.float)

    # correlation
    correl_mat = np.corrcoef(ts)

    # debug
    #ts2 = np.swapaxes(ts, axis1=0, axis2=1)
    #conn_measure = connectome.ConnectivityMeasure(kind="partial correlation")
    #correl_mat_lib = conn_measure.fit_transform([ts2])[0]

    # 1. use binary adjacency matrix, nodes features are from correl_mat
    """
                # node feature matrix of the graph
                features = np.split(correl_mat, correl_mat.shape[1])
                node_features = torch.tensor(features, dtype=torch.float)
                node_features = node_features.squeeze(1)
                # binary adjacency matrix, use absolute value????
                adjacency_mat = np.where((correl_mat > self.threshold), 1, 0)
                edge_attr = None
                """

    # 2. use weighted adjacency matrix, node features are one-hot embedding (representing node's location info)

    # node_features = torch.eye(correl_mat.shape[1], dtype=torch.float)
    node_features = torch.tensor(netmap, dtype=torch.float)

    # keep the weight if the absolute value of correlation is larger than the threshold, else discard
    correl_mat -= np.eye(correl_mat.shape[0])
    # correl_mat = np.abs(correl_mat)
    adjacency_mat = np.where((correl_mat > threshold), correl_mat, 0)

    # indices of edges in the graph
    src_nodes, tgt_nodes = np.nonzero(adjacency_mat)
    assert src_nodes.shape == tgt_nodes.shape
    edge_indices = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)

    # print(f'Num of edges: {edge_indices.shape[1]}')
    # for 2. method, need edge_attr in Data(), [num_edges, dim_edge_features]
    # convert to tensor
    adjacency_mat = torch.tensor(adjacency_mat, dtype=torch.float)
    edge_attr = adjacency_mat[adjacency_mat.nonzero(as_tuple=True)]
    assert edge_indices.shape[1] == edge_attr.shape[0]

    # edge_attr = torch.unsqueeze(edge_attr, dim=1).to(torch.float)
    # creating a data object
    data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long))
    return data


def graph_from_series_inverted(label, ts, threshold, netmap):
    # converting timeseries str => float

    ts = np.array(ts).astype(np.float)

    # correlation
    correl_mat = np.corrcoef(ts)
    # 1. use binary adjacency matrix, nodes features are from correl_mat
    """
                # node feature matrix of the graph
                features = np.split(correl_mat, correl_mat.shape[1])
                node_features = torch.tensor(features, dtype=torch.float)
                node_features = node_features.squeeze(1)
                # binary adjacency matrix, use absolute value????
                adjacency_mat = np.where((correl_mat > self.threshold), 1, 0)
                edge_attr = None
                """

    # 2. use weighted adjacency matrix, node features are one-hot embedding (representing node's location info)



    # keep the weight if the absolute value of correlation is larger than the threshold, else discard
    correl_mat -= np.eye(correl_mat.shape[0])
    # correl_mat = np.abs(correl_mat)
    adjacency_mat = np.where((netmap > threshold), correl_mat, 0)

    # node_features = torch.eye(correl_mat.shape[1], dtype=torch.float)
    # node_features = torch.tensor(netmap, dtype=torch.float)
    node_features = torch.tensor(correl_mat, dtype=torch.float)

    # indices of edges in the graph
    src_nodes, tgt_nodes = np.nonzero(adjacency_mat)
    assert src_nodes.shape == tgt_nodes.shape
    edge_indices = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    # for 2. method, need edge_attr in Data(), [num_edges, dim_edge_features]
    # convert to tensor
    adjacency_mat = torch.tensor(adjacency_mat, dtype=torch.float)
    edge_attr = adjacency_mat[adjacency_mat.nonzero(as_tuple=True)]
    assert edge_indices.shape[1] == edge_attr.shape[0]

    # edge_attr = torch.unsqueeze(edge_attr, dim=1).to(torch.float)
    # creating a data object
    data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long))
    return data

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.unsqueeze(1).expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def ConvertRLtoLR(series):
    # convert first RLtoLR
    series[:, 0:1200] = series[:, 0:1200][::-1]
    # convert second RLtoLR
    series[:, 2400:3600] = series[:, 2400:3600][::-1]
    return series

def graph_from_series_torch(label, ts, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # correlation
    correl_mat = corrcoef(ts.squeeze())
    # 1. use binary adjacency matrix, nodes features are from correl_mat
    """
                # node feature matrix of the graph
                features = np.split(correl_mat, correl_mat.shape[1])
                node_features = torch.tensor(features, dtype=torch.float)
                node_features = node_features.squeeze(1)
                # binary adjacency matrix, use absolute value????
                adjacency_mat = np.where((correl_mat > self.threshold), 1, 0)
                edge_attr = None
                """

    # 2. use weighted adjacency matrix, node features are one-hot embedding (representing node's location info)
    node_features = torch.eye(correl_mat.shape[1], dtype=torch.float).to(device)

    # keep the weight if the absolute value of correlation is larger than the threshold, else discard
    correl_mat = correl_mat - torch.eye(correl_mat.shape[0]).to(device)
    # correl_mat = np.abs(correl_mat)

    correl_mat[(correl_mat < threshold)] = 0
    adjacency_mat = correl_mat

    # indices of edges in the graph
    edge_indices = torch.nonzero(adjacency_mat)
    # for 2. method, need edge_attr in Data(), [num_edges, dim_edge_features]
    # convert to tensor
    edge_attr = adjacency_mat[adjacency_mat.nonzero(as_tuple=True)]

    # edge_attr = torch.unsqueeze(edge_attr, dim=1).to(torch.float)
    # creating a data object

    #data = Data(x=node_features, edge_index=edge_indices.T, edge_attr=edge_attr, y=torch.tensor(label))
    return node_features, edge_indices.T, edge_attr


