import os.path as osp
import pandas as pd
import shutil
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

from utils_dataset import graph_from_series, graph_from_series_inverted, graph_from_series_ABIDE


def delete_preprocessed(root):
    print('Deleting processed dir')
    try:
        shutil.rmtree(osp.join(root, 'processed'))
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

class UKBBDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, is_sex=True, threshold=0.02, delete=True):
        if delete:
            delete_preprocessed(root)
        print('Loading UKBB dataset with threshold:', threshold)
        self.is_sex = is_sex
        self.threshold = threshold
        super(UKBBDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # should return a list instead of a string, fix the error of getting the number of graphs
        return ['graph_ukbb.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # load the ukbb data as pandas dataframe
        df = pd.read_excel(osp.join(self.root, 'UKBB2_playground.xlsx'), sheet_name='feature_id')

        # numpy array with the underlying data of the DataFrame
        ukbb_numpy = df.values

        # labels of patients
        if self.is_sex:
            # load the sex labels
            node_labels = torch.LongTensor(ukbb_numpy[:, 1])
        else:
            # load the age labels
            # node_labels = torch.unsqueeze(torch.FloatTensor(2015 - ukbb_numpy[:, 2]), dim=1)
            node_labels = torch.unsqueeze(torch.FloatTensor(1953 - ukbb_numpy[:, 2]), dim=1) # zero mean

        # extract patient features
        # 14503 x 441 => 14503 x 437
        patient_features = ukbb_numpy[:, 4:]  # ignore first four columns. WHAT IS 4th COLUMN?

        # replace `NaN` feature values by mean for each patient
        patient_features = np.where(
            np.isnan(patient_features),
            np.ma.array(patient_features, mask=np.isnan(patient_features)).mean(axis=0),
            patient_features)
        # ensure no more `NaN` values
        assert np.isnan(patient_features).all() == False

        # normalize the features for each patient, use l2 norm
        patient_features = patient_features / np.linalg.norm(patient_features, axis=0, keepdims=True)
        # verify normalization
        assert np.linalg.norm(patient_features, axis=0).all() <= 1

        # compute correlation matrix as L2 distance between features
        correl_l2 = euclidean_distances(patient_features, patient_features)
        # check if diagonal elements are 0
        assert np.diagonal(correl_l2).all() == 0

        # build binary adjacency matrix
        # larger l2 distance means weaker connectivity
        adjacency_mat = np.where((correl_l2 < self.threshold), 1, 0)

        # convert adj. matrix to sparse matrix
        adjacency_mat = sparse.csr_matrix(adjacency_mat).tocoo()
        scr_nodes = torch.from_numpy(adjacency_mat.row.astype(np.int64)).to(torch.long)
        tgt_nodes = torch.from_numpy(adjacency_mat.col.astype(np.int64)).to(torch.long)
        # stack scr_nodes and tgt_nodes to obtain the edge_indices
        edge_indices = torch.stack([scr_nodes, tgt_nodes], dim=0)

        # indices of edges in the graph
        """
        src_nodes, tgt_nodes = np.nonzero(adjacency_mat)
        assert src_nodes.shape == tgt_nodes.shape
        edge_indices = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
        """

        # splitting the features by ROW WISE(no. of patients)
        patient_features = np.split(patient_features, patient_features.shape[0])
        node_features = torch.tensor(patient_features, dtype=torch.float)
        node_features = node_features.squeeze(1)

        # creating a data object
        data = Data(x=node_features,
                    edge_index=edge_indices,
                    y=node_labels)

        # define train mask
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[:int(0.8 * data.num_nodes)] = 1  # train only on the 80% nodes
        # define test mask
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # test on 20 % nodes
        data.test_mask[- int(0.2 * data.num_nodes):] = 1

        # checking for transformations if any
        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, 'graph_ukbb.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_ukbb.pt'))
        return data


class HCPDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, is_sex=True, threshold=0.5, delete=False, bin=True):
        if delete:
            delete_preprocessed(root)
        print('Loading HCP')
        self.is_sex = is_sex
        self.threshold = threshold
        # use weighted edge or binary edge
        self.bin = bin
        super(HCPDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        processed_file_names = []
        for i in range(1003):
            processed_file_names.append('graph_{}_{}.pt'.format('sex' if self.is_sex else 'age',i))
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        hcp_node_timeseries = np.load(osp.join(self.root, 'hcp/HCP_node_timeseries_data.npy'), allow_pickle=True)
        hcp_timeseries_labels = np.load(osp.join(self.root, 'hcp/HCP_node_timeseries_labelsdata.npy'), allow_pickle=True)
        graph_labels = hcp_timeseries_labels[:, 1 if self.is_sex else 2] - 1
        # 1003 x 4800 x 15 => 1003 x 15 x 4800
        timeseries = np.swapaxes(hcp_node_timeseries, axis1=1, axis2=2)
        # after map the correlation to [0,1], compute the corresponding threshold
        thres0 = (self.threshold+1)/2
        thres1 = (-self.threshold+1)/2

        for sub, ts in enumerate(timeseries):
            data = graph_from_series(graph_labels[sub], ts, self.threshold)
            # checking for transformations if any
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'graph_{}_{}.pt'.format('sex' if self.is_sex else 'age', sub)))

        print('Processed data')


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_{}_{}.pt'.format('sex' if self.is_sex else 'age',idx)))
        return data

import random
class HCPDataset_Age_Reduced(Dataset):
    '''
    This is a dataset with the age with binary classification. Uses correlation matrix
    '''
    def __init__(self, root, transform=None, pre_transform=None, is_sex=False, threshold=0.5, delete=False):
        if delete:
            delete_preprocessed(root)
        print('Loading HCP')
        assert(not is_sex)

        self.threshold = threshold
        self.hcp_timeseries_labels = np.load(osp.join(root, 'hcp/HCP_node_timeseries_labelsdata.npy'),
                                        allow_pickle=True)
        self.graph_labels = self.hcp_timeseries_labels[:, 2] - 2 # -1, 0, 1, 2

        #merge first two buckets and last two
        self.graph_labels[self.graph_labels == 2] = 1
        self.graph_labels[self.graph_labels == -1] = 0

        # with the upper merging this part is unneceary generally
        self.mapped_indices = (self.graph_labels == 0).nonzero()[0]
        self.mapped_indices = np.append(self.mapped_indices, (self.graph_labels == 1).nonzero()[0])
        #random.shuffle(self.indices)

        super(HCPDataset_Age_Reduced, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        processed_file_names = []
        for i in range(1003): # 776
            processed_file_names.append('graph_{}_{}.pt'.format('age', i))
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        hcp_node_timeseries = np.load(osp.join(self.root, 'hcp/HCP_node_timeseries_data.npy'), allow_pickle=True)
        # 1003 x 4800 x 15 => 1003 x 15 x 4800
        timeseries = np.swapaxes(hcp_node_timeseries, axis1=1, axis2=2)

        for sub, ts in enumerate(timeseries):
            data = graph_from_series(self.graph_labels[sub], ts, self.threshold)
            # checking for transformations if any
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'graph_{}_{}.pt'.format('age', sub)))

        print('Processed data')


    def len(self):
        return len(self.mapped_indices)

    def get(self, idx):
        mapped_idx = self.mapped_indices[idx]
        data = torch.load(osp.join(self.processed_dir, 'graph_{}_{}.pt'.format('age', mapped_idx)))
        return data


class HCPDataset_Timepoints(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, is_sex=True, threshold=0.5, num_cuts=4, in_memory=True, num_nodes=200, is_regression=False):
        print('Loading HCP into timepoints')
        self.is_sex = is_sex
        if is_sex and is_regression:
            raise Exception
        self.threshold = threshold
        hcp_timeseries_labels = np.load('data/hcp/HCP_node_timeseries_labelsdata.npy', allow_pickle=True)
        self.labels = hcp_timeseries_labels[:, 1 if self.is_sex else 2] - 1
        self.num_cuts = num_cuts
        self.num_nodes = num_nodes

        self.in_memory = in_memory
        self.graphs = []
        super(HCPDataset_Timepoints, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        processed_file_names = []
        for i in range(1003):
            processed_file_names.append('timeseries_{}.npy'.format(i))
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        hcp_node_timeseries = np.load(osp.join(self.root, f'HCP_{self.num_nodes}_nodes_timeseries_data.npy'), allow_pickle=True).astype(np.float)
        hcp_netmats = np.load(osp.join(self.root, f'HCP_{self.num_nodes}_nodes_netmats.npy'), allow_pickle=True).astype(np.float)

        #hcp_netmats -= hcp_netmats.mean(axis=0) # mean matrix please
        #std = hcp_netmats.std(axis=0) # 50x50
        #hcp_netmats = np.divide(hcp_netmats, std, out=np.zeros_like(hcp_netmats), where=std != 0)

        # 1003 x 4800 x 15 => 1003 x 15 x 4800
        timeseries = np.swapaxes(hcp_node_timeseries, axis1=1, axis2=2)
        if self.in_memory:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #self.timeseries = torch.tensor(timeseries)
            #self.timeseries.to(device)

            step = int(4800 / self.num_cuts)
            for sub, ts in enumerate(timeseries):
                individual_graphs = [graph_from_series(self.labels[sub], ts[:, i * step: (i + 1) * step], self.threshold, hcp_netmats[sub]).to(device) for i in range(self.num_cuts)]
                self.graphs.append(individual_graphs)

        else:
            for sub, ts in enumerate(timeseries):
                np.save(osp.join(self.processed_dir, 'timeseries_{}.npy'.format(sub)), ts, allow_pickle=True)
        print('Processed data')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # 15 x 4800
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.in_memory:
            return self.graphs[idx]
        else:
            raise NotImplementedError
            ts = np.load(osp.join(self.processed_dir, 'timeseries_{}.npy'.format(idx)), allow_pickle=True)
            ts[:, 0:1200] = ts[:, 0:1200][::-1]
            # convert second RLtoLR
            ts[:, 2400:3600] = ts[:, 2400:3600][::-1]

            step = int(4800 / self.num_cuts)
            graphs = [graph_from_series_inverted(self.labels[idx], ts[:, i*step: (i+1)*step], self.threshold, netmap=hcp_netmats[sub]).to(device) for i in range(self.num_cuts)]
        return graphs


import ABIDEParser
class ABIDE(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, label='autism', threshold=0.5, num_cuts=4, in_memory=True, num_nodes=111):

        self.threshold = threshold

        #hcp_timeseries_labels = np.load('data/hcp/HCP_node_timeseries_labelsdata.npy', allow_pickle=True)
        #self.labels = hcp_timeseries_labels[:, 1 if self.is_sex else 2] - 1
        self.num_cuts = num_cuts
        self.num_nodes = num_nodes
        if num_nodes != 111:
            raise IndexError

        self.in_memory = in_memory
        self.graphs = []

        # Get class labels
        self.subject_IDs = ABIDEParser.get_ids()

        label_id = \
            'SEX' if label == 'sex' else \
            'DX_GROUP' if label == 'autism' else \
            'AGE_AT_SCAN' if label == 'age' else ''
        self.labels = np.array([(int(l) - 1) if not label == 'age' else float(l) for l in
                                list(ABIDEParser.get_subject_score(self.subject_IDs, score=label_id).values())]) # what's this?

        super(ABIDE, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        processed_file_names = []
        for i in range(196):
            processed_file_names.append('timeseries_{}.npy'.format(i))
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we have only the atlas ho. 111 are the nodes
        subjects_timeseries = ABIDEParser.get_timeseries(self.subject_IDs, 'ho') # [(196, 111)]

        for sub, ts in enumerate(subjects_timeseries):
            step = int(ts.shape[0] / self.num_cuts)

            full_connectivity_matrix = ABIDEParser.subject_connectivity(ts, self.subject_IDs[sub], 'ho', 'correlation', save=False) # 111x111
            full_connectivity_matrix = np.arctanh(full_connectivity_matrix - np.eye(full_connectivity_matrix.shape[0]))
            full_connectivity_matrix /= full_connectivity_matrix.std()

            # assertion
            # assert not np.isnan(np.sum(full_connectivity_matrix))

            individual_graphs = [graph_from_series_ABIDE(self.labels[sub],
                                                         ts[i * step : (i + 1) * step, :],
                                                         self.threshold,
                                                         full_connectivity_matrix).to(device) for i in range(self.num_cuts)]
            self.graphs.append(individual_graphs)


        pass

    def len(self):
        return len(self.subject_IDs)

    def get(self, idx):
        return self.graphs[idx]

class HCPDataset_Raw(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, is_sex=True,  in_memory=True, num_cuts=32, num_nodes=200):
        print('Loading HCP into timepoints')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_nodes = num_nodes
        self.is_sex = is_sex
        hcp_timeseries_labels = np.load('data/hcp/HCP_node_timeseries_labelsdata.npy', allow_pickle=True)

        if self.is_sex:
            self.labels = torch.tensor(hcp_timeseries_labels[:, 1] - 1, dtype=torch.long).to(self.device)
            self.mapped_indices = torch.tensor(np.arange(1003))
        else:
            self.labels = torch.tensor(hcp_timeseries_labels[:, 2] - 1, dtype=torch.long).to(self.device)
            self.mapped_indices = torch.tensor(np.arange(1003))

            '''
            graph_labels = hcp_timeseries_labels[:, 2] - 2
            # merge first two buckets and last two
            graph_labels[graph_labels == 2] = 1
            graph_labels[graph_labels == -1] = 0

            # with the upper merging this part is unneceary generally
            self.mapped_indices = (graph_labels == 0).nonzero()[0]
            self.mapped_indices = self.mapped_indices[0:350] # take only the first 350
            self.mapped_indices = np.append(self.mapped_indices, (graph_labels == 1).nonzero()[0])
            # random.shuffle(self.indices)

            print(f'Data points of class "young": {len(self.mapped_indices[graph_labels[self.mapped_indices] == 0])}')
            print(f'Data points of class "old": {len(self.mapped_indices[graph_labels[self.mapped_indices] == 1])}')

            self.labels = torch.tensor(graph_labels, dtype=torch.long).to(self.device)
            '''

        self.num_cuts = num_cuts

        self.in_memory = in_memory
        self.timeseries = []
        super(HCPDataset_Raw, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        processed_file_names = []
        for i in range(1003):
            processed_file_names.append('timeseries_{}.npy'.format(i))
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        hcp_node_timeseries = np.load(f'data/hcp/HCP_{self.num_nodes}_nodes_timeseries_data.npy', allow_pickle=True).astype(np.float)
        # 1003 x 4800 x 15 => 1003 x 15 x 4800
        timeseries = np.swapaxes(hcp_node_timeseries, axis1=1, axis2=2)

        # Normalize to 0-mean std var
        timeseries = timeseries - timeseries.mean(axis=0)
        timeseries = timeseries / timeseries.std()

        # timeseries -= timeseries.min()
        # timeseries /= timeseries.max()

        if self.in_memory:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            step = int(4800 / self.num_cuts)
            for sub, ts in enumerate(timeseries):
                ts[:, 0:1200] = ts[:, 0:1200][::-1]
                # convert second RLtoLR
                ts[:, 2400:3600] = ts[:, 2400:3600][::-1]
                cut_series = [
                    torch.FloatTensor(ts[:, i * step: (i + 1) * step]).to(device)
                    for i in range(self.num_cuts)]

                self.timeseries.append(cut_series)

        else:
            for sub, ts in enumerate(timeseries):
                np.save(osp.join(self.processed_dir, 'timeseries_{}.npy'.format(sub)), ts, allow_pickle=True)
        print('Processed data')

    def len(self):
        return len(self.mapped_indices)

    def get(self, idx):
        mapped_idx = self.mapped_indices[idx]

        # 15 x 4800
        if self.in_memory:
            return [self.timeseries[mapped_idx], self.labels[mapped_idx]]
        else:
            raise NotImplementedError




if __name__ == "__main__":
    d = ABIDE(root='data/')
    # UKBBDataset(root='data/', is_sex=False)