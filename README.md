# MLMA WS20 Brain Signal Analysis 

This is the code to the paper "DG-GRU: Dynamic Graph based GatedRecurrent Unit for age and gender predictionusing Brain Imaging". 

Frameworks: 
- PyTorch
- Torch Geometric

Datasets:
- HCP
- ABIDE
- UKBB (different setting)

The data is expected in a folder in the repository. The folder structure for HCP should read as following: 

<repository>/data/hcp/HCP_node_timeseries_labelsdata.npy
<repository>/data/hcp/HCP<number of nodes>_nodes_timeseries_data.npy
<repository>/data/hcp/HCP<number of nodes>_nodes_netmats.npy