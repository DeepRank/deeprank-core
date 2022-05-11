# DeepRank-GNN


[![Build Status](https://github.com/DeepRank/DeepRank-GNN/workflows/build/badge.svg)](https://github.com/DeepRank/DeepRank-GNN/actions)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f3f98b2d1883493ead50e3acaa23f2cc)](https://app.codacy.com/gh/DeepRank/DeepRank-GNN?utm_source=github.com&utm_medium=referral&utm_content=DeepRank/DeepRank-GNN&utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/DeepRank/deeprank-gnn-2/badge.svg?branch=main)](https://coveralls.io/github/DeepRank/deeprank-gnn-2?branch=main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5705564.svg)](https://doi.org/10.5281/zenodo.5705564)

![alt-text](./deeprank_gnn.png)

## Installation

### Dependencies

Before installing DeepRank-GNN you need to install:

 * [pytorch](https://pytorch.org/): `conda install pytorch -c pytorch`. Note that by default the CPU version of pytorch will be installed, but you can also customize that installation following the instructions on pytorch website.
 * [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg` (recommended).
 * [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) `conda install pytorch-cluster -c pyg`
 * [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) `conda install pytorch-sparse -c pyg`
 * [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) `conda install pytorch-scatter -c pyg`
 * [torch_spline_conv](https://github.com/rusty1s/pytorch_spline_conv) `conda install pytorch-spline-conv -c pyg`
 * [numpy](https://numpy.org) `conda install numpy`
 * [scipy](https://scipy.org) `conda install -c anaconda scipy`
 * [h5py](https://docs.h5py.org) `conda install -c anaconda h5py`
 * [networkx](https://networkx.org) `conda install -c anaconda networkx`
 * [matplotlib](https://matplotlib.org) `conda install -c conda-forge matplotlib`
 * [pdb2sql](https://pdb2sql.readthedocs.io) `pip install pdb2sql`
 * [sklearn](https://scikit-learn.org) `conda install -c anaconda scikit-learn`
 * [chart-studio](https://help.plot.ly) `conda install -c conda-forge chart-studio`
 * [BioPython](https://biopython.org) `conda install -c conda-forge biopython`
 * [python-louvain](https://github.com/taynaud/python-louvain) `conda install -c conda-forge python-louvain`
 * [markov-clustering](https://github.com/guyallard/markov_clustering) `pip install markov-clustering`
 * [tqdm](https://pypi.python.org/pypi/tqdm) `conda install -c conda-forge tqdm`
 * [freesasa](https://github.com/mittinatten/freesasa) `conda install -c hydroid freesasa`
 * [msms](https://ssbio.readthedocs.io/en/latest/instructions/msms.html): `conda install -c bioconda msms`. *For MacOS with M1 chip users*: you can follow [these instructions](https://ssbio.readthedocs.io/en/latest/instructions/msms.html). 
 * [reduce](https://github.com/rlabduke/reduce) Follow the instructions in the README

### DeepRank-GNN installation

[//]: # (Once the dependencies installed, you can install the latest release of DeepRank-GNN using the PyPi package manager:)

[//]: # (```)
[//]: # (pip install DeepRank-GNN)
[//]: # (```)

You can get all the new developments by cloning the repo and installing the code with

```
git clone https://github.com/DeepRank/deeprank-gnn-2
cd deeprank-gnn-2
pip install -e ./
```

[//]: # (The documentation can be found here : https://deeprank-gnn.readthedocs.io/)

## Generate Graphs

The process of generating graphs is called preprocessing. In order to do so, one needs query objects, describing how the graphs should be built.

```python
from deeprank_gnn.preprocess import PreProcessor
from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.feature import bsa, pssm, amino_acid, biopython

feature_modules = [bsa, pssm, amino_acid, biopython]

preprocessor = PreProcessor(feature_modules, "train-data")

preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='1ATN_1w.pdb', chain_id1="A", chain_id2="B",
                                                           pssm_paths={"A": "1ATN.A.pdb.pssm", "B": "1ATN.B.pdb.pssm"})
preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='1ATN_2w.pdb', chain_id1="A", chain_id2="B",
                                                           pssm_paths={"A": "1ATN.A.pdb.pssm", "B": "1ATN.B.pdb.pssm"})
preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='1ATN_3w.pdb', chain_id1="A", chain_id2="B",
                                                           pssm_paths={"A": "1ATN.A.pdb.pssm", "B": "1ATN.B.pdb.pssm"})
preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='1ATN_4w.pdb', chain_id1="A", chain_id2="B",
                                                           pssm_paths={"A": "1ATN.A.pdb.pssm", "B": "1ATN.B.pdb.pssm"})

preprocessor.start()  # start builfing graphs from the queries

preprocessor.wait()  # wait for all jobs to complete

print(preprocessor.output_paths)  # print the paths of the generated files

```

The user is free to implement his/her own query class. Each implementation requires the `build_graph` method to be present.


## Graph Interaction Network

Using the graph interaction network is rather simple :


```python
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.models.metrics import OutputExporter, ScatterPlotExporter

database = './hdf5/1ACB_residue.hdf5'

metrics_output_directory = "./metrics"
metrics_exporters = [OutputExporter(metrics_output_directory),
                     ScatterPlotExporter(metrics_output_directory, 5)]

nn = NeuralNet(database, GINet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2],
               metrics_exporters=metrics_exporters)

nn.train(nepoch=50, validate=False)
```

## Custom GNN

It is also possible to define new network architecture and to specify the loss and optimizer to be used during the training.

```python


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class CustomNet(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(d.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        data = max_pool(cluster, data)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = scatter_mean(x, batch, dim=0)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nn = NeuralNet(database, CustomNet,
               node_feature=['type', 'polarity', 'bsa',
                             'depth', 'hse', 'ic', 'pssm'],
               edge_feature=['dist'],
               target='irmsd',
               index=range(400),
               batch_size=64,
               percent=[0.8, 0.2],
               device=device)
nn.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
nn.loss = MSELoss()

nn.train(nepoch=50)
```

## h5x support

After installing  `h5xplorer`  (https://github.com/DeepRank/h5xplorer), you can execute the python file `deeprank_gnn/h5x/h5x.py` to explorer the connection graph used by DeepRank-GNN. The context menu (right click on the name of the structure) allows to automatically plot the graphs using `plotly` as shown below.

![alt-text](./h5_deeprank_gnn.png)
