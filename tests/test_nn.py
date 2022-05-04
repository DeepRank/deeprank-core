import tempfile
import shutil
import os

import unittest

from deeprank_gnn.tools.CustomizeGraph import add_target
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.sGAT import sGAT


def _model_base_test(work_directory, hdf5_path, model_class,
                     node_features, edge_features,
                     task, target):

    NN = NeuralNet(hdf5_path, model_class,
                   node_feature=node_features,
                   edge_feature=edge_features,
                   target=target,
                   index=None,
                   task=task,
                   batch_size=64,
                   percent=[0.8, 0.2],
                   outdir=work_directory)

    NN.train(nepoch=1, validate=True)

    NN.save_model('test.pth.tar')

    NN_cpy = NeuralNet(hdf5_path, model_class,
                       pretrained_model='test.pth.tar')


class TestNeuralNet(unittest.TestCase):

    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_ginet(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", GINet,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd')

    def test_ginet_class(self):
        _model_base_test(self.work_directory, "tests/hdf5/variants.hdf5", GINet,
                         ['size', 'polarity', 'sasa', 'ic', 'pssm'], ['dist'],
                         'class', 'bin_class')

    def test_fout(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", FoutNet,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd')

    def test_sgat(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", sGAT,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd')


if __name__ == "__main__":
    unittest.main()
