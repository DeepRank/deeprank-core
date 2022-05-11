import tempfile
import shutil
import os
import unittest
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.sGAT import sGAT
from deeprank_gnn.models.metrics import (
    OutputExporter,
    TensorboardBinaryClassificationExporter,
)


def _model_base_test( # pylint: disable=too-many-arguments
    hdf5_path,
    model_class,
    node_features,
    edge_features,
    task,
    target,
    metrics_exporters,
):

    NN = NeuralNet(
        hdf5_path,
        model_class,
        node_feature=node_features,
        edge_feature=edge_features,
        target=target,
        index=None,
        task=task,
        batch_size=64,
        percent=[0.8, 0.2],
        metrics_exporters=metrics_exporters,
    )

    NN.train(nepoch=10, validate=True)

    NN.save_model("test.pth.tar")

    NeuralNet(hdf5_path, model_class, pretrained_model="test.pth.tar")


class TestNeuralNet(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_ginet(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            GINet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [OutputExporter(self.work_directory)],
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet_class(self):
        _model_base_test(
            "tests/hdf5/variants.hdf5",
            GINet,
            ["size", "polarity", "sasa", "ic", "pssm"],
            ["dist"],
            "class",
            "bin_class",
            [TensorboardBinaryClassificationExporter(self.work_directory)],
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_fout(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            FoutNet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [],
        )

    def test_sgat(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            sGAT,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [],
        )


if __name__ == "__main__":
    unittest.main()
