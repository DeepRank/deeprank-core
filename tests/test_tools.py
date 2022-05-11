import tempfile
import os
import shutil
import unittest
import numpy as np
from deeprank_gnn.tools.pssm_3dcons_to_deeprank import pssm_3dcons_to_deeprank
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv
from deeprank_gnn.tools.CustomizeGraph import add_target
from deeprank_gnn.tools.embedding import manifold_embedding


class TestTools(unittest.TestCase):
    def setUp(self):

        self.pdb_path = "./tests/data/pdb/1ATN/"
        self.pssm_path = "./tests/data/pssm/1ATN/1ATN.A.pdb.pssm"
        self.ref = "./tests/data/ref/1ATN/"
        self.h5_train_ref = "tests/data/train_ref/train_data.hdf5"

        self.h5_graphs = "tests/hdf5/1ATN_ppi.hdf5"

    def test_pssm_convert(self):
        pssm_3dcons_to_deeprank(self.pssm_path)

    def test_h52csv(self):
        hdf5_to_csv(self.h5_train_ref)

    def test_add_target(self):

        f, target_path = tempfile.mkstemp(prefix="target", suffix=".lst")
        os.close(f)

        f, graph_path = tempfile.mkstemp(prefix="1ATN_ppi", suffix=".hdf5")
        os.close(f)

        try:
            target_list = ""
            for i in range(1, 11):
                target_list += f"1ATN_{i}w {i}\n"

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(target_list)

            shutil.copy(self.h5_graphs, graph_path)

            add_target(graph_path, "test_target", target_path)
        finally:
            os.remove(target_path)
            os.remove(graph_path)

    def test_embeding(self):
        pos = np.random.rand(110, 3)
        for method in ["tsne", "spectral", "mds"]:
            _ = manifold_embedding(pos, method=method)


if __name__ == "__main__":
    unittest.main()
