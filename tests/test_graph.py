import unittest
import h5py
from deeprank_gnn.tools.visualization import hdf5_to_networkx, plotly_2d, plotly_3d
from deeprank_gnn.tools.score import get_all_scores


class TestGraph(unittest.TestCase):
    def setUp(self):
        with h5py.File("tests/hdf5/1ATN_ppi.hdf5", "r") as f5:
            self.networkx_graph = hdf5_to_networkx(f5["1ATN_1w"])

        self.pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
        self.reference_path = "tests/data/pdb/1ATN/1ATN_2w.pdb"

    def test_score(self):
        get_all_scores(self.pdb_path, self.reference_path)

    def test_plot_2d(self):
        plotly_2d(self.networkx_graph, "1ATN", disable_plot=True)

    def test_plot_3d(self):
        plotly_3d(self.networkx_graph, "1ATN", disable_plot=True)


if __name__ == "__main__":
    unittest.main()
