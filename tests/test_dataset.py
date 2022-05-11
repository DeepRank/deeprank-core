import unittest
from deeprank_gnn.DataSet import HDF5DataSet


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.database = "tests/hdf5/1ATN_ppi.hdf5"

    def test_dataset(self):
        HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            index=None,
        )

    def test_dataset_filter(self):
        HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            index=None,
            dict_filter={"irmsd": "<10"},
        )


if __name__ == "__main__":
    unittest.main()
