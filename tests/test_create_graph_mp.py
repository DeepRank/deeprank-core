import tempfile
import shutil
import os

import unittest
from deeprank_gnn.GraphGenMP import GraphHDF5


class TestCreateGraph(unittest.TestCase):

    def setUp(self):

        self.pdb_path = 'tests/data/pdb/3C8P/'
        self.pssm_path = './tests/data/pssm/3C8P/'
        self.ref = './tests/data/ref/3C8P/'

        self._output_files = []
        self._work_directories = []

    def tearDown(self):
        for file_path in self._output_files:
            os.remove(file_path)

        for dir_path in self._work_directories:
            shutil.rmtree(dir_path)

    def _make_work_directory(self):
        path = tempfile.mkdtemp()
        self._work_directories.append(path)
        return path

    def _make_output_file(self):
        f, path = tempfile.mkstemp(suffix=".hdf5")
        os.close(f)
        self._output_files.append(path)
        return path

    def test_create_serial_with_bio(self):
        GraphHDF5(pdb_path=self.pdb_path, ref_path=self.ref, pssm_path=self.pssm_path,
                  graph_type='residue', outfile=self._make_output_file(),
                  nproc=2, tmpdir=self._make_work_directory(), biopython=True)

    def test_create_serial(self):
        GraphHDF5(pdb_path=self.pdb_path, ref_path=self.ref, pssm_path=self.pssm_path,
                  graph_type='residue', outfile=self._make_output_file(),
                  nproc=2, tmpdir=self._make_work_directory(), biopython=False)


if __name__ == "__main__":
    unittest.main()
