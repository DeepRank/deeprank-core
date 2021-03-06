from tempfile import mkdtemp
from shutil import rmtree
import os
import h5py
from deeprankcore.preprocess import preprocess
from deeprankcore.models.query import SingleResidueVariantResidueQuery
from deeprankcore.domain.amino_acid import alanine, phenylalanine
import deeprankcore.feature.sasa
from tests.utils import PATH_TEST


def test_preprocess():
    """
    Tests preprocessing several PDB files into their feature representation HDF5 file.

    NB: This test sometimes fails, especially on Mac OSX. The PreProcessor class relies on python multiprocessing
    library's Queue. We need to check whether the queue is empty, which is not always reliable, the only working
    implementation on Mac OSX is really not reliable.
    """

    output_directory = mkdtemp()

    prefix = os.path.join(output_directory, "test-preprocess")

    try:
        count_queries = 10
        queries = []
        for number in range(1, count_queries + 1):
            query = SingleResidueVariantResidueQuery(
                str(PATH_TEST / "data/pdb/101M/101M.pdb"),
                "A",
                number,
                None,
                alanine,
                phenylalanine,
                pssm_paths={"A": str(PATH_TEST / "data/pssm/101M/101M.A.pdb.pssm")},
            )
            queries.append(query)

        output_paths = preprocess([deeprankcore.feature.sasa], queries, prefix, 10)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

    finally:
        rmtree(output_directory)
