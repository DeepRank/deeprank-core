from black import main
import numpy

from pdb2sql import pdb2sql
from deeprank_gnn.tools.pdb import get_surrounding_residues


if __name__ == "__main__":
    pdb_path = "tests/data/pdb/2Y69/2Y69.pdb"

    close_residues = get_surrounding_residues(pdb_path, 'E', 74, None, 10.0)
    print(close_residues)