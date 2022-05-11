from pdb2sql import pdb2sql
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools.pdb import get_residue_contact_pairs
from deeprank_gnn.domain.amino_acid import alanine


def test_add_pssm():
    residue_pairs = get_residue_contact_pairs("tests/data/pdb/1ATN/1ATN_1w.pdb",
                                              "A", "B", 10.0)
    for residue in residue_pairs[0]:
        chain = residue.chain
        with open("tests/data/pssm/1ATN/1ATN.{}.pdb.pssm".format(chain.id), 'rt') as f:
            chain.pssm = parse_pssm(f, chain)

    # Verify that each residue is present and that the data makes sense:
    for residue1, residue2 in residue_pairs:
        chain1 = residue1.chain
        chain2 = residue2.chain

        assert residue1 in chain1.pssm
        assert residue2 in chain2.pssm

        assert isinstance(chain1.pssm[residue1].information_content, float)
        assert isinstance(chain2.pssm[residue2].information_content, float)

        assert isinstance(chain1.pssm[residue1].conservations[alanine], float)
        assert isinstance(chain2.pssm[residue2].conservations[alanine], float)
