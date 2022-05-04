import os

from pdb2sql import StructureSimilarity


def get_all_scores(pdb_path, reference_pdb_path):
    """Assigns scores (lrmsd, irmsd, fnat, dockQ, bin_class, capri_class) to a protein graph

    Args:
        pdb_path (path): path to the scored pdb structure
        reference_pdb_path (path): path to the reference structure required to compute the different score
    """

    ref_name = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    sim = StructureSimilarity(pdb_path, reference_pdb_path)

    scores = {}

    # Input pre-computed zone files
    if os.path.exists(ref_name + ".lzone"):
        scores["lrmsd"] = sim.compute_lrmsd_fast(
            method="svd", lzone=ref_name + ".lzone"
        )
        scores["irmsd"] = sim.compute_irmsd_fast(
            method="svd", izone=ref_name + ".izone"
        )

    # Compute zone files
    else:
        scores["lrmsd"] = sim.compute_lrmsd_fast(method="svd")
        scores["irmsd"] = sim.compute_irmsd_fast(method="svd")

    scores["fnat"] = sim.compute_fnat_fast()
    scores["dockQ"] = sim.compute_DockQScore(
        scores["fnat"], scores["lrmsd"], scores["irmsd"]
    )
    scores["bin_class"] = scores["irmsd"] < 4.0

    scores["capri_class"] = 5
    for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
        if scores["irmsd"] < thr:
            scores["capri_class"] = val

    return scores
