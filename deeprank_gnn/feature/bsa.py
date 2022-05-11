import logging
import freesasa
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.domain.feature import FEATURENAME_BURIEDSURFACEAREA


logging.getLogger(__name__)


def add_features(pdb_path: str, graph: Graph, *args, **kwargs):

    "calculates the buried surface area (BSA): the area of the protein, that only gets exposed in monomeric state"

    sasa_complete_structure = freesasa.Structure() # pylint: disable=c-extension-no-member
    sasa_chain_structures = {}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure() # pylint: disable=c-extension-no-member

            for atom in residue.atoms:
                sasa_chain_structures[chain_id].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                        atom.residue.number, atom.residue.chain.id,
                                                        atom.position[0], atom.position[1], atom.position[2])
                sasa_complete_structure.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                atom.residue.number, atom.residue.chain.id,
                                                atom.position[0], atom.position[1], atom.position[2]) 

        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure() # pylint: disable=c-extension-no-member

            sasa_chain_structures[chain_id].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                    atom.residue.number, atom.residue.chain.id,
                                                    atom.position[0], atom.position[1], atom.position[2])
            sasa_complete_structure.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                            atom.residue.number, atom.residue.chain.id,
                                            atom.position[0], atom.position[1], atom.position[2])

            area_key = "atom"
            selection = (f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})")
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

    sasa_complete_result = freesasa.calc(sasa_complete_structure) # pylint: disable=c-extension-no-member
    sasa_chain_results = {chain_id: freesasa.calc(structure)
                          for chain_id, structure in sasa_chain_structures.items()} # pylint: disable=c-extension-no-member

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            area_key = "residue"
            selection = (f"residue, (resi {residue.number_string}) and (chain {residue.chain.id})")

        elif isinstance(node.id, Atom):
            atom = node.id
            chain_id = atom.residue.chain.id
            area_key = "atom"
            selection = (f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})")

        area_monomer = freesasa.selectArea(selection, \
            sasa_chain_structures[chain_id], sasa_chain_results[chain_id])[area_key] # pylint: disable=c-extension-no-member
        area_multimer = freesasa.selectArea(selection, sasa_complete_structure, sasa_complete_result)[area_key] # pylint: disable=c-extension-no-member

        node.features[FEATURENAME_BURIEDSURFACEAREA] = area_monomer - area_multimer