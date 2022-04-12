from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.polarity import Polarity


alanine = AminoAcid("Alanine", "ALA", "A", charge=-0.37, polarity=Polarity.APOLAR, size=1, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=0)
cysteine = AminoAcid("Cysteine", "CYS", "C", charge=-0.64, polarity=Polarity.POLAR, size=2, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=1)
aspartate = AminoAcid("Aspartate", "ASP", "D", charge=-1.37, polarity=Polarity.NEGATIVE_CHARGE, size=4, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=4, index=2)
glutamate = AminoAcid("Glutamate", "GLU", "E", charge=-1.37, polarity=Polarity.NEGATIVE_CHARGE, size=5, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=4, index=3)
phenylalanine = AminoAcid("Phenylalanine", "PHE", "F", charge=-0.37, polarity=Polarity.APOLAR, size=7, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=4)
glycine = AminoAcid("Glycine", "GLY", "G", charge=-0.37, polarity=Polarity.APOLAR, size=0, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=5)
histidine = AminoAcid("Histidine", "HIS", "H", charge=-0.29, polarity=Polarity.POLAR, size=6, count_hydrogen_bond_donors=2, count_hydrogen_bond_acceptors=2, index=6)
isoleucine = AminoAcid("Isoleucine", "ILE", "I", charge=-0.37, polarity=Polarity.APOLAR, size=4, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=7)
leucine = AminoAcid("Leucine", "LEU", "L", charge=-0.37, polarity=Polarity.APOLAR, size=4, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=8)
lysine = AminoAcid("Lysine", "LYS", "K", charge=-0.36, polarity=Polarity.POSITIVE_CHARGE, size=5, count_hydrogen_bond_donors=3, count_hydrogen_bond_acceptors=0, index=9)
methionine = AminoAcid("Methionine", "MET", "M", charge=-0.37, polarity=Polarity.APOLAR, size=4, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=10)
proline = AminoAcid("Proline", "PRO", "P", charge=0.0, polarity=Polarity.APOLAR, size=3, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=11)
serine = AminoAcid("Serine", "SER", "S", charge=-0.80, polarity=Polarity.POLAR, size=2, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=2, index=12)
threonine = AminoAcid("Threonine", "THR", "T", charge=-0.80, polarity=Polarity.POLAR, size=3, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=2, index=13)
tryptophan = AminoAcid("Tryptophan", "TRP", "W", charge=-0.79, polarity=Polarity.POLAR, size=10, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=0, index=14)
tyrosine = AminoAcid("Tyrosine", "TYR", "Y", charge=-0.80, polarity=Polarity.POLAR, size=8, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=1, index=15)
valine = AminoAcid("Valine", "VAL", "V", charge=-0.37, polarity=Polarity.APOLAR, size=3, count_hydrogen_bond_donors=0, count_hydrogen_bond_acceptors=0, index=16)
selenocysteine = AminoAcid("Selenocysteine", "SEC", "U", charge=0.0, polarity=Polarity.POLAR, size=2, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=2, index=cysteine.index)
pyrrolysine = AminoAcid("Pyrrolysine", "PYL", "O", charge=0.0, polarity=Polarity.POLAR, size=13, count_hydrogen_bond_donors=1, count_hydrogen_bond_acceptors=4, index=lysine.index)
arginine = AminoAcid("Arginine", "ARG", "R", charge=-1.65, polarity=Polarity.POSITIVE_CHARGE, size=7, count_hydrogen_bond_donors=5, count_hydrogen_bond_acceptors=0, index=17)
asparagine = AminoAcid("Asparagine", "ASN", "N", charge=-1.22, polarity=Polarity.POLAR, size=4, count_hydrogen_bond_donors=2, count_hydrogen_bond_acceptors=2, index=18)
glutamine = AminoAcid("Glutamine", "GLN", "Q", charge=-1.22, polarity=Polarity.POLAR, size=5, count_hydrogen_bond_donors=2, count_hydrogen_bond_acceptors=2, index=19)


# source for the number of hydrogen bond donors and acceptors:
# https://bionumbers.hms.harvard.edu/files/Hydrogen%20bond%20donors%20and%20acceptors%20from%20amino%20acid%20residues.pdf

amino_acids = [alanine, arginine, asparagine, aspartate, cysteine, glutamate, glutamine, glycine,
               histidine, isoleucine, leucine, lysine, methionine, phenylalanine, proline, serine,
               threonine, tryptophan, tyrosine, valine]
