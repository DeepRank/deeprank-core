from pdb2sql.interface import interface

try:
    import freesasa

except ImportError:
    print("Freesasa not found")


class BSA():
    def __init__(self, pdb_data, sqldb=None, chainA="A", chainB="B"):
        """Compute the burried surface area feature

        Freesasa is required for this feature.

        https://freesasa.github.io

        >>> wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
        >>> tar -xvf freesasa-2.0.3.tar.gz
        >>> cd freesasa
        >>> ./configure CFLAGS=-fPIC (--prefix /home/<user>/)
        >>> make
        >>> make install

        Since release 2.0.3 the python bindings are separate module
        >>> pip install freesasa

        Args :
            pdb_data (list(byte) or str): pdb data or filename of the pdb
            sqldb (pdb2sql.interface instance or None, optional) if the sqldb is None the sqldb will be created
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain

        Example :

        >>> bsa = BSA('1AK4.pdb')
        >>> bsa.get_structure()
        >>> bsa.get_contact_residue_sasa()
        >>> bsa.sql.close()

        """

        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
        else:
            self.sql = sqldb
        self.chains_label = [chainA, chainB]

        freesasa.setVerbosity(freesasa.nowarnings) # pylint: disable=c-extension-no-member

    def get_structure(self):
        """Get the pdb structure of the molecule."""

        # we can have a str or a list of bytes as input
        if isinstance(self.pdb_data, str):
            self.complex = freesasa.Structure(self.pdb_data) # pylint: disable=c-extension-no-member
        else:
            self.complex = freesasa.Structure() # pylint: disable=c-extension-no-member
            atomdata = self.sql.get("name,resName,resSeq,chainID,x,y,z")
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = f"{atomName[0]:>2}"
                self.complex.addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )
        self.result_complex = freesasa.calc(self.complex) # pylint: disable=c-extension-no-member

        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = freesasa.Structure() # pylint: disable=c-extension-no-member
            atomdata = self.sql.get("name,resName,resSeq,chainID,x,y,z", chainID=label)
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = f"{atomName[0]:>2}"
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )
            self.result_chains[label] = freesasa.calc(self.chains[label]) # pylint: disable=c-extension-no-member

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute the feature value."""

        self.bsa_data = {}
        self.bsa_data_xyz = {}

        res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(res.keys())
        res = res[keys[0]] + res[keys[1]]

        for r in res:
            chain_id, residue_number, _ = r

            # define the selection string and the bsa for the complex
            select_str = (f"res, (resi {residue_number}) and (chain {chain_id})",)
            asa_complex = freesasa.selectArea( # pylint: disable=c-extension-no-member
                select_str, self.complex, self.result_complex
            )["res"]

            # define the selection string and the bsa for the isolated
            select_str = (f"res, resi {residue_number}",)
            asa_unbound = freesasa.selectArea( # pylint: disable=c-extension-no-member
                select_str, self.chains[chain_id], self.result_chains[chain_id]
            )["res"]

            # define the bsa
            bsa = asa_unbound - asa_complex

            # define the xyz key : (chain,x,y,z)
            # chain = {"A": 0, "B": 1}[r[0]]
            # xyz = np.mean(self.sql.get("x,y,z", resSeq=r[1], chainID=r[0]), 0)
            # xyzkey = tuple([chain] + xyz.tolist())

            # put the data in dict
            self.bsa_data[r] = [bsa]
