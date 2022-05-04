import glob
import sys


def pssm_3dcons_to_deeprank(pssm_file):

    # pssm = open(pssm_file, 'r').readlines()
    with open(pssm_file, "r", encoding = "utf-8") as f:
        pssm = f.readlines()

    pssm_name = pssm_file.rsplit(".", 1)[0]

    with open(f"{pssm_name}.deeprank.pssm", "w", encoding = "utf-8") as new_pssm:

        firstline = True

        for line in pssm:

            if firstline is True:
                firstline = False
                new_pssm.write(
                    "pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E\
        G    H    I    L    K    M    F    P    S    T    W    Y    V   IC\n"
                )

            if len(line.split()) == 44:
                resid = line[0:6].strip()
                resn = line[6]
                pssm_content = line[11:90]
                ic = line.split()[-1]

                new_pssm.write(
                    "{0:>5} {1:1} {0:>5} {1:1}    {2} {3}\n".format( # pylint: disable=consider-using-f-string
                        resid, resn, pssm_content, ic
                    )
                )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            """\n
This scripts converts the 3dcons pssm files into deeprank pssm format

Usage:
python 3dcons_to_deeprank_pssm.py [path_to_pssm]
"""
        )

    else:
        try:
            pssm_path = sys.argv[1]
            for pssm_file in glob.glob(f"{pssm_path}/*.pssm"):
                pssm_3dcons_to_deeprank(pssm_file)
        except BaseException:
            print("You must provide the path to the pssm files")
