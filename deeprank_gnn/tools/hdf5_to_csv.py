import sys
import h5py
import pandas as pd


def hdf5_to_csv(hdf5_path): # pylint: disable=too-many-locals

    hdf5 = h5py.File(hdf5_path, "r+")
    name = hdf5_path.split(".")[0]

    first = True
    for epoch in hdf5.keys():
        for dataset in hdf5[f"{epoch}"].keys():
            mol = hdf5[f"{epoch}/{dataset}/mol"]
            epoch_lst = [epoch] * len(mol)
            dataset_lst = [dataset] * len(mol)

            outputs = hdf5[f"{epoch}/{dataset}/outputs"]
            targets = hdf5[f"{epoch}/{dataset}/targets"]
            if len(targets) == 0:
                targets = "n" * len(mol)

            # This section is specific to the binary class
            # it adds the raw output, i.e. probabilities to belong to the class 0 and the class 1, to the prediction hdf5
            # This way, binary information can be transformed back to
            # continuous data and used for ranking
            if "raw_output" in hdf5[f"{epoch}/{dataset}"].keys():
                if first:
                    header = [
                        "epoch",
                        "set",
                        "model",
                        "targets",
                        "prediction",
                        "raw_prediction_0",
                        "raw_prediction_1",
                    ]
                    with open(f"{name}.csv", "w", encoding="utf-8") as output_file:
                        output_file.write("," + ",".join(header) + "\n")
                    first = False
                # probability of getting 0
                outputs_0 = hdf5[f"{epoch}/{dataset}/raw_output"][()][:, 0]
                # probability of getting 1
                outputs_1 = hdf5[f"{epoch}/{dataset}/raw_output"][()][:, 1]
                dataset_df = pd.DataFrame(
                    list(
                        zip(
                            epoch_lst,
                            dataset_lst,
                            mol,
                            targets,
                            outputs,
                            outputs_0,
                            outputs_1,
                        )
                    ),
                    columns=header,
                )

            else:
                if first:
                    header = ["epoch", "set", "model", "targets", "prediction"]
                    with open(f"{name}.csv", "w", encoding = "utf-8") as output_file:
                        output_file.write("," + ",".join(header) + "\n")
                    first = False
                dataset_df = pd.DataFrame(
                    list(zip(epoch_lst, dataset_lst, mol, targets, outputs)),
                    columns=header,
                )

            dataset_df.to_csv(f"{name}.csv", mode="a", header=False)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            """\n
This scripts converts the hdf5 output files of GraphProt into csv files

Usage:
python hdf5_to_csv.py file.hdf5
"""
        )

    else:
        try:
            hdf5_path = sys.argv[1]
            hdf5_to_csv(hdf5_path)

        except BaseException:
            print("Please make sure that your input file if a HDF5 file")
