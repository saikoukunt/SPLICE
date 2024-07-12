import scipy.io
import numpy as np


def mat2pyCellFileCleanup(matFileName):
    """
    This function takes a matlab Cells filename (as documented here
    https://github.com/Brody-Lab/UberPhys?tab=readme-ov-file#final-datafile-format-cells-file ),
    loads it, and converts into a Python dictionary

    Parameters:
    -----------
    matFileName: str, a Matlab Cells file filename

    Returns:
    --------
    a python dictionary with the same structure and info as the matlab Cells file.
    """
    mat = scipy.io.loadmat(
        matFileName
    )  # load the file. Note that we're not using squeeze_me=True or struct_as_record=False

    myFile = dict()  # this is where we'll put everything

    for k in mat.keys():
        # special cases:
        if k == "Trials":
            # structs are complicated, and Trials is a struct so we deal with it as a special case
            # we'll turn structs into dictionaries:
            myFile["Trials"] = dict()
            for f in mat["Trials"].dtype.fields:
                # print("Doing Trials." + f)
                if f == "stateTimes":  # again a struct, to be turned into a dict
                    stateTimes = dict()
                    # for unknown reasons it comes in with a bunch of extra dimensions, so we squeeze them out:
                    for i in mat["Trials"]["stateTimes"][0][0][0][0].dtype.fields:
                        stateTimes[i] = mat["Trials"]["stateTimes"][0][0][0][0][i][:, 0]
                    myFile["Trials"]["stateTimes"] = stateTimes
                elif f == "leftBups" or f == "rightBups":
                    # different trials have different number of bups, so
                    # we'll make a list of arrays, one for each trial
                    theBups = mat["Trials"][f][0][0]
                    myBups = []
                    for trial in range(theBups.size):
                        if theBups[trial][0].size == 0:
                            myBups.append(np.array([]))
                        else:
                            myBups.append(theBups[trial][0][0, :])
                    myFile["Trials"][f] = myBups
                    # print("Did " + f)
                    # print(myFile["Trials"][f])
                else:
                    # generic case of a field in the Trials struct:
                    try:
                        myFile["Trials"][f] = mat["Trials"][f][0][0][:, 0]
                    except:
                        myFile["Trials"][f] = mat["Trials"][f][0][0]
                if f in ("is_hit", "violated"):
                    # these are logicals, so we'll convert them to bools
                    myFile["Trials"][f] = myFile["Trials"][f].astype(bool)

        elif k == "quality_metrics":
            # another struct, to be turned into a dict
            myFile["quality_metrics"] = dict()
            for i in mat["quality_metrics"].dtype.fields:
                try:
                    myFile["quality_metrics"][i] = mat["quality_metrics"][i][0][0][:, 0]
                except:
                    myFile["quality_metrics"][i] = mat["quality_metrics"][i][0][0]
        elif k == "rat":  # we want this to be turned into a string
            myFile["rat"] = mat["rat"][0].astype(str)
        elif k == "electrode":
            myFile[k] = mat[k][:, 0]
        elif k == "raw_spike_time_s":
            # we'll turn this into a list of arrays, since the number of spikes varies from cell to cell
            spikeTimes = []
            for i in range(mat[k].size):
                if mat[k][i][0].size == 0:
                    spikeTimes.append(np.array([]))
                else:
                    spikeTimes.append(mat[k][i][0][:, 0])
            myFile[k] = spikeTimes
        elif k == "region":
            region = []
            for i in range(mat[k].size):
                if mat[k][i][0].shape[0] == 0:
                    region.append("None")
                    print("No region found for cell %d, entering 'None'" % i)
                else:
                    region.append(mat[k][i][0][0].astype(str))
            myFile[k] = region
        elif k == "standard_inclusion_criteria":
            myFile[k] = mat[k].squeeze()
        else:
            # final generic case
            try:
                myFile[k] = mat[k][0][0]
                # print("Did " + k)
            except:
                # It's not a matrix, so we'll just copy it
                myFile[k] = mat[k]

    return myFile
