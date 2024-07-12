"""
pyUberPhys.utilities.utilities

Utilities for pyUberPhys

Functions:
----------

quartileIds(x)
    Given an np.array of numbers, divides it up into quartiles,
    and returns an np.array of the same size, with each element
    being the quartile that the corresponding element of x is in
    (where the quartiles are numbered from 0 to 3).

splitTimes(timeStamps, boundaries)
    Given a list of timeStamps and a list of boundaries, split the timeStamps
    into a list of arrays of timeStamps, where each array of timeStamps is
    bounded by the boundaries. 

gaussianFilter(sigma, noWarn=False)
    Returns a 1D Gaussian filter with standard deviation sigma, in units of bins

smoothIt(mysignal, sigma, noWarn=False)
    Smooths a signal with a Gaussian filter with standard deviation sigma, in units of bins

binEvents(eventTimes, t, dt)
    Given a list of event times, bin them into bins of size dt, and return
    a list of the number of events in each bin. 

torchBinEvents(eventTimes, t, dt)
    Given a list of event times, bin them into bins of size dt, and return
    a list of the number of events in each bin. 

selectCell(G, cellnum, alignOn="clicks_on", removeFirstBup=False)
    Given a dictionary from mat2pythonCellFileCleanup(), and a cell number, return a pandas dataframe with the assembled data
    for that cell.

"""

import math
from sqlite3 import adapt

# import GPy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sympy import use

# ------------------
#
# quartileIds(x)
#
# ------------------


def quartileIds(x):
    """
    Given an np.array of numbers, divides it up into quartiles,
    and returns an np.array of the same size, with each element
    being the quartile that the corresponding element of x is in
    (where the quartiles are numbered from 0 to 3).

    Parameters
    ----------
    x : np.array
        np.array of numbers

    Returns
    -------
    np.array
        np.array of quartile identities (0 to 3), same size as x
    """
    boundaries = np.percentile(x, [25, 50, 75])
    out = np.zeros_like(x)
    out[x < boundaries[0]] = 0
    out[(x >= boundaries[0]) & (x < boundaries[1])] = 1
    out[(x >= boundaries[1]) & (x < boundaries[2])] = 2
    out[x >= boundaries[2]] = 3

    return out


# ------------------
#
# splitTimes
#
# ------------------


def splitTimes(timeStamps, boundaries):
    """
    Given a list of timeStamps and a list of boundaries, split the timeStamps
    into a list of arrays of timeStamps, where each array of timeStamps is
    bounded by the boundaries. When a timeStamp is equal to a boundary, it is
    included as the first element of the new chunk, not the previous.

    If there are no timeStamps before the first boundary, the first array of timeStamps will be
    empty. The number of arrays of timeStamps will be equal to one plus number of boundaries.

    Thus, for example, splitTimes([2.0], [1.0, 2.0]) will result in nothing before 1.0,
    nothing in [1.0, 2.0), and the single timestap 2.0 in [2.0 ...], giving:
      [[],
       [],
       [2.0]]


    Parameters
    ----------
    timeStamps : np.array
        np.array of timeStamps
    boundaries : np.array
        np.array of boundaries

    Returns
    -------
    list
        List of arrays of timeStamps, length one plus the length of boundaries

    Example
    -------
    >>> splitTimes(np.arange(0, 10), np.array([2, 5]))

    [array([0, 1]), array([2, 3, 4]), array([5, 6, 7, 8, 9])]

    """

    splitTimes = []
    chunk = -1
    thisChunkIndexStart = 0

    for i in range(timeStamps.size):
        # first case: we just reached a boundary, define the chunk before it
        if chunk < boundaries.size - 1 and timeStamps[i] >= boundaries[chunk + 1]:
            splitTimes.append(timeStamps[thisChunkIndexStart:i])
            thisChunkIndexStart = i
            chunk += 1
            # while we pass boundaries without needing to go to the next TimeStamp, add an empty chunk
            while (
                chunk < boundaries.size - 1 and timeStamps[i] >= boundaries[chunk + 1]
            ):
                splitTimes.append(np.array([]))
                chunk += 1
        # second case: we just reached the last boundary, define the chunk after it
        elif chunk == boundaries.size - 1:
            splitTimes.append(timeStamps[thisChunkIndexStart:])
            break

        # Otherwise, if we reached the end of the timeStamps before reaching the last boundary,
        # we're not going to have more timeStamps telling us we're passing boundaries
        if i == timeStamps.size - 1:
            # first define the chunk we're in
            splitTimes.append(timeStamps[thisChunkIndexStart:])
            chunk += 1
            # while there are boundaries left, add empty chunks
            while chunk < boundaries.size:
                splitTimes.append(np.array([]))
                chunk += 1

    return splitTimes


testSplitTimes = False
if testSplitTimes:
    print(splitTimes(np.array([3.1, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])))


# ------------------
#
# gaussianFilter and smoothIt
#
# ------------------


def gaussianFilter(sigma, noWarn=False):
    """
    Returns a 1D Gaussian filter with standard deviation sigma, in units of bins
    This means that sigma < 5 will be pretty poorly represented

    The sum of the all of the elements of the returned filter will be 1

    Parameters:
    -----------
    sigma: float
        The standard deviation of the Gaussian, in bins
    noWarn: bool, default=False
        Whether to print a warning if sigma<5

    Returns:
    --------
    f: np.array
        A 1D Gaussian filter with standard deviation sigma, in units of bins

    centerBin: int, the index into the bin with the maximum value of the filter

    """
    if (sigma < 5) and noWarn == False:
        print(
            "Warning: sigma <5, discretization means the filter will be poorly represented"
        )
    c = int(np.ceil(4 * sigma))
    x = np.arange(-c, c + 1)
    centerBin = c

    f = np.exp(-(x**2) / (2 * sigma**2))
    return f / f.sum(), centerBin


def smoothIt(mysignal, sigma, noWarn=False):
    """
    Smooths a signal with a Gaussian filter with standard deviation sigma, in units of bins

    Parameters:
    -----------
    mysignal: np.array
        The signal to be smoothed
    noWarn: bool, default=False
        Whether to print a warning if sigma<5

    sigma: float
        The standard deviation of the Gaussian, in bins

    Returns:
    --------
    smoothedSignal: np.array
        The smoothed signal. Will be the same length as mysignal, and
        its peaks will coincide with the original's peaks

    """
    f, centerBin = gaussianFilter(sigma, noWarn=noWarn)
    smoothedSignal = np.convolve(mysignal, f, mode="full")[centerBin:-centerBin]
    return smoothedSignal


# ------------------
#
# binEvents and torchBinEvents
#
# ------------------


def binEvents(eventTimes, t, dt):
    """
    Given a list of event times, bin them into bins of size dt, and return
    a list of the number of events in each bin. The first bin is centered at
    t[0] + dt/2, and the last bin is centered at t[-1] - dt/2.

    Parameters
    ----------
    eventTimes : np.array
        np.array of event times
    t : np.array
        np.array of times
    dt : float
        The size of the bins

    Returns
    -------
    np.array
        np.array of number of events in each bin

    """
    return np.histogram(
        eventTimes, bins=np.arange(t[0] - dt / 2, t[-1] + 3 * dt / 4, dt)
    )[0]


def torchBinEvents(eventTimes, t, dt):
    """
    Given a list of event times, bin them into bins of size dt, and return
    a list of the number of events in each bin. The first bin is centered at
    t[0] + dt/2, and the last bin is centered at t[-1] - dt/2.

    Parameters
    ----------
    eventTimes : np.array
        np.array of event times
    t : np.array
        np.array of times
    dt : float
        The size of the bins

    Returns
    -------
    np.array
        np.array of number of events in each bin

    """
    return torch.histogram(
        eventTimes, torch.tensor(np.arange(t[0] - dt / 2, t[-1] + 3 * dt / 4, dt))
    )[0]


# ------------------
#
#  selectCell(), a function to turn a loaded file into a dataframe, with spikes from a particular, indicated neuron id
#
# ------------------


def selectCell(G, cellnum, alignOn="clicks_on", removeFirstBup=False):
    """
    Given a cell number, return a pandas dataframe with the following columns:
    pokedR, is_hit, sending_trialnum, cpoke_in, clicks_on, clicks_off, cpoke_out, spoke, stimDur,
    leftBups, L (# of leftBups), rightBups, R (# of rightBups), stereo_clock, spikeTimes,
    originalTrialNum (which is the trial number in the cell file, where trials are counted
    including violation trials, etc.)

    The dataframe will contain only the trials where the animal did not violate
    and poked into one of the side ports.

    All times are absolute times in seconds, except for clicks_on, stereo_click,
    leftBups, rightBups, cpoke_out, and spikeTimes, which are relative to the alignOn time.

    Parameters
    ----------
    G : dict
        Dictionary of the cell file
    cellnum : int
        Cell number
    alignOn : str, optional
        What to align the spike times on. The default is "clicks_on".
    removeFirstBup : bool, optional
        Whether to remove the first bup from leftBups and rightBups. The default is False.
        Set this to True if you want to use a stereoClick kernel, since that
        kernel will represent these first bups

    Returns
    -------
    pandas.DataFrame
        Dataframe with the following columns:
        pokedR, is_hit, sending_trialnum, cpoke_in, clicks_on, clicks_off, cpoke_out,
        spoke, stimDur, leftBups, L, rightBups, R, stereo_click, spikeTimes

    Example
    --------
    >> df = selectCell(G, 23, alignOn="cpoke_in", removeFirstBup=True)
    >> df.loc[1].leftBups

    """
    nTrials = G["nTrials"]
    originalTrialNum = np.arange(1, nTrials + 1)

    nonViol = ~np.isnan(G["Trials"]["pokedR"])
    df = pd.DataFrame({"pokedR": G["Trials"]["pokedR"][nonViol] == 1})

    df.insert(df.shape[1], "is_hit", G["Trials"]["is_hit"][nonViol])
    df.insert(
        df.shape[1],
        "sending_trialnum",
        G["Trials"]["stateTimes"]["sending_trialnum"][nonViol],
    )
    df.insert(df.shape[1], "cpoke_in", G["Trials"]["stateTimes"]["cpoke_in"][nonViol])
    df.insert(df.shape[1], "clicks_on", G["Trials"]["stateTimes"]["clicks_on"][nonViol])
    df.insert(
        df.shape[1], "clicks_off", G["Trials"]["stateTimes"]["clicks_off"][nonViol]
    )
    df.insert(df.shape[1], "cpoke_out", G["Trials"]["stateTimes"]["cpoke_out"][nonViol])
    df.insert(df.shape[1], "spoke", G["Trials"]["stateTimes"]["spoke"][nonViol])
    df.insert(df.shape[1], "stimDur", df.clicks_off - df.clicks_on)

    df.insert(
        df.shape[1],
        "leftBups",
        [i for (i, v) in zip(G["Trials"]["leftBups"], nonViol) if v],
    )
    df.insert(
        df.shape[1],
        "L",
        list(
            map(
                lambda x: x.size,
                [i for (i, v) in zip(G["Trials"]["leftBups"], nonViol) if v],
            )
        ),
    )
    df.insert(
        df.shape[1],
        "rightBups",
        [i for (i, v) in zip(G["Trials"]["rightBups"], nonViol) if v],
    )
    df.insert(
        df.shape[1],
        "R",
        list(
            map(
                lambda x: x.size,
                [i for (i, v) in zip(G["Trials"]["rightBups"], nonViol) if v],
            )
        ),
    )

    stereoClick = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if (
            (df.leftBups[i].size == 0)
            or (df.rightBups[i].size == 0)
            or (df.leftBups[i][0] != df.rightBups[i][0])
        ):
            raise ValueError("No stereo click on trial %d" % i)
        stereoClick[i] = df.leftBups[i][0]
        if removeFirstBup:
            df.at[i, "leftBups"] = df.leftBups[i][1:]
            df.at[i, "rightBups"] = df.rightBups[i][1:]
    df.insert(df.shape[1], "stereo_click", stereoClick)
    df.insert(df.shape[1], "originalTrialNum", originalTrialNum[nonViol].astype(int))

    # if we're aligning on stereo_click, we need to convert it from
    # it's default of being relative to clicks_on into absolute time,
    # since we're going to use it to align spike times
    #     Then below, we won't correct stereo_click for alignOnTime- clicks_on

    if alignOn == "stereo_click":
        df.stereo_click = df.stereo_click + df.clicks_on

    alignOnTime = df[alignOn]

    # By default, leftBups, rightBups, and stereo_click are relative to clicks_on,
    # so we need to subtract alignOnTime - clicks_on
    df.leftBups = df.leftBups - (alignOnTime - df.clicks_on)
    # --- Failed attenpt at storing single-element arrays in a dataframe as a non-zero dim awrray
    # singleLeftBups = list(map(lambda x : not isinstance(x, np.ndarray), df.leftBups))
    # gu = np.arange(0, len(singleLeftBups))[singleLeftBups].astype(int)
    # for i in range(gu.size):
    #     df.at[gu[i], "leftBups"] = np.array([df.leftBups[gu[i]]])
    # ---- end failed attempt
    df.rightBups = df.rightBups - (alignOnTime - df.clicks_on)

    if alignOn == "stereo_click":
        df.stereo_click = 0
    else:
        df.stereo_click = df.stereo_click - (alignOnTime - df.clicks_on)

    # for other columns, we need to subtract alignOnTime
    df.clicks_on = df.clicks_on - alignOnTime
    df.clicks_off = df.clicks_off - alignOnTime
    df.cpoke_in = df.cpoke_in - alignOnTime
    df.cpoke_out = df.cpoke_out - alignOnTime

    rawSpikeTimes = G["raw_spike_time_s"][cellnum]
    spikeTimes = splitTimes(rawSpikeTimes, df.sending_trialnum)[1:]
    for i in range(len(spikeTimes)):
        spikeTimes[i] = spikeTimes[i] - alignOnTime[i]

    df.insert(df.shape[1], "spikeTimes", spikeTimes)

    return df

    # -------- SAI HELPER FUNCTIONS -----------


def selectCells(G, cellnums, alignOn="clicks_on", removeFirstBup=False):
    """
    Given a cell number, return a pandas dataframe with the following columns:
    pokedR, is_hit, sending_trialnum, cpoke_in, clicks_on, clicks_off, cpoke_out, spoke, stimDur,
    leftBups, L (# of leftBups), rightBups, R (# of rightBups), stereo_clock, spikeTimes,
    originalTrialNum (which is the trial number in the cell file, where trials are counted
    including violation trials, etc.)

    The dataframe will contain only the trials where the animal did not violate
    and poked into one of the side ports.

    All times are absolute times in seconds, except for clicks_on, stereo_click,
    leftBups, rightBups, cpoke_out, and spikeTimes, which are relative to the alignOn time.

    Parameters
    ----------
    G : dict
        Dictionary of the cell file
    cellnum : int
        Cell number
    alignOn : str, optional
        What to align the spike times on. The default is "clicks_on".
    removeFirstBup : bool, optional
        Whether to remove the first bup from leftBups and rightBups. The default is False.
        Set this to True if you want to use a stereoClick kernel, since that
        kernel will represent these first bups

    Returns
    -------
    pandas.DataFrame
        Dataframe with the following columns:
        pokedR, is_hit, sending_trialnum, cpoke_in, clicks_on, clicks_off, cpoke_out,
        spoke, stimDur, leftBups, L, rightBups, R, stereo_click, spikeTimes

    Example
    --------
    >> df = selectCell(G, 23, alignOn="cpoke_in", removeFirstBup=True)
    >> df.loc[1].leftBups

    """
    nTrials = G["nTrials"]
    originalTrialNum = np.arange(1, nTrials + 1)

    nonViol = ~np.isnan(G["Trials"]["pokedR"])
    df = pd.DataFrame({"pokedR": G["Trials"]["pokedR"][nonViol] == 1})

    df.insert(df.shape[1], "is_hit", G["Trials"]["is_hit"][nonViol])
    df.insert(
        df.shape[1],
        "sending_trialnum",
        G["Trials"]["stateTimes"]["sending_trialnum"][nonViol],
    )
    df.insert(df.shape[1], "cpoke_in", G["Trials"]["stateTimes"]["cpoke_in"][nonViol])
    df.insert(df.shape[1], "clicks_on", G["Trials"]["stateTimes"]["clicks_on"][nonViol])
    df.insert(
        df.shape[1], "clicks_off", G["Trials"]["stateTimes"]["clicks_off"][nonViol]
    )
    df.insert(df.shape[1], "cpoke_out", G["Trials"]["stateTimes"]["cpoke_out"][nonViol])
    df.insert(df.shape[1], "spoke", G["Trials"]["stateTimes"]["spoke"][nonViol])
    df.insert(df.shape[1], "stimDur", df.clicks_off - df.clicks_on)

    df.insert(
        df.shape[1],
        "leftBups",
        [i for (i, v) in zip(G["Trials"]["leftBups"], nonViol) if v],
    )
    df.insert(
        df.shape[1],
        "L",
        list(
            map(
                lambda x: x.size,
                [i for (i, v) in zip(G["Trials"]["leftBups"], nonViol) if v],
            )
        ),
    )
    df.insert(
        df.shape[1],
        "rightBups",
        [i for (i, v) in zip(G["Trials"]["rightBups"], nonViol) if v],
    )
    df.insert(
        df.shape[1],
        "R",
        list(
            map(
                lambda x: x.size,
                [i for (i, v) in zip(G["Trials"]["rightBups"], nonViol) if v],
            )
        ),
    )

    stereoClick = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if (
            (df.leftBups[i].size == 0)
            or (df.rightBups[i].size == 0)
            or (df.leftBups[i][0] != df.rightBups[i][0])
        ):
            raise ValueError("No stereo click on trial %d" % i)
        stereoClick[i] = df.leftBups[i][0]
        if removeFirstBup:
            df.at[i, "leftBups"] = df.leftBups[i][1:]
            df.at[i, "rightBups"] = df.rightBups[i][1:]
    df.insert(df.shape[1], "stereo_click", stereoClick)
    df.insert(df.shape[1], "originalTrialNum", originalTrialNum[nonViol].astype(int))

    # if we're aligning on stereo_click, we need to convert it from
    # it's default of being relative to clicks_on into absolute time,
    # since we're going to use it to align spike times
    #     Then below, we won't correct stereo_click for alignOnTime- clicks_on

    if alignOn == "stereo_click":
        df.stereo_click = df.stereo_click + df.clicks_on

    alignOnTime = df[alignOn]

    # By default, leftBups, rightBups, and stereo_click are relative to clicks_on,
    # so we need to subtract alignOnTime - clicks_on
    df.leftBups = df.leftBups - (alignOnTime - df.clicks_on)
    # --- Failed attenpt at storing single-element arrays in a dataframe as a non-zero dim awrray
    # singleLeftBups = list(map(lambda x : not isinstance(x, np.ndarray), df.leftBups))
    # gu = np.arange(0, len(singleLeftBups))[singleLeftBups].astype(int)
    # for i in range(gu.size):
    #     df.at[gu[i], "leftBups"] = np.array([df.leftBups[gu[i]]])
    # ---- end failed attempt
    df.rightBups = df.rightBups - (alignOnTime - df.clicks_on)

    if alignOn == "stereo_click":
        df.stereo_click = 0
    else:
        df.stereo_click = df.stereo_click - (alignOnTime - df.clicks_on)

    # for other columns, we need to subtract alignOnTime
    df.clicks_on = df.clicks_on - alignOnTime
    df.clicks_off = df.clicks_off - alignOnTime
    df.cpoke_in = df.cpoke_in - alignOnTime
    df.cpoke_out = df.cpoke_out - alignOnTime

    spikeTimes = np.empty(len(df.clicks_on), dtype="object")
    for i in range(len(df.clicks_on)):
        spikeTimes[i] = []

    cellTimes = np.zeros(cellnums.shape[0], dtype="object")
    for i in range(cellnums.shape[0]):
        rawSpikeTimes = G["raw_spike_time_s"][cellnums[i]]
        cellTimes[i] = splitTimes(rawSpikeTimes, df.sending_trialnum)[1:]
        for j in range(len(cellTimes[i])):
            cellTimes[i][j] = cellTimes[i][j] - alignOnTime[j]

    for i in range(len(df.clicks_on)):
        for j in range(cellnums.shape[0]):
            spikeTimes[i].append(cellTimes[j][i])

    df.insert(df.shape[1], "spikeTimes", spikeTimes)

    return df


def binEventsBounds(eventTimes, dt, bounds):
    """
    Given a list of event times, bin them into bins of size dt, and return
    a list of the number of events in each bin. The first bin is centered at
    t[0] + dt/2, and the last bin is centered at t[-1] - dt/2.

    Parameters
    ----------
    eventTimes : np.array
        np.array of event times
    t : np.array
        np.array of times
    dt : float
        The size of the bins

    Returns
    -------
    np.array
        np.array of number of events in each bin

    """
    return np.histogram(
        eventTimes,
        bins=np.arange(bounds[0], bounds[1], dt),
    )[
        0
    ].astype("float")


def binSmoothTrialSpikeTimes(trial, bounds, dt, sigma, adaptive=True):
    n_bins = math.floor((bounds[1] - bounds[0]) / dt)

    frs = np.zeros((len(trial), n_bins))
    bin = np.zeros((len(trial), n_bins))

    for i in range(len(trial)):
        spikes = trial[i]
        binned = binEventsBounds(spikes, dt, bounds)
        if adaptive:
            sigma = np.maximum(1 / (binned.mean() / 0.05), 0.1) / 0.05
            if sigma == np.inf:
                sigma = binned.shape[0]
            rates = gaussian_filter1d(binned, math.floor(sigma))
        else:
            rates = gaussian_filter1d(binned, math.floor(sigma / dt))
        frs[i, :] = rates
        bin[i, :] = binned

    return frs, bin


def getSessionRates(df, dt, sigma, adaptive=True):
    rates = np.zeros((len(df)), dtype="object")
    binned = np.zeros((len(df)), dtype="object")
    for i in range(len(df)):
        rates[i], binned[i] = binSmoothTrialSpikeTimes(
            df["spikeTimes"][i],
            [df["stereo_click"][i], df["clicks_off"][i]],
            dt,
            sigma,
            adaptive=adaptive,
        )

    df.insert(df.shape[1], "binned", binned)
    df.insert(df.shape[1], "rates", rates)


def binTrialSpikeTimes(trial, bounds, dt):
    n_bins = math.floor((bounds[1] - bounds[0]) / dt)

    bin = np.zeros((len(trial), n_bins))

    for i in range(len(trial)):
        spikes = trial[i]
        bin[i, :] = binEventsBounds(spikes, dt, bounds)

    return bin


def getSessionBins(df, dt):
    bins = np.zeros((len(df)), dtype="object")
    for i in range(len(df)):
        bins[i] = binTrialSpikeTimes(
            df["spikeTimes"][i],
            [df["stereo_click"][i], df["clicks_off"][i]],
            dt,
        )

    df.insert(df.shape[1], "binned", bins)


def GP_smooth(bins, bin_size, lengthscale, variance):
    """
    Smooths the input data using Gaussian Process Regression.

    ### Parameters:
    - `bins` (numpy.ndarray): Input data array of shape (# of neurons, # of samples).
    - `bin_size` (float): The width in seconds of each bin.

    ### Returns:
    - `y_pred` (numpy.ndarray): Smoothed data array of shape (# of neurons, # of samples).
    """
    n = bins.shape[1]
    x = np.arange(0, n * bin_size, bin_size).reshape(-1, 1)[:n]

    gp = GaussianProcessRegressor().fit(x, bins.T)

    return gp.predict(x).T  # type: ignore


def getGPRates(df, dt, lengthscale, variance):
    """
    Calculate the rates using Gaussian Process smoothing for each row in the DataFrame.

    ### Parameters:
    - `df` (pandas.DataFrame): The input Trials DataFrame containing binned spike counts.
    - `dt` (float): The time step used for binning.
    - `lengthscale` (float): The GP kernel lengthscale as a multiple of `dt`.
    - `variance` (float): The GP kernel variance.

    ### Returns:
    - `rates` (pandas.DataFrame): A DataFrame containing the calculated rates for each trial.
    """

    rates = np.zeros((len(df)), dtype="object")
    for i in range(len(df)):
        print(i)
        rates[i] = GP_smooth(df["binned"][i], dt, lengthscale, variance)

    df.insert(df.shape[1], "rates", rates)


def add_trial_start_stop(df):
    trial_max = [
        max([np.max(row) for row in trial if len(row) > 0])
        for trial in df["spikeTimes"]
    ]
    trial_min = [
        min([np.min(row) for row in trial if len(row) > 0])
        for trial in df["spikeTimes"]
    ]

    df.insert(df.shape[1], "spikeStart", trial_min)
    df.insert(df.shape[1], "spikeEnd", trial_max)


def bin_region(region_times, bin_size):
    """
    Bins the given spike times.

    ### Parameters:
    - `region_times` (numpy.ndarray): Array of shape (num_neurons, num_spikes) containing
            the spike times for each neuron.
    - `bin_size` (float): The size of each bin in time units.

    ### Returns:
    - `binned` (numpy.ndarray): Array of shape (num_neurons, num_bins) containing the
            binned spike counts for each neuron.
    """

    max_time = np.max(np.vectorize(max)(region_times))

    num_neurons = region_times.shape[0]
    num_bins = int(np.ceil(max_time / bin_size))

    binned = np.zeros((num_neurons, num_bins))

    for i in range(num_neurons):
        binned[i], _ = np.histogram(region_times[i], bins=num_bins, range=(0, max_time))

    return binned
