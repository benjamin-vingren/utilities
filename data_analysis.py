# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:23:12 2024

@author: benjer
"""

import numpy as np


def running_average(a, n):
    """
    Compute the running average and standard deviation of an input array.

    Parameters
    ----------
    a : array_like,
        Input array to compute the running average and standard deviation.
    n : int,
        Number of elements (plus/minus) to include in the running average.

    Returns
    -------
    average : ndarray,
        Array of the same shape as the input array 'a' containing the running
        averages.
    std : ndarray,
        Array of the same shape as the input array 'a' containing the running
        standard deviations.

    Notes
    -----
    The function computes the running average and standard deviation of the
    input array by dividing it into three parts:
    - The beginning and end parts of length 'n', where the running average and
      standard deviation are computed independently.
    - The middle part of the array, where the running average and standard
      deviation are computed over a window of length '2n+1'.
    """
    # Beginning/end part of list
    start_of_list = np.empty([n, 2 * n])
    start_of_list[:] = np.nan

    end_of_list = np.empty([n, 2 * n])
    end_of_list[:] = np.nan

    for i in range(n):
        # Choose beginning/end of list
        start_of_list[i][0:i + n + 1] = a[0:i + n + 1]
        end_of_list[i][-2 * n + i:] = a[-2 * n + i:]

    # Calculate average
    start_average = np.nanmean(start_of_list, axis=1)
    end_average = np.nanmean(end_of_list, axis=1)

    # Calculate standard deviation
    start_std = np.nanstd(start_of_list, axis=1)
    end_std = np.nanstd(end_of_list, axis=1)

    # Middle part of list
    mid_of_list = np.zeros([len(a) - 2 * n, 2 * n + 1])
    for i in range(len(a) - 2 * n):
        # Choose middle part of list
        mid_of_list[i] = a[i:2 * n + i + 1]

    # Calculate average
    mid_average = np.mean(mid_of_list, axis=1)

    # Calculate standard devitation
    mid_std = np.std(mid_of_list, axis=1)

    # Concatenate arrays
    average = np.concatenate((start_average, mid_average, end_average))
    std = np.concatenate((start_std, mid_std, end_std))

    return average, std


def get_bin_centres(bin_edges):
    """
    Compute the center positions of histogram bins from bin edges.

    Parameters
    ----------
    bin_edges : array_like
        A one-dimensional array of monotonically increasing bin edge values.
        The length of `bin_edges` must be one greater than the number of bins.

    Returns
    -------
    bin_centres : ndarray
        A one-dimensional array containing the center position of each bin,
        with length ``len(bin_edges) - 1``.

    Notes
    -----
    This function assumes uniform bin widths and computes the bin centers
    by subtracting half of the bin width from the upper bin edges.
    """
    return bin_edges[1:] - np.diff(bin_edges)[0] / 2