# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:17:36 2024

@author: benjer
"""

from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np


def get_sobol_samples(bounds, num_samples, seed=None):
    """
    Generate Sobol samples within specified bounds.

    The Sobol sequence is a low-discrepancy sequence used in quasi-Monte
    Carlo methods for numerical integration and sampling. This function
    generates samples within the user-provided bounds.

    Parameters
    ----------
    bounds : dict
        A dictionary where keys are parameter names and values are lists of two
        floats [lower_bound, upper_bound], specifying the sampling range for
        each parameter.
    num_samples : int, optional
        The number of samples to generate (n_samples should be a power of 2).
    seed : int or None, optional
        Seed for the Sobol sequence generator. If 'None', the seed is not set.
        Default is 'None'.

    Returns
    -------
    sobol_samples : dict
        A dictionary where keys correspond to the parameter names from
        'bounds', and values are arrays of sampled values for each parameter.

    Examples
    --------
    Generate Sobol samples for two parameters:

    >>> bounds = {'x1': [-5.0, 10.0], 'x2': [0.0, 15.0]}
    >>> samples = get_sobol_samples(bounds, num_samples=2**9, seed=42)
    >>> samples
    {'x1': array([ 1.46544212,  9.42368189,  3.46797881, ...]),
     'x2': array([12.21551958,  4.67114914,  8.96751457, ...])}
    """    
    # Select Sobol sampler
    sampler = qmc.Sobol(len(bounds), seed=seed)
    
    # Sample num_samples between [0, 1)
    samples = sampler.random(num_samples)
    
    # Convert samples to values within bounds provided by user
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    
    sobol_samples = {name: x_n for name, x_n in zip(bounds.keys(), samples.T)}
    return sobol_samples


def get_grid_samples(parameters):
    """
    Generate a grid of samples from provided parameter arrays.

    Parameters
    ----------
    parameters : list of array-like
        A list of arrays where each array represents possible values for a
        single parameter.

    Returns
    -------
    flat_grid : list of numpy.ndarray
        A list of flattened arrays representing the coordinate values of
        the generated grid.

    Examples
    --------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([4, 5, 6])
    >>> get_grid_samples([x1, x2])
    [array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
     array([4, 5, 6, 4, 5, 6, 4, 5, 6])]
    """
    # Create the grid
    grid = np.meshgrid(*parameters)
    
    # Return flattened arrays
    flat_grid = [p.flatten() for p in grid]
    
    return flat_grid
    
    
def get_latin_hypercube_samples(bounds, num_samples, seed=None):
    """
    Generate Latin Hypercube samples within specified bounds.

    Parameters
    ----------
    bounds : dict
        A dictionary where keys are parameter names and values are lists of two
        floats [lower_bound, upper_bound], specifying the sampling range for
        each parameter.
    num_samples : int
        The number of samples to generate (n_samples should be a power of 2).
    seed : int or None, optional
        Seed for the LHS generator. If 'None', the seed is not set.

    Returns
    -------
    lh_samples : dict
        A dictionary where keys correspond to the parameter names from
        'bounds', and values are arrays of sampled values for each parameter.

    Examples
    --------
    >>> bounds = {'x1': [0, 1], 'x2': [5, 10]}
    >>> get_latin_hypercube_samples(bounds, num_samples=2**9, seed=123)
    {'x1': array([0.153, 0.534, 0.876, ...]),
     'x2': array([7.234, 6.581, 9.124, ...])}
    """
    
    # Select latin hypercube sampler
    sampler = qmc.LatinHypercube(len(bounds), seed=seed)
    
    # Sample num_samples between [0, 1)
    samples = sampler.random(num_samples)
    
    # Convert samples to values within bounds provided by user
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    
    lh_samples = {name: x_n for name, x_n in zip(bounds.keys(), samples.T)}
    return lh_samples


def get_random_samples(bounds, num_samples, seed=None):
    """
    Generate random samples within specified bounds.

    This function generates uniformly distributed random samples for each
    parameter within its specified bounds.

    Parameters
    ----------
    bounds : dict
        A dictionary where keys are parameter names and values are lists of two
        floats [lower_bound, upper_bound], specifying the sampling range for
        each parameter.
    num_samples : int
        The number of samples to generate.
    seed : int or None, optional
        Seed for the random number generator. If 'None', the seed is not set.

    Returns
    -------
    rand_samples : dict
        A dictionary where keys correspond to the parameter names from
        'bounds', and values are arrays of sampled values for each parameter.

    Examples
    --------
    >>> bounds = {'x1': [0, 1], 'x2': [5, 10]}
    >>> get_random_samples(bounds, num_samples=4, seed=123)
    {'x1': array([0.134, 0.567, 0.852, 0.295]),
     'x2': array([7.632, 6.198, 5.467, 9.821])}
    """
    # Select random sampler
    rng = np.random.default_rng(seed)
    samples = rng.random([num_samples, len(bounds)])
    
    # Convert samples to values within bounds provided by user
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    
    rand_samples = {name: x_n for name, x_n in zip(bounds.keys(), samples.T)}
    return rand_samples


if __name__ == '__main__':
    pass
    
    
    
    