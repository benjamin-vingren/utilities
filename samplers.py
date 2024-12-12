# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:17:36 2024

@author: benjer
"""

from scipy.stats.qmc import Sobol
from scipy.stats import qmc
import matplotlib.pyplot as plt


def get_sobol_samples(bounds, num_samples=10, seed=None):
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
        The number of Sobol samples to generate. Default is 10.
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
    >>> samples = get_sobol_samples(bounds, num_samples=5, seed=42)
    >>> samples
    {'x1': array([ 1.46544212,  9.42368189,  3.46797881, -4.47429018, -1.42899086]),
     'x2': array([12.21551958,  4.67114914,  8.96751457,  0.39763907,  9.59546417])}
    """    
    # Select Sobol sampler
    sampler = Sobol(len(bounds), seed=seed)
    
    # Sample num_samples between [0, 1)
    samples = sampler.random(num_samples)
    
    # Convert samples to values within bounds provided by user
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    
    sobol_samples = {name: x_n for name, x_n in zip(bounds.keys(), samples.T)}
    return sobol_samples


if __name__ == '__main__':
    bounds = {'x1': [-5.0, 10.0], 'x2': [0.0, 15.0]}
    sobol_samples = get_sobol_samples(bounds, num_samples=500, seed=42)
    
    plt.figure('Sobol sampler')
    plt.plot(sobol_samples['x1'], sobol_samples['x2'], marker='.', 
             linestyle='None', markersize=0.5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    