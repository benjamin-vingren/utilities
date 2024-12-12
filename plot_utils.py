# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:29:06 2024

@author: benjer
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import os
from scipy.interpolate import griddata
import matplotlib.colors as colors


def get_linestyle(ls):
    if ls == 'dash-dot-dotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif ls == 'loosely dotted':
        return (0, (1, 10))
    elif ls == 'dotted':
        return (0, (1, 2))
    elif ls == 'densely dotted':
        return (0, (1, 1))
    elif ls == 'long dash with offset':
        return (5, (10, 3))
    elif ls == 'loosely dashed':
        return (0, (5, 10))
    elif ls == 'dashed':
        return (0, (5, 5))
    elif ls == 'densely dashed':
        return (0, (5, 1))
    elif ls == 'loosely dashdotted':
        return (0, (3, 10, 1, 10))
    elif ls == 'dashdotted':
        return (0, (3, 5, 1, 5))
    elif ls == 'densely dashdotted':
        return (0, (3, 1, 1, 1))
    elif ls == 'dashdotdotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif ls == 'loosely dashdotdotted':
        return (0, (3, 10, 1, 10, 1, 10))
    elif ls == 'densely dashdotdotted':
        return (0, (3, 1, 1, 1, 1, 1))

    else:
        print('Available linestyles:')
        print('loosely dotted')
        print('dotted')
        print('densely dotted')
        print('long dash with offset')
        print('loosely dashed')
        print('dashed')
        print('densely dashed')
        print('loosely dashdotted')
        print('dashdotted')
        print('densely dashdotted')
        print('dashdotdotted')
        print('loosely dashdotdotted')
        print('densely dashdotdotted')


        raise Exception(f'Unknown linestyle: {ls}')
        

def get_colors(n_colors):
    colors_list = ['k']
    for c in mcolors.TABLEAU_COLORS:
        colors_list.append(c)

    for c in mcolors.CSS4_COLORS:
        colors_list.append(c)

    # Remove whites
    white = ['whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'linen',
             'oldlace', 'floralwhite', 'ivory', 'lightyellow', 'honeydew',
             'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'lavenderblush']
    for w in white:
        colors_list.remove(w)

    if n_colors > len(colors_list):
        print('Not enough colors in list, returning random colors.')
        return random_colors(n_colors)
    return colors_list[:n_colors]


def random_colors(size=1, seed=0):
    if seed:
        np.random.seed(seed)
    if size == 1:
        colors = [np.random.uniform(0, 1), np.random.uniform(
            0, 1), np.random.uniform(0, 1)]
    else:
        colors = [[np.random.uniform(0, 1), np.random.uniform(
            0, 1), np.random.uniform(0, 1)] for i in range(size)]

    return colors


def multipage(filename, figs=None, dpi=200, check=True, tight_layout=True):
    '''
    Saves all open figures to a PDF.
    '''
    if check:
        if os.path.exists(filename):
            inp = input(f'Overwrite {filename}? [y/n] ')
            if inp != 'y':
                return 0
            else:
                print('Overwriting file.')
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        if tight_layout:
            fig.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig(pp, format='pdf')
    pp.close()


def get_markers(n):
    """Return a number of markers"""
    markers = np.array(['.', '+', 'x', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p',
                        '*', 'D', 'd', '|', '_', 'P', 'X'])
    if n > len(markers):
        raise ValueError(
            f'Not enough unique markers, max(n) = {len(markers)}.')
    return markers[0:n]


def set_nes_plot_style():
    matplotlib.rcParams['interactive'] = True
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'nes_plots.mplstyle')
    plt.style.use(filename)
    

def plot_contour(x1, x2, obj_fcn, title, c_levels, log):
    """
    Plot contour and scatter plots for x1 and x2 with their evaluated objective
    function.

    This function generates a contour plot and a scatter plot to visualize the
    relationship between x1 and x2 and their corresponding evaluated objective
    function. 

    Parameters
    ----------
    x1 : numpy.ndarray
        Array of evaluated x1 values.
    x2 : numpy.ndarray
        Array of evaluated x2 values.
    obj_fcn : numpy.ndarray
        Array of the evaluated values f(x1, x2).
    title : str
        Title for the plots. This is typically a string that helps identify the
        dataset.
    c_levels : numpy.ndarray
        Array of contour levels to be used in the contour plot.
    log : bool
        Logarithmic scale for scatter plot (True/False).

    Returns
    -------
    contour : matplotlib.pyplot.figure
        Matplotlib figure object of the contour plot.
    scatter : matplotlib.pyplot.figure
        Matplotlib figure object of the scatter plot.
    """
    # Scatter plot
    if log:
        norm = colors.LogNorm()
    else:
        norm = None
    plt.figure(f'Scatter {title}')
    scatter = plt.scatter(x1, x2, c=obj_fcn, s=0.5, norm=norm)
    plt.colorbar(scatter, label='$f(x_1, x_2)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    # Create grid coordinates
    xi = np.linspace(x1.min(), x1.max(), 50)
    yi = np.linspace(x2.min(), x2.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate z values on the grid
    Zi = griddata((x1, x2), obj_fcn, (Xi, Yi), method='cubic')

    # Contour plot
    plt.figure(f'Contour {title}', figsize=(8, 6))
    contour = plt.contourf(Xi, Yi, Zi, levels=c_levels, cmap='viridis', 
                           norm=norm)

    plt.colorbar(label='$f(x_1, x_2)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
        
    # Move exponential to the left
    t = plt.gca().yaxis.get_offset_text()
    t.set_x(-0.18)

    return contour, scatter

if __name__ == '__main__':
    set_nes_plot_style()