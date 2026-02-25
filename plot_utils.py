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
import locale


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


def multipage(filename, figs=None, check=True, combine_pdf=False):
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

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    if not combine_pdf:
        for i, fig in enumerate(figs):
            pp = PdfPages(f'{filename}_{i}.pdf')
            fig.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
            pp.close()
    else:
        pp = PdfPages(filename)

        for fig in figs:
            fig.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
        pp.close()


def multiPNG(filename, dpi=400, check=True):
    '''
    Saves all open figures to PNG.
    '''
    if check:
        if os.path.exists(filename):
            inp = input(f'Overwrite {filename}? [y/n] ')
            if inp != 'y':
                return 0
            else:
                print('Overwriting file.')

    for i, fignum in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fignum)
        fig.savefig(f'{filename}_{i}.png', dpi=dpi, bbox_inches='tight',
                    pad_inches=0)


def get_markers(n):
    """Return a number of markers"""
    markers = np.array(['.', '+', 'x', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p',
                        '*', 'D', 'd', '|', '_', 'P', 'X'])
    if n > len(markers):
        raise ValueError(
            f'Not enough unique markers, max(n) = {len(markers)}.')
    return markers[0:n]


def set_nes_plot_style(large_font=False, swedish=False):
    matplotlib.rcParams['interactive'] = True
    dirname = os.path.dirname(__file__)
    if large_font and swedish:
        locale.setlocale(locale.LC_ALL, "sv_SE.UTF-8")
        filename = os.path.join(dirname, 'nes_plots_swedish_large.mplstyle')
    elif large_font:
        filename = os.path.join(dirname, 'nes_plots_large.mplstyle')
    elif swedish:
        locale.setlocale(locale.LC_ALL, "sv_SE.UTF-8")
        filename = os.path.join(dirname, 'nes_plots_swedish.mplstyle')
    else:
        filename = os.path.join(dirname, 'nes_plots.mplstyle')

    plt.style.use(filename)


def plot_contour(x1, x2, obj_fcn, title, c_levels, log, f_name='',
                 vmin=None, vmax=None):
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
        Figure title for the plots. This is typically a string that helps
        identify the dataset.
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
    plt.colorbar(scatter, label=f'$f_{{{f_name}}}(x_1, x_2)$')
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
                           norm=norm, vmin=vmin, vmax=vmax)

    plt.colorbar(label=f'$f_{{{f_name}}}(x_1, x_2)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    # Move exponential to the left
    t = plt.gca().yaxis.get_offset_text()
    t.set_x(-0.18)

    return contour, scatter


def plot_matrix(matrix, bin_edges=None, log=False, xlabel=None, ylabel=None,
                colorbar_label=None, vmin=None, vmax=None, title=None):
    """
    Plot a 2D matrix as a color-coded heatmap using histogram binning.

    Parameters
    ----------
    matrix : array_like
        2D array to be visualized as a heatmap.
    bin_edges : tuple of array_like, optional
        A tuple containing two arrays representing the bin edges for the x and
        y axes, respectively. If None, uniform bin edges are generated based
        on the matrix shape.
    log : bool, default=False
        If True, applies logarithmic normalization to the color scale.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    colorbar_label : str, optional
        Label for the colorbar.
    vmin : float, optional
        Minimum value for colormap normalization. Overrides default behavior.
    vmax : float, optional
        Maximum value for colormap normalization. Overrides default behavior.

    Returns
    -------
    hist2d : tuple
        A tuple containing the histogram output from
        `matplotlib.pyplot.hist2d`.
    """

    plt.figure()
    if bin_edges is None:
        x_bin_edges = np.arange(0, np.shape(matrix)[0] + 1)
        y_bin_edges = np.arange(0, np.shape(matrix)[1] + 1)
    else:
        x_bin_edges = bin_edges[0]
        y_bin_edges = bin_edges[1]

    x_bin_centres = x_bin_edges[1:] - np.diff(x_bin_edges) / 2
    y_bin_centres = y_bin_edges[1:] - np.diff(y_bin_edges) / 2

    # Create fill for x and y
    x_repeated = np.tile(x_bin_centres, len(y_bin_centres))
    y_repeated = np.repeat(y_bin_centres, len(x_bin_centres))
    weights = np.ndarray.flatten(np.transpose(matrix))

    # Set white background
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)

    if log:
        normed = matplotlib.colors.LogNorm(vmin=1)
    else:
        normed = None

    # Create 2D histogram using weights
    hist2d = plt.hist2d(x_repeated, y_repeated, bins=(x_bin_edges, y_bin_edges),
                        weights=weights, cmap=my_cmap, norm=normed, vmin=vmin,
                        vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=colorbar_label)
    plt.title(title)

    return hist2d


def plot_heatmap(x, y, z, bin_edges_x, bin_edges_y, xlabel, ylabel,
                 colorbar_label, vmin=0, vmax=1, title=None):
    """
    Plot a 2D heatmap of mean z-values binned over x and y.

    The function computes the mean of ``z`` within each 2D bin defined by
    ``bin_edges_x`` and ``bin_edges_y`` and visualizes the result as a heatmap.

    Parameters
    ----------
    x : array_like
        One-dimensional array of x-coordinates.
    y : array_like
        One-dimensional array of y-coordinates.
    z : array_like
        One-dimensional array of values associated with each (x, y) pair.
        Must have the same length as ``x`` and ``y``.
    bin_edges_x : array_like
        Bin edge definitions along the x-axis.
    bin_edges_y : array_like
        Bin edge definitions along the y-axis.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    colorbar_label : str
        Label for the colorbar representing the binned ``z`` values.
    vmin : float, optional
        Minimum value for the colormap normalization. Default is 0.
    vmax : float, optional
        Maximum value for the colormap normalization. Default is 1.
    title : str or None, optional
        Title of the plot. If ``None``, no title is displayed.

    Returns
    -------
    hist2d : object
        The heatmap object returned by ``plot_matrix()``.

    Notes
    -----
    - Bins that contain no data points will result in ``NaN`` values due to
      the mean computation.
    - Bin intervals are defined as ``(bin_edges[i], bin_edges[i + 1]]``.
    """
    # Loop over x
    bin_values = np.zeros([len(bin_edges_x) - 1, len(bin_edges_y) - 1])
    for i in range(len(bin_edges_x) - 1):
        mask_x = (x > bin_edges_x[i]) & (x <= bin_edges_x[i + 1])

        # Loop over y
        for j in range(len(bin_edges_y) - 1):
            mask_y = (y > bin_edges_y[j]) & (y <= bin_edges_y[j + 1])

            # z mean value for bin
            bin_values[i, j] = z[mask_x & mask_y].mean()

    hist2d = pu.plot_matrix(bin_values, xlabel=xlabel, ylabel=ylabel,
                            colorbar_label=colorbar_label,
                            bin_edges=[bin_edges_x, bin_edges_y],
                            vmin=vmin, vmax=vmax, title=title)

    return hist2d

if __name__ == '__main__':
    set_nes_plot_style()