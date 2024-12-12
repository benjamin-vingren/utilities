# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:33:32 2024

@author: benjer
"""

import json
import os
import pickle
import numpy as np


def json_write_dictionary(file_name, to_save, check=True):
    if check:
        if os.path.exists(file_name):
            inp = input(
                f'{file_name} already exists.\nDo you want to overwrite it? [y/n] ')
            if inp != 'y':
                print('File was not overwritten.')
                return 0
    with open(file_name, 'w') as handle:
        json.dump(to_save, handle)


def json_read_dictionary(file_name):
    with open(file_name, 'r') as handle:
        j = json.load(handle)
    return j


def unpickle(file_name):
    with open(file_name, 'rb') as handle:
        A = pickle.load(handle)
        return A


def numpify(dictionary):
    """Take dictionary of lists and return dictionary with numpy arrays."""
    to_return = {}
    for key in dictionary.keys():
        to_return[key] = np.array(dictionary[key])

    return to_return


def listify(dictionary):
    """
    Converts any NumPy arrays in a dictionary to lists.

    Parameters
    ----------
    dictionary : dict
        A dictionary possibly containing NumPy arrays.

    Returns
    -------
    dict
        A new dictionary with the same keys and values as the input dictionary,
        but where NumPy arrays have been replaced with lists.

    Notes
    -----
    This function recursively iterates through the input dictionary and 
    converts any NumPy arrays it finds to lists. It also handles string values 
    and nested dictionaries.
    """
    to_return = {}
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            to_return[key] = value.tolist()
        elif isinstance(value, np.str_):
            to_return[key] = str(value)
        elif isinstance(value, dict):
            to_return[key] = listify(value)
        else:
            to_return[key] = value

    return to_return
