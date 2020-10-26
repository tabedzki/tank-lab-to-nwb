"""Authors: Ben Dichter, Cody Baker."""
import numpy as np
from scipy.io import loadmat, matlab
from dateutil.parser import parse as dateparse

try:
    from typing import ArrayLike
except ImportError:
    from numpy import ndarray
    from typing import Union, Sequence
    # adapted from numpy typing
    ArrayLike = Union[bool, int, float, complex, list, ndarray, Sequence]


def check_module(nwbfile, name, description=None):
    """
    Check if processing module exists. If not, create it. Then return module.

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)

    Returns
    -------
    pynwb.module

    """
    if name in nwbfile.modules:
        return nwbfile.modules[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)


def find_discontinuities(tt, factor=10000):
    """Find discontinuities in a timeseries. Returns the indices before each discontinuity."""
    dt = np.diff(tt)
    before_jumps = np.where(dt > np.median(dt) * factor)[0]

    if len(before_jumps):
        out = np.array([tt[0], tt[before_jumps[0]]])
        for i, j in zip(before_jumps, before_jumps[1:]):
            out = np.vstack((out, [tt[i + 1], tt[j]]))
        out = np.vstack((out, [tt[before_jumps[-1] + 1], tt[-1]]))
        return out
    else:
        return np.array([[tt[0], tt[-1]]])


def mat_obj_to_dict(mat_struct):
    """Recursive function to convert nested matlab struct objects to dictionaries."""
    dict_from_struct = {}
    for field_name in mat_struct.__dict__['_fieldnames']:
        dict_from_struct[field_name] = mat_struct.__dict__[field_name]
        if isinstance(dict_from_struct[field_name], matlab.mio5_params.mat_struct):
            dict_from_struct[field_name] = mat_obj_to_dict(dict_from_struct[field_name])
    return dict_from_struct


def convert_mat_file_to_dict(mat_file_name):
    """
    Convert mat-file to dictionary object.

    It calls a recursive function to convert all entries
    that are still matlab objects to dictionaries.
    """
    data = loadmat(mat_file_name, struct_as_record=False, squeeze_me=True)
    for key in data:
        if isinstance(data[key], matlab.mio5_params.mat_struct):
            data[key] = mat_obj_to_dict(data[key])
    return data


def date_array_to_dt(array):
    """
    Auxiliary function for converting the array of datetime information into datetime objects.

    Parameters
    ----------
    array : array of floats of the form [Y,M,D,H,M,S].

    Returns
    -------
    datetime

    """
    temp = [str(round(x)) for x in array[0:-1]]
    date_text = temp[0] + "-" + temp[1] + "-" + temp[2] + "T" + temp[3] + ":" + temp[4] + ":" + str(array[-1])
    return dateparse(date_text)
