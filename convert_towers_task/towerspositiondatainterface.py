"""Authors: Cody Baker and Ben Dichter."""
from nwb_conversion_tools.utils import get_base_schema, get_schema_from_hdmf_class
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from pynwb.file import TimeIntervals
from pynwb.behavior import SpatialSeries, Position
from hdmf.backends.hdf5.h5_utils import H5DataIO
import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from dateutil.parser import parse as dateparse
from datetime import timedelta
from .utils import check_module


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
    temp = [str(round(x)) for x in array[0][0:-1]]
    date_text = temp[0] + "-" + temp[1] + "-" + temp[2] + "T" + temp[3] + ":" + temp[4] + ":" + str(array[0][-1])
    return dateparse(date_text)


class TankPositionInterface(BaseDataInterface):
    """Description here."""

    @classmethod
    def get_input_schema(cls):
        """
        Place description here.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return dict(
            source_data=dict(
                required=['folder_path'],
                properties=dict(
                    folder_path=dict(type='string')
                )
            )
        )

    def __init__(self, **input_args):
        super().__init__(**input_args)

    def get_metadata_schema(self):
        """
        Place description here.

        Returns
        -------
        metadata_schema : TYPE
            DESCRIPTION.

        """
        metadata_schema = get_base_schema()

        # ideally most of this be automatically determined from pynwb docvals
        metadata_schema['properties']['SpatialSeries'] = get_schema_from_hdmf_class(SpatialSeries)
        required_fields = ['SpatialSeries']
        for field in required_fields:
            metadata_schema['required'].append(field)

        return metadata_schema

    def convert_data(self, nwbfile: NWBFile, metadata_dict: dict,
                     stub_test: bool = False, include_spike_waveforms: bool = False):
        """
        Primary conversion function for the custom ttank lab positional interface.

        Parameters
        ----------
        nwbfile : NWBFile
            DESCRIPTION.
        metadata_dict : dict
            DESCRIPTION.
        stub_test : bool, optional
            DESCRIPTION. The default is False.
        include_spike_waveforms : bool, optional
            DESCRIPTION. The default is False.

        """
        session_path = self.input_args['folder_path']
        # TODO: move this to get_metadata in main converter
        # session_start_time = metadata_dict['session_start_time']
        mat_file = session_path + ".mat"
        matin = loadmat(mat_file)
        session_start_time = date_array_to_dt(matin['log']['session'][0][0]['start'][0][0][0])

        # task_types = metadata_dict['task_types']

        subject_path, session_id = os.path.split(session_path)

        pos_obj = Position(name='_position')

        mat_file = session_path + ".mat"
        matin = loadmat(mat_file)
        pos_data = matin #[...]
        tt = matin #[...]
        conversion = 1 # need to change?
        exp_times = matin #[...]

        # Processed position
        spatial_series_object = SpatialSeries(
            name='_{}_spatial_series',
            data=H5DataIO(pos_data, compression='gzip'),
            reference_frame='unknown', conversion=conversion,
            resolution=np.nan,
            timestamps=H5DataIO(tt, compression='gzip'))
        pos_obj.add_spatial_series(spatial_series_object)
        check_module(nwbfile, 'behavior', 'contains processed behavioral data').add_data_interface(pos_obj)

        # Intervals
        if Path(mat_file).is_file():
            nwbfile.add_epoch_column('label', 'name of epoch')

            blocks = matin['log']['block']
            epoch_start_fields = blocks[0][0]['start'][0]
            n_epochs = len(epoch_start_fields)
            epoch_start_dts = [date_array_to_dt(x[0]) for x in epoch_start_fields]
            epoch_durationss = [timedelta(seconds=x[0][0]) for x in blocks[0][0]['duration'][0]]

            epoch_windows = []
            for j in range(n_epochs):
                start = epoch_start_dts[j] - session_start_time
                end = start + epoch_durationss[j]
                epoch_windows.append([start.total_seconds(), end.total_seconds()])

            for j, window in enumerate(epoch_windows):
                nwbfile.add_epoch(start_time=window[0], stop_time=window[1], label='Epoch'+str(j))

            trial_start_fields = []
            trial_duration_fields = []
            n_trials = []
            for j in range(n_epochs):
                trial_start_fields.append(blocks[0][0]['trial'][0][j]['start'][0])
                trial_duration_fields.append(blocks[0][0]['trial'][0][j]['duration'][0])
                n_trials.append(len(trial_start_fields[j]))
            trial_starts = [y[0][0]+epoch_windows[j][0] for j, x in enumerate(trial_start_fields) for y in x]
            trial_ends = [y1[0][0]+epoch_windows[j][0]+y2[0][0]
                          for j, x in enumerate(zip(trial_start_fields, trial_duration_fields))
                          for y1, y2 in zip(x[0], x[1])]

            for k in range(len(trial_starts)):
                nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])
