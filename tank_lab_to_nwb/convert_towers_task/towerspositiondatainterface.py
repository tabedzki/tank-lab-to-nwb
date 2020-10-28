"""Authors: Cody Baker and Ben Dichter."""
from nwb_conversion_tools.utils import get_base_schema, get_schema_from_hdmf_class
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from pynwb.behavior import SpatialSeries, Position
from hdmf.backends.hdf5.h5_utils import H5DataIO
import numpy as np
from pathlib import Path
from datetime import timedelta
from ..utils import check_module, convert_mat_file_to_dict, mat_obj_to_dict, array_to_dt


class TowersPositionInterface(BaseDataInterface):
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
            required=['folder_path'],
            properties=dict(
                folder_path=dict(type='string')
            )
        )

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

    def convert_data(self, nwbfile: NWBFile, metadata_dict: dict, stub_test: bool = False):
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

        """
        session_path = self.input_args['folder_path']
        mat_file = session_path + ".mat"
        matin = convert_mat_file_to_dict(mat_file)
        # TODO: move this to get_metadata in main converter
        session_start_time = array_to_dt(matin['log']['session']['start'])
        # session_start_time = metadata_dict['session_start_time']

        # Intervals
        if Path(mat_file).is_file():
            nwbfile.add_epoch_column('label', 'name of epoch')

            n_epochs = len(matin['log']['block'])
            epoch_start_dts = [array_to_dt(x.start) for x in matin['log']['block']]
            epoch_durations = [timedelta(seconds=x.duration) for x in matin['log']['block']]
            for j in range(n_epochs):
                start = epoch_start_dts[j] - session_start_time
                end = start + epoch_durations[j]
                nwbfile.add_epoch(start_time=start, stop_time=end, label='Epoch'+str(j))

            trial_starts = [trial.start for epoch in matin['log']['block'] for trial in epoch.trial]
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

            # Processed position
            pos_obj = Position(name='_position')

            pos_data = np.empty((0, 3))
            pos_timestamps = []
            for block in blocks:
                block = mat_obj_to_dict(block)
                for trial in block['trial']:
                    trial = mat_obj_to_dict(trial)
                    trial_absolute_time = trial['start'] + trial['time']
                    trial_position = trial['position']
                    trial_truncated_time = trial_absolute_time[:len(trial_position)]
                    pos_timestamps.extend(trial_truncated_time)
                    pos_data = np.concatenate([pos_data, trial_position], axis=0)

            conversion = 1.0  # need to change?

            spatial_series_object = SpatialSeries(
                name='_{}_spatial_series',
                data=H5DataIO(pos_data, compression='gzip'),
                reference_frame='unknown', conversion=conversion,
                resolution=np.nan,
                timestamps=H5DataIO(pos_timestamps, compression='gzip')
            )
            pos_obj.add_spatial_series(spatial_series_object)
            check_module(nwbfile, 'behavior', 'contains processed behavioral data').add_data_interface(pos_obj)
