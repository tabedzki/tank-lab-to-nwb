"""Authors: Cody Baker Szonja Weigl and Ben Dichter."""
from datetime import timedelta
from pathlib import Path

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils import get_base_schema, get_schema_from_hdmf_class
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import SpatialSeries, Position
from ..utils import check_module, convert_mat_file_to_dict, array_to_dt


class VirmenDataInterface(BaseDataInterface):
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
        Primary conversion function for the custom tank lab behavioral interface.

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

            epoch_start_dts = [array_to_dt(x.start) for x in matin['log']['block']]
            epoch_durations = [timedelta(seconds=x.duration) for x in matin['log']['block']]
            epoch_start_nwb = [(epoch_start_dt - session_start_time).total_seconds()
                               for epoch_start_dt in epoch_start_dts]
            epoch_end_nwb = [(epoch_start_dt - session_start_time + epoch_duration).total_seconds()
                             for epoch_start_dt, epoch_duration in zip(epoch_start_dts, epoch_durations)]
            for j, (start, end) in enumerate(zip(epoch_start_nwb, epoch_end_nwb)):
                nwbfile.add_epoch(start_time=start, stop_time=end, label='Epoch'+str(j+1))

            trial_starts = [trial.start + epoch_start_nwb[0]
                            for epoch in matin['log']['block']
                            for trial in epoch.trial]
            trial_durations = [trial.duration for epoch in matin['log']['block'] for trial in epoch.trial]
            trial_ends = [start_time + duration for start_time, duration in zip(trial_starts, trial_durations)]
            for k in range(len(trial_starts)):
                nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])

            # Processed position, velocity
            pos_obj = Position(name="Position")

            timestamps = []
            pos_data = np.empty((0, 3))
            velocity_data = np.empty((0, 2))
            for epoch in matin['log']['block']:
                for trial in epoch.trial:
                    trial_total_time = trial.start + epoch_start_nwb[0] + trial.time
                    trial_position = trial.position
                    trial_velocity = trial.velocity[:, :-1]
                    trial_truncated_time = trial_total_time[:trial_position.shape[0]]
                    timestamps.extend(trial_truncated_time)
                    pos_data = np.concatenate([pos_data, trial_position], axis=0)
                    velocity_data = np.concatenate([velocity_data, trial_velocity], axis=0)
            pos_obj.add_spatial_series(
                SpatialSeries(
                    name="SpatialSeries",
                    data=H5DataIO(pos_data, compression="gzip"),
                    reference_frame="unknown",
                    resolution=np.nan,
                    timestamps=H5DataIO(timestamps, compression="gzip")
                )
            )
            velocity_ts = TimeSeries(name='Velocity',
                                     data=H5DataIO(velocity_data, compression="gzip"),
                                     unit='cm/s',
                                     resolution=np.nan,
                                     timestamps=H5DataIO(timestamps, compression="gzip"))

            behavioral_processing_module = check_module(nwbfile, 'behavior', 'contains processed behavioral data')
            behavioral_processing_module.add_data_interface(pos_obj)
            behavioral_processing_module.add_data_interface(velocity_ts)
