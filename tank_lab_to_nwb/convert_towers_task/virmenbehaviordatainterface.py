"""Authors: Cody Baker Szonja Weigl and Ben Dichter."""
from datetime import timedelta
from pathlib import Path

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils import get_base_schema, get_schema_from_hdmf_class
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection
from ..utils import check_module, convert_mat_file_to_dict, array_to_dt, create_indexed_array


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
                file_path=dict(type='string')
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
        mat_file = self.input_args['file_path']
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

            epoch_maze_ids = [epoch.mazeID for epoch in matin['log']['block']]
            epoch_reward_mil = [epoch.rewardMiL for epoch in matin['log']['block']]
            epoch_stimulus_config = [epoch.stimulusConfig for epoch in matin['log']['block']]
            nwbfile.add_epoch_column(name='maze_id',
                                     description='identifier of the ViRMEn maze',
                                     data=epoch_maze_ids)
            nwbfile.add_epoch_column(name='reward_ml',
                                     description='reward in ml',
                                     data=epoch_reward_mil)
            nwbfile.add_epoch_column(name='stimulus_config',
                                     description='stimulus configuration number',
                                     data=epoch_stimulus_config)

            trial_starts = [trial.start + epoch_start_nwb[0]
                            for epoch in matin['log']['block']
                            for trial in epoch.trial]
            trial_durations = [trial.duration for epoch in matin['log']['block'] for trial in epoch.trial]
            trial_ends = [start_time + duration for start_time, duration in zip(trial_starts, trial_durations)]
            for k in range(len(trial_starts)):
                nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])

            trial_excess_travel = [trial.excessTravel for epoch in matin['log']['block']
                                   for trial in epoch.trial]
            nwbfile.add_trial_column(name='excess_travel',
                                     description='total distance traveled during the trial '
                                                 'normalized to the length of the maze',
                                     data=trial_excess_travel)

            # Processed cue timing
            left_cue_onsets = [trial.start + epoch_start_nwb[0] + trial.time[trial.cueOnset[0] - 1]
                               for epoch in matin['log']['block'] for trial in epoch.trial if
                               np.any(trial.cueOnset[0])]
            left_padding = np.full((len(trial_starts) - len(left_cue_onsets)), np.nan)
            left_cue_onsets = np.concatenate([left_cue_onsets, left_padding], axis=0)
            left_cue_onset_data, left_cue_data_indices = create_indexed_array(left_cue_onsets)

            right_cue_onsets = [trial.start + epoch_start_nwb[0] + trial.time[trial.cueOnset[1] - 1]
                                for epoch in matin['log']['block'] for trial in epoch.trial if
                                np.any(trial.cueOnset[1])]
            right_padding = np.full((len(trial_starts) - len(right_cue_onsets)), np.nan)
            right_cue_onsets = np.concatenate([right_cue_onsets, right_padding], axis=0)
            right_cue_onset_data, right_cue_data_indices = create_indexed_array(right_cue_onsets)

            cue_onset_table = DynamicTable(
                name='CueOnset',
                description='iteration numbers of ViRMEn when cues were switched on',
                id=left_cue_data_indices
            )

            cue_onset_table.add_column(
                name='left_cue_onset',
                description='onset times of left cues',
                data=left_cue_onset_data,
                index=left_cue_data_indices,
            )

            cue_onset_table.add_column(
                name='right_cue_onset',
                description='onset times of right cues',
                data=right_cue_onset_data,
                index=right_cue_data_indices,
            )

            nwbfile.add_trial_column(name=cue_onset_table.name,
                                     description=cue_onset_table.description,
                                     index=cue_onset_table.left_cue_onset_index,
                                     table=cue_onset_table)

            # Processed position, velocity, viewAngle
            pos_obj = Position(name="Position")
            view_angle_obj = CompassDirection(name='ViewAngle')

            timestamps = []
            pos_data = np.empty((0, 2))
            velocity_data = np.empty_like(pos_data)
            trial_cue_orientation = []
            view_angle_data = []
            for epoch in matin['log']['block']:
                for trial in epoch.trial:
                    trial_total_time = trial.start + epoch_start_nwb[0] + trial.time
                    timestamps.extend(trial_total_time)

                    padding = np.full((trial.time.shape[0] - trial.position.shape[0], 2), np.nan)
                    trial_position = trial.position[:, :-1]
                    trial_velocity = trial.velocity[:, :-1]
                    trial_view_angle = trial.position[:, -1]
                    pos_data = np.concatenate([pos_data, trial_position, padding], axis=0)
                    velocity_data = np.concatenate([velocity_data, trial_velocity, padding], axis=0)
                    view_angle_data = np.concatenate([view_angle_data, trial_view_angle,
                                                      padding[:, 0]], axis=0)
                    if (trial.cueCombo[0] == 1).all():
                        trial_cue_orientation.append('left')
                    elif (trial.cueCombo[1] == 1).all():
                        trial_cue_orientation.append('right')
                    else:
                        trial_cue_orientation.append('both')
            pos_obj.add_spatial_series(
                SpatialSeries(
                    name="SpatialSeries",
                    data=H5DataIO(pos_data, compression="gzip"),
                    reference_frame="unknown",
                    conversion=0.01,
                    resolution=np.nan,
                    timestamps=H5DataIO(timestamps, compression="gzip")
                )
            )
            velocity_ts = TimeSeries(name='Velocity',
                                     data=H5DataIO(velocity_data, compression="gzip"),
                                     unit='cm/s',
                                     resolution=np.nan,
                                     timestamps=H5DataIO(timestamps, compression="gzip"))
            nwbfile.add_trial_column(name='cue_orientation',
                                     description='orientation of the cues depending on '
                                                 'which side it was presented',
                                     data=trial_cue_orientation)
            view_angle_obj.add_spatial_series(
                SpatialSeries(
                    name="SpatialSeries",
                    data=H5DataIO(view_angle_data, compression="gzip"),
                    reference_frame="unknown",
                    resolution=np.nan,
                    timestamps=H5DataIO(timestamps, compression="gzip")
                )
            )
            behavioral_processing_module = check_module(nwbfile, 'behavior', 'contains processed behavioral data')
            behavioral_processing_module.add_data_interface(pos_obj)
            behavioral_processing_module.add_data_interface(velocity_ts)
            behavioral_processing_module.add_data_interface(view_angle_obj)
