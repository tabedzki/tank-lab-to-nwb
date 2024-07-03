"""Authors: Cody Baker Szonja Weigl and Ben Dichter."""
from datetime import timedelta, datetime
from pathlib import Path
from copy import deepcopy

import warnings
from typing import Optional

from neuroconv.utils.dict import DeepDict
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import get_base_schema, get_schema_from_hdmf_class
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection
from ndx_tank_metadata import LabMetaDataExtension, RigExtension, MazeExtension
from neuroconv.datainterfaces import SpikeGLXRecordingInterface

from neuroconv.utils import FilePathType, get_schema_from_method_signature, dict_deep_update

from ..utils import check_module, convert_mat_file_to_dict, array_to_dt, create_indexed_array, \
    flatten_nested_dict, convert_function_handle_to_str, create_and_store_indexed_array


class VirmenDataInterface(BaseDataInterface):
    """Conversion class for Virmen behavioral data."""

    def __init__(
        self,
        file_path: FilePathType,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        file_path : FilePathType
            Path to virman .mat file.
        verbose : bool, default: True
            Whether to output verbose text.
        """

        self.verbose = verbose
        file_path = Path(file_path)
        folder_path = file_path.parent
        super().__init__(
            file_path=file_path,
            folder_path=folder_path,
            verbose=verbose,
        )
        mat_file = self.source_data['file_path']
        self._mat_dict = convert_mat_file_to_dict(mat_file)
        self._times = None

    def _get_session_start_time(self):
        session_start_time = array_to_dt(self._mat_dict['log']['session']['start'])
        return session_start_time

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        return dict(
            required=['file_path'],
            properties=dict(
                file_path=dict(type='string')
            )
        )

    def get_metadata(self) -> DeepDict:
        """
        Get metadata from the mat file and add it to the metadata dictionary.
        """

        super_metadata = super().get_metadata()

        local_log_copy = deepcopy(self._mat_dict["log"])
        metadata = deepcopy(self._mat_dict)
        session = local_log_copy["session"]
        # experimenter = [", ".join(session["experimenter"].split(" ")[::-1])]
        #! !TODO: Fetch the experimenter from the database
        experimenter = ["FAKE PERSON"]
        session_start_time = self._get_session_start_time()

        metadata_from_mat_dict = dict(
            Subject=dict(subject_id=local_log_copy["animal"]),
            NWBFile=dict(experimenter=experimenter, session_start_time=session_start_time),
        )


        metadata = dict_deep_update(super_metadata, metadata_from_mat_dict, copy=True)
        # super_metadata.add(metadata)

        # metadata = dict_deep_update(super_metadata, metadata, copy=True)
        # metadata = dict_deep_update(super_metadata, metadata_from_mat_dict, copy=True)


        return metadata


    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):

        metadata_copy = deepcopy(self._mat_dict)
        nwbfile.add_epoch_column('label', 'name of epoch')

        # ------------------------ Adding LabMetaDataExtension ----------------------- #

        experiment_metadata = metadata_copy['log']['version']

        if isinstance(metadata_copy['log']['block'], dict):
            epochs = [metadata_copy['log']['block']]
        else:
            epochs = metadata_copy['log']['block']

        trials = [trial for epoch in epochs for trial in epoch['trial'] if
                    not np.isnan(trial['start'])]

        # Extension of lab metadata
        experiment_metadata = metadata_copy['log']['version']
        subject_metadata = metadata_copy['log']['animal']
        rig_atrr = ['rig', 'simulationMode', 'hasDAQ', 'hasSyncComm', 'minIterationDT',
                    'arduinoPort', 'sensorDotsPerRev', 'ballCircumference', 'toroidXFormP1',
                    'toroidXFormP2', 'colorAdjustment', 'soundAdjustment', 'nidaqDevice',
                    'nidaqPort', 'nidaqLines', 'syncClockChannel', 'syncDataChannel',
                    'rewardChannel', 'rewardSize', 'rewardDuration', 'laserChannel',
                    'rightPuffChannel', 'leftPuffChannel', 'webcam_name']

        rig = {k: v for k, v in experiment_metadata['rig'].items() if k in rig_atrr}
        rig.update((k, v.astype(np.int32)) for k, v in rig.items() if
                    isinstance(v, np.ndarray) and v.dtype == np.uint8)

        # Wrap the rig_attr sensorDorsPerRev in a list to match the expected format since sometimes it is a scalar
        rig['sensorDotsPerRev'] = [rig['sensorDotsPerRev']] if 'sensorDotsPerRev' in rig and isinstance(rig['sensorDotsPerRev'], (int, float)) else rig.get('sensorDotsPerRev')

        for maze in experiment_metadata["mazes"]:
            for k, v in maze["criteria"].items():
                if k.startswith('warmup') and isinstance(v, np.ndarray) and len(v) == 0:
                    maze["criteria"][k] = np.nan

        rig_extension = RigExtension(name='rig', **rig)

        maze_extension = MazeExtension(name='mazes',
                                        description='description of the mazes')

        for maze in experiment_metadata['mazes']:
            flatten_maze_dict = flatten_nested_dict(maze)
            maze_extension.add_row(**dict((k, v) for k, v in flatten_maze_dict.items()
                                            if k in MazeExtension.mazes_attr))

        num_trials = len(trials)
        session_end_time = array_to_dt(metadata_copy['log']['session']['end']).isoformat()
        converted_metadata = convert_function_handle_to_str(mat_file_path=self.source_data['file_path'])

        lab_meta_data = dict(
            name='LabMetaData',
            experiment_name=converted_metadata[
                'experiment_name'] if 'experiment_name' in converted_metadata else '',
            world_file_name=experiment_metadata['name'],
            protocol_name=converted_metadata[
                'protocol_name'] if 'protocol_name' in converted_metadata else '',
            stimulus_bank_path=subject_metadata['stimulusBank'] if subject_metadata[
                'stimulusBank'] else '',
            commit_id=experiment_metadata['repository'],
            location=experiment_metadata['rig']['rig'],
            num_trials=num_trials,
            session_end_time=session_end_time,
            rig=rig_extension,
            mazes=maze_extension,
            # session_performance=0.,  # comment out to add (not in behavior file)
        )

        nwbfile.add_lab_meta_data(LabMetaDataExtension(**lab_meta_data))


        # --------------------------- Calculating Intervals -------------------------- #

        if isinstance(metadata_copy['log']['block'], dict):
            epochs = [metadata_copy['log']['block']]
        else:
            epochs = metadata_copy['log']['block']

        trials = [trial for epoch in epochs for trial in epoch['trial'] if
                    not np.isnan(trial['start'])]



        # ------------------------- Adding epochs information ------------------------- #

        session_start_time = self._get_session_start_time()

        epoch_start_dts = [array_to_dt(epoch['start']) for epoch in epochs]
        epoch_durations_dts = [timedelta(seconds=epoch['duration']) for epoch in epochs]
        epoch_start_nwb = [(epoch_start_dt - session_start_time).total_seconds()
                            for epoch_start_dt in epoch_start_dts]
        epoch_end_nwb = [(epoch_start_dt - session_start_time + epoch_duration).total_seconds()
                            for epoch_start_dt, epoch_duration in
                            zip(epoch_start_dts, epoch_durations_dts)]
        for j, (start, end) in enumerate(zip(epoch_start_nwb, epoch_end_nwb)):
            nwbfile.add_epoch(start_time=start, stop_time=end, label='Epoch' + str(j + 1))

        epoch_maze_ids = [epoch['mazeID'] for epoch in epochs]
        epoch_main_maze_ids = [epoch['mainMazeID'] for epoch in epochs]
        epoch_easy_flag = [epoch['easyBlockFlag'] for epoch in epochs]
        epoch_first_trial = [epoch['firstTrial'] for epoch in epochs]
        epoch_num_trials = []
        for epoch in epochs:
            num_non_empty_trials = 0
            for trial in epoch['trial']:
                if not np.isnan(trial['start']):
                    num_non_empty_trials += 1
            epoch_num_trials.append(num_non_empty_trials)

        epoch_durations = [epoch['duration'] for epoch in epochs]
        epoch_reward_mil = [epoch['rewardMiL'] for epoch in epochs]
        epoch_stimulus_config = [epoch['stimulusConfig'] for epoch in epochs]

        nwbfile.add_epoch_column(name='maze_id',
                                    description='number of maze run in an epoch',
                                    data=epoch_maze_ids)
        nwbfile.add_epoch_column(name='main_maze_id',
                                    description='number of maze of "highest" level for subject',
                                    data=epoch_main_maze_ids)
        nwbfile.add_epoch_column(name='easy_epoch',
                                    description='1 if block was flagged as easy '
                                                '(maze_id < main_maze_id)',
                                    data=epoch_easy_flag)
        nwbfile.add_epoch_column(name='first_trial',
                                    description='first trial run in an epoch',
                                    data=epoch_first_trial)
        nwbfile.add_epoch_column(name='num_trials',
                                    description='number of trials in an epoch',
                                    data=epoch_num_trials)
        nwbfile.add_epoch_column(name='duration',
                                    description='duration of epoch in seconds',
                                    data=epoch_durations)
        nwbfile.add_epoch_column(name='reward_ml',
                                    description='reward in ml',
                                    data=epoch_reward_mil)
        nwbfile.add_epoch_column(name='stimulus_config',
                                    description='stimulus configuration number',
                                    data=epoch_stimulus_config)

        # ------------------------- Adding trial information ------------------------- #
        # -------- This information stays the same throughout the specific trial ------ #

        trial_starts = [trial['start'] + epoch_start_nwb[0] for trial in trials]
        trial_durations = [trial['duration'] for trial in trials]
        trial_ends = [start_time + duration for start_time, duration in
                        zip(trial_starts, trial_durations)]
        for k in range(len(trial_starts)):
            nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])

        nwbfile.add_trial_column(name='duration',
                                    description='duration of trial in seconds',
                                    data=trial_durations)
        trial_idx = [trial_id for num_trials in epoch_num_trials
                        for trial_id in np.arange(0, num_trials)]
        nwbfile.add_trial_column(name='trial_id',
                                    description='number of trial in block',
                                    data=trial_idx)

        trial_columns = [
            ( 'iterations', 'number of iterations (frames) for entire trial'),
            ( 'iCueEntry', 'iteration number when subject entered cue region'),
            ( 'iMemEntry', 'iteration number when subject entered memory region'),
            ( 'iTurnEntry', 'iteration number when subject entered turn region'),
            ( 'iArmEntry', 'iteration number when subject entered arm region'),
            ( 'iBlank', 'iteration number when screen is turned off'),
            ( 'excessTravel', 'total distance traveled during the trial '
                             'normalized to the length of the maze'),
            ( 'rewardScale', 'multiplier of reward for each correct trial'),

            ( 'StartCycle',	'The spatial frequency of the first stimulus shown to the mouse.'),
            ( 'EndCycle',	'The spatial frequency of the second stimulus shown to the mouse.'),
            ( 'rule',	'Specifies what the mouse should do to receive reward. Can be "StartCycle < EndCycle Left" or "StartCycle < EndCycle Right"'),
            ( 'trialNum',	'The trial number (not reliable, bugged).'),
            ( 'multibiasBeta',	'Scalar value indicating the strength of the multibias.'),
            ( 'multibiasTau',	'Scalar value indicating the history dependence of the multibias.'),
            ( 'pairNum',	'Integer value indicating which stimulus pair was shown, is a row index into stimulusTable.'),
            ( 'wallGuide',	'Boolean, whether wallGuides were active or not on this trial.'),
            ( 'alpha_plus',	'Scalar, how much to multiply step_size by when moving the moon beacon trigger forward.'),
            ( 'alpha_minus',	'Scalar, how much to multiply step_size by when moving the moon beacon trigger back.'),
            ( 'moonBeaconEnabled',	'Boolean, whether the moon beacon is enabled or not.'),
            ( 'moonBeaconPos',	'Scalar [cm], current position of the moon beacon trigger relative to the moonBeaconTrigger.'),
            ( 'moonBeaconTrigger',	'String indicating the trigger point for the moon. Can be "Sa" or "Sb".'),
            ( 'step_size',	'Scalar [cm], how much to move the moon beacon triggerpoint on each trial.'),
            ( 'lsrON',	'Whether the laser is on for a given trial, 0=laser off, 1=laser on'),
            ( 'iLaserOn',	'Virmen iteration when laser is turn on (if it is, otherwise 0)'),
            ( 'iLaserOff',	'Virmen iteration when laser is turn off (if it is, otherwise 0)'),
            ( 'moonDistHint',	'Distance from start at which moon beacon appears'),
            ( 'forcedChoice',	'Whether a trial is a forced choice L-maze environment, 0=T-maze, 1=L-maze'),
                            ]

        for column_name, desc in trial_columns:
            data = [trial[column_name] for trial in trials if column_name in trial]
            if data:
                nwbfile.add_trial_column(name=column_name,
                                            description=desc,
                                            data=data)

        if 'trial_choice' in converted_metadata:
            nwbfile.add_trial_column(name='choice',
                                        description='choice (L=Left,R=Right,nil=Trial violation)',
                                        data=converted_metadata['trial_choice'])
        if 'trial_type' in converted_metadata:
            nwbfile.add_trial_column(name='trial_type',
                                        description='type of trial (L=Left,R=Right)',
                                        data=converted_metadata['trial_type'])


        # --------------------- Processed cue timing and position -------------------- #
        # -------- This information stays the same throughout the specific trial ------ #

        left_cue_presence = [trial['cueCombo'][0] if len(trial['cueCombo'])
                                else trial['cueCombo'] for trial in trials]
        left_cue_presence_data, left_cue_presence_indices = create_indexed_array(
            left_cue_presence)

        right_cue_presence = [trial['cueCombo'][1] if len(trial['cueCombo'])
                                else trial['cueCombo'] for trial in trials]
        right_cue_presence_data, right_cue_presence_indices = create_indexed_array(
            right_cue_presence)

        left_cue_onset = [
            trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOnset'][0] - 1]
            if np.any(trial['cueOnset'][0]) else trial['cueOnset'][0] for trial in trials]

        right_cue_onset = [
            trial['start']
            + epoch_start_nwb[0]
            + trial['time'][trial['cueOnset'][1] - 1]
            if np.any(trial['cueOnset'][1]) else trial['cueOnset'][1] for trial in trials]

        left_cue_offset = [
            trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOffset'][0] - 1]
            if np.any(trial['cueOffset'][0]) else trial['cueOffset'][0] for trial in trials]

        right_cue_offset = [
            trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOffset'][1] - 1]
            if np.any(trial['cueOffset'][1]) else trial['cueOffset'][1] for trial in trials]

        left_cue_position = [trial['cuePos'][0] if len(trial['cuePos'])
                                else trial['cuePos'] for trial in trials]

        right_cue_position = [trial['cuePos'][1] if len(trial['cuePos'])
                                else trial['cuePos'] for trial in trials]

        baseCycles = [trial['baseCycles'] if len(trial['baseCycles']) else trial['baseCycles'] for trial in trials]

        trial_columns = [
            ( 'left_cue_presence','Indicates whether the nth cue appeared on the left',),
            ( 'right_cue_presence','Indicates whether the nth cue appeared on the right',),
            ( 'left_cue_onset','Onset times of left cues'),
            ( 'right_cue_onset', 'Onset times of right cues'),
            ( 'left_cue_offset', 'Offset times of left cues'),
            ( 'right_cue_offset', 'Offset times of right cues'),
            ( 'left_cue_position', 'Position of left cues'),
            ( 'right_cue_position', 'Position of right cues'),
            # ( 'hitHistory',	'Boolean vector tracking whether the mouse got the trial correct or not.'),
            # ( 'classHistory',	'Vector tracking which (StartCycle,EndCycle) stimulus pair was shown to the mouse, with each value being a row index into stimulusTable.'),
            ( 'baseCycles',	'The base set of spatial frequencies from which StartCycle and EndCycle can be drawn.'),
        ]

        if 'stimulusTable' in trial:

            stimulusTable_columns = zip(*[(trial['stimulusTable'][:, i] if len(trial['stimulusTable']) else trial['stimulusTable']
                                        for i in range(8)) for trial in trials])

            # Unpack the transposed columns into separate variables
            (stimulusTable_pairNum, stimulusTable_prob, stimulusTable_side,
            stimulusTable_freq_stimulus_one, stimulusTable_freq_stimulus_two,
            stimulusTable_cumulative_stimulus_hitrate, stimulusTable_stimulus_ntimes_shown,
            stimulusTable_stimulus_post_prob) = stimulusTable_columns

            trial_columns.extend( [
            ( 'stimulusTable_prob', 'Prior probability of each pair'),
            ( 'stimulusTable_freq_stimulus_one', 'Frequency of first stimulus'),
            ( 'stimulusTable_freq_stimulus_two', 'Frequency of second stimulus'),
            ( 'stimulusTable_cumulative_stimulus_hitrate', 'Cumulative hitrate for this stimulus pair'),
            ( 'stimulusTable_stimulus_ntimes_shown', 'Number of times this pair has been shown'),
            ( 'stimulusTable_stimulus_post_prob', 'Posterior probability of showing this pair'),
            ( 'stimulusTable_side', 'Correct side for each pair'),
            ])

            stimulusTable_pairNum = [trial['stimulusTable'][:,0] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_prob = [trial['stimulusTable'][:,1] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_side = [trial['stimulusTable'][:,2] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_freq_stimulus_one = [trial['stimulusTable'][:,3] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_freq_stimulus_two = [trial['stimulusTable'][:,4] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_cumulative_stimulus_hitrate = [trial['stimulusTable'][:,5] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_stimulus_ntimes_shown = [trial['stimulusTable'][:,6] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]
            stimulusTable_stimulus_post_prob = [trial['stimulusTable'][:,7] if len(trial['stimulusTable']) else trial['stimulusTable'] for trial in trials ]



        for (variable_name, description) in trial_columns:
            try:
                variable = locals()[variable_name]
            except:
                raise ValueError(f"No such variable as {variable_name} is defined.")
            create_and_store_indexed_array(ndarray=variable, array_name=variable_name, description=description, nwbfile=nwbfile)



        # ------------------ Processed position, velocity, viewAngle ----------------- #
        # ---------- This information changes throughout the specific trial ----------- #

        pos_obj = Position(name="Position")
        view_angle_obj = CompassDirection(name='ViewAngle')

        timestamps = []
        pos_data = np.empty((0, 2))
        velocity_data = np.empty_like(pos_data)
        view_angle_data = []
        collision = []

        for trial in trials:
            trial_total_time = trial['start'] + epoch_start_nwb[0] + trial['time']
            timestamps.extend(trial_total_time.astype(np.float64, casting='same_kind'))

            # Padding exists since the time array is longer than all the other arrays.
            # The other arrays are all the same length
            padding = np.full((trial['time'].shape[0] - trial['position'].shape[0], 2), np.nan)
            trial_position = trial['position'][:, :-1]
            trial_velocity = trial['velocity'][:, :-1]
            trial_view_angle = trial['position'][:, -1]
            trial_collision = trial['collision']
            pos_data = np.concatenate([pos_data, trial_position, padding], axis=0)
            velocity_data = np.concatenate([velocity_data, trial_velocity, padding], axis=0)
            view_angle_data = np.concatenate([view_angle_data, trial_view_angle,
                                                padding[:, 0]], axis=0)
            collision = np.concatenate([collision, trial_collision, padding[:, 0]], axis=0)

        time = TimeSeries(name='Time',
                            data=H5DataIO(timestamps, compression="gzip"),
                            unit='s',
                            resolution=np.nan,
                            timestamps=H5DataIO(timestamps, compression="gzip"))


        pos_obj.add_spatial_series(
            SpatialSeries(
                name="SpatialSeries",
                data=H5DataIO(pos_data, compression="gzip"),
                reference_frame="(0,-80) is the start of the 'sample' region (or 'cue' region) which varies by maze and task.",
                description="The position of the animal by ViRMEN iteration.",
                unit='cm/s',
                # conversion=0.01,
                resolution=np.nan,
                timestamps=H5DataIO(timestamps, compression="gzip")
            )
        )
        velocity_ts = TimeSeries(
            name='Velocity',
            data=H5DataIO(velocity_data, compression="gzip"),
            description="The velocity of the animal by ViRMEN iteration.",
            unit='cm/s',
            resolution=np.nan,
            timestamps=H5DataIO(timestamps, compression="gzip")
        )

        view_angle_obj.add_spatial_series(
            SpatialSeries(
                name="SpatialSeries",
                data=H5DataIO(view_angle_data, compression="gzip"),
                reference_frame="unknown",
                description="The velocity view angle of the animal by ViRMEN iteration in the unit of degrees.",
                unit="degrees",
                resolution=np.nan,
                timestamps=H5DataIO(timestamps, compression="gzip")
            )
        )
        collision_ts = TimeSeries(name='Collision',
                                    data=H5DataIO(collision, compression="gzip"),
                                    unit='bool',
                                    resolution=np.nan,
                                    timestamps=H5DataIO(timestamps, compression="gzip"),
                                    description='boolean to indicate for each frame' ' whether collision was detected')

        behavioral_processing_module = check_module(nwbfile, 'behavior',
                                                    'contains processed behavioral data')
        behavioral_processing_module.add_data_interface(pos_obj)
        behavioral_processing_module.add_data_interface(velocity_ts)
        behavioral_processing_module.add_data_interface(view_angle_obj)
        behavioral_processing_module.add_data_interface(time)
        behavioral_processing_module.add_data_interface(collision_ts)

        return nwbfile

