"""Authors: Cody Baker and Ben Dichter."""
from datetime import timedelta
from pathlib import Path

from dateutil.parser import parse as dateparse
from isodate import duration_isoformat
from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict, flatten_nested_dict, array_to_dt
from ndx_tank_metadata import LabMetaDataExtension, RigExtension, MazeExtension


class TowersRawNWBConverter(NWBConverter):
    """Secondary conversion class for the Towers task; does not sychronize with ttl or write spiking output."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        VirmenData=VirmenDataInterface,
    )

    def get_metadata(self):
        """Auto-populate as much metadata as possible."""
        vermin_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = vermin_file_path.stem
        date_text = [id_part for id_part in session_id.split('_') if id_part.isdigit()][0]
        session_start = dateparse(date_text, yearfirst=True)

        metadata = super().get_metadata()
        metadata['NWBFile'].update(
                session_start_time=session_start.astimezone(),
                session_id=session_id,
                institution="Princeton",
                lab="Tank"
        )
        metadata.update(
            Subject=dict(
                species="Mus musculus"
            )
        )

        if vermin_file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=vermin_file_path)
            subject_data = session_data['log']['animal']
            experiment_metadata = session_data['log']['version']
            age_in_iso_format = duration_isoformat(timedelta(weeks=subject_data['importAge']))

            metadata['Subject'].update(
                subject_id=subject_data['name'],
                age=age_in_iso_format
            )

            # Add lab metadata
            rig_extension = RigExtension(name='rig', **experiment_metadata['rig'])

            maze_extension = MazeExtension(name='mazes',
                                           description='description of the mazes')

            for maze in experiment_metadata['mazes']:
                flatten_maze_dict = flatten_nested_dict(maze)
                maze_extension.add_row(**flatten_maze_dict)

            num_trials = len(
                    [trial for epoch in session_data['log']['block'] for trial in epoch['trial']])
            session_end = array_to_dt(session_data['log']['session']['end']).isoformat()
            lab_meta_data = dict(
                name='LabMetaData',
                experiment_name='enter experiment name',
                world_file_name=experiment_metadata['name'],
                protocol_name='enter protocol name',
                stimulus_bank_path=subject_data['stimulusBank'] if subject_data[
                    'stimulusBank'] else '',
                commit_id=experiment_metadata['repository'],
                location=experiment_metadata['rig']['rig'],
                num_trials=num_trials,
                session_end_time=session_end,
                rig=rig_extension,
                mazes=maze_extension,
                # session_performance=0.,  # comment out to add (not in behavior file)
            )
            metadata['lab_meta_data'] = LabMetaDataExtension(**lab_meta_data)
        else:
            print(f"Warning: no subject file detected for session {session_id}!")

        return metadata
