"""Authors: Cody Baker and Ben Dichter."""
import os
from datetime import datetime, timedelta

from dateutil.parser import parse as dateparse
from isodate import duration_isoformat
from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface

from .towerspositiondatainterface import TowersPositionInterface
from ..utils import convert_mat_file_to_dict


class TowersNWBConverter(NWBConverter):
    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        TowersPosition=TowersPositionInterface,
    )

    def __init__(self, **input_args):
        self._recording_type = 'SpikeGLXRecording'
        super().__init__(**input_args)

    def get_recording_type(self):
        return self._recording_type

    def get_metadata(self):
        session_path = self.data_interface_objects['TowersPosition'].input_args['folder_path']
        subject_path, session_id = os.path.split(session_path)

        session_name = os.path.splitext(session_id)[0]
        date_text = [name for name in session_name.split('_') if name.isdigit()][0]
        session_start = dateparse(date_text, yearfirst=True)

        metadata = dict(
            NWBFile=dict(
                identifier=session_id,
                session_start_time=session_start.astimezone(),
                file_create_date=datetime.now().astimezone(),
                session_id=session_id,
                institution="Princeton",
                lab="Tank"
            ),
            Subject=dict(
            ),
            # self.get_recording_type(): {
            #     'Ecephys': {
            #         'subset_channels': all_shank_channels,
            #         'Device': [{
            #             'description': session_id + '.xml'
            #         }],
            #         'ElectrodeGroup': [{
            #             'name': f'shank{n+1}',
            #             'description': f'shank{n+1} electrodes'
            #         } for n, _ in enumerate(shank_channels)],
            #         'Electrodes': [
            #             {
            #                 'name': 'shank_electrode_number',
            #                 'description': '0-indexed channel within a shank',
            #                 'data': shank_electrode_number
            #             },
            #             {
            #                 'name': 'group_name',
            #                 'description': 'the name of the ElectrodeGroup this electrode is a part of',
            #                 'data': shank_group_name
            #             }
            #         ],
            #         'ElectricalSeries': {
            #             'name': 'ElectricalSeries',
            #             'description': 'raw acquisition traces'
            #         }
            #     }
            # },
            SpikeGLXRecording=None,
            TowersPosition=dict()
        )

        if os.path.isfile(session_path + ".mat"):
            session_data = convert_mat_file_to_dict(mat_file_name=session_path)
            subject_data = session_data['log']['animal']

            key_map = dict(name='subject_id', importWeight='weight')
            [metadata['Subject'].update({key_map[k]: str(subject_data[k])}) for k in key_map if k in subject_data]

            age_in_iso_format = duration_isoformat(timedelta(weeks=subject_data['importAge']))
            metadata['Subject'].update(age=age_in_iso_format)
        else:
            print(f"Warning: no subject file detected for session {session_path}!")

        return metadata
