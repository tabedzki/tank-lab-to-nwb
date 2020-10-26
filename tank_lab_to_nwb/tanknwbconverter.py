"""Authors: Cody Baker and Ben Dichter."""
from nwb_conversion_tools import NWBConverter
from nwb_conversion_tools import SpikeGLXRecordingInterface
from tank_lab_to_nwb.convert_towers_task.towerspositiondatainterface import TowersPositionInterface
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime, timedelta
from dateutil.parser import parse as dateparse

from .utils import convert_mat_file_to_dict


class TankNWBConverter(NWBConverter):
    data_interface_classes = dict(
            SpikeGLXRecording=SpikeGLXRecordingInterface,
            TankPosition=TowersPositionInterface,
    )

    def __init__(self, **input_args):
        self._recording_type = 'SpikeGLXRecording'
        super().__init__(**input_args)

    def get_recording_type(self):
        return self._recording_type

    def get_metadata(self):
        session_path = self.data_interface_objects['NeuroscopeSorting'].input_args['folder_path']
        subject_path, session_id = os.path.split(session_path)

        session_name = os.path.splitext(session_id)[0]
        date_text = [name for name in session_name.split('_') if name.isdigit()][0]
        session_start = dateparse(date_text, yearfirst=True)

        if os.path.isfile(session_path):
            session_data = convert_mat_file_to_dict(mat_file_name=session_path)
            subject_data = session_data['log']['animal']

            for key in ['name', 'importAge', 'normWeight', 'genotype']:
                if key not in subject_data.keys():
                    subject_data[key] = 'unknown'

            subject_id = subject_data['name']
            age = subject_data['importAge']
            # TODO: check if assumption correct (importAge is in days)
            if 'importDate' in subject_data:
                date_text = '/'.join(map(str, subject_data['importDate']))
                if isinstance(age, int):
                    subject_data['date_of_birth'] = \
                        datetime.strptime(date_text, "%Y/%m/%d") - timedelta(days=age)

        else:
            subject_id = 'unknown'
            age = 'unknown'
            subject_data = {}
            subject_data.update({'genotype': 'unknown',
                                 'normWeight': 'unknown',
                                 'date_of_birth': 'unknown'})
            print(f"Warning: no subject file detected for session {session_path}!")

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
                subject_id=subject_id,
                age=str(age),  # TODO: check format (days?)
                species="Mus musculus",  # TODO: check species
                weight=str(subject_data['normWeight']),
                genotype=subject_data['genotype'],
                date_of_birth=subject_data['date_of_birth']
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
            TankPosition=dict()
        )

        return metadata
