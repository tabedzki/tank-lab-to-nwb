"""Authors: Cody Baker and Ben Dichter."""
from nwb_conversion_tools import NWBConverter
from . import spikeglxdatainterface  # keeping this here until the nwb-conv-tools update
from .tankpositiondatainterface import TankPositionInterface
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime
from dateutil.parser import parse as dateparse


class TankNWBConverter(NWBConverter):
    data_interface_classes = dict(
            SpikeGLXRecording=spikeglxdatainterface.SpikeGLXRecordingInterface,
            TankPosition=TankPositionInterface,
    )

    def __init__(self, **input_args):
        self._recording_type = 'SpikeGLXRecording'
        super().__init__(**input_args)

    def get_recording_type(self):
        return self._recording_type

    def get_metadata(self):
        session_path = self.data_interface_objects['NeuroscopeSorting'].input_args['folder_path']
        subject_path, session_id = os.path.split(session_path)

        date_text = "..."
        session_start = dateparse(date_text, yearfirst=True)

        # TODO: adjust this part to correctly pull relevant subject info
        subject_filepath = "..."
        subject_id = "..."
        if os.path.isfile(subject_filepath):
            subject_df = loadmat(subject_filepath)
            subject_data = {}
            for key in ['genotype', 'DOB', 'implantation', 'Probe', 'Surgery', 'virus injection', 'mouseID']:
                names = subject_df.iloc[:, 0]
                if key in names.values:
                    subject_data[key] = subject_df.iloc[np.argmax(names == key), 1]
            if isinstance(subject_data['DOB'], datetime):
                age = str(session_start - subject_data['DOB'])
            else:
                age = None
        else:
            age = 'unknown'
            subject_data = {}
            subject_data.update({'genotype': 'unknown'})
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
                age=age,
                species="Mus musculus",  # TODO: check species
                weight="..."  # TODO: fill in
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
