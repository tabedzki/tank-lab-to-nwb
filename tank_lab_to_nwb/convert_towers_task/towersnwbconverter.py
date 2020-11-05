"""Authors: Cody Baker and Ben Dichter."""
import os
from datetime import datetime, timedelta
from pathlib import Path

from dateutil.parser import parse as dateparse
from isodate import duration_isoformat
from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict


class TowersNWBConverter(NWBConverter):
    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        VirmenData=VirmenDataInterface,
    )

    def get_metadata(self):
        file_path = Path(self.data_interface_objects['VirmenData'].input_args['file_path'])
        session_id = file_path.stem

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
            Subject=dict(),
            SpikeGLXRecording=None,
            VirmenData=dict()
        )

        if file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=file_path)
            subject_data = session_data['log']['animal']

            key_map = dict(name='subject_id', importWeight='weight')
            [metadata['Subject'].update({key_map[k]: str(subject_data[k])}) for k in key_map if k in subject_data]

            age_in_iso_format = duration_isoformat(timedelta(weeks=subject_data['importAge']))
            metadata['Subject'].update(age=age_in_iso_format)
        else:
            print(f"Warning: no subject file detected for session {session_id}!")

        return metadata
