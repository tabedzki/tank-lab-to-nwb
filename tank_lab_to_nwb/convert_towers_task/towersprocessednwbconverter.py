"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path

from dateutil.parser import parse as dateparse
from nwb_conversion_tools import NWBConverter, SIPickleRecordingExtractorInterface, SIPickleSortingExtractorInterface

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict


class TowersProcessedNWBConverter(NWBConverter):
    """Primary conversion class for the Tank lab Towers task processing pipeline."""

    data_interface_classes = dict(
        SIRecording=SIPickleRecordingExtractorInterface,
        SISorting=SIPickleSortingExtractorInterface,
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

        if vermin_file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=vermin_file_path)
            subject_data = session_data['log']['animal']
            metadata.update(
                Subject=dict(
                    subject_id=subject_data['name']
                )
            )
        else:
            print(f"Warning: no subject file detected for session {session_id}!")

        return metadata
