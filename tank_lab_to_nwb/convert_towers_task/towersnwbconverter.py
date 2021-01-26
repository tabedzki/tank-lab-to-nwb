"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path
from typing import Optional, Union
import numpy as np

from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface
import spikeextractors as se

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict

OptionalArrayType = Optional[Union[list, np.ndarray]]
PathType = Union[Path, str]


class TowersNWBConverter(NWBConverter):
    """Primary conversion class for the Tank lab Towers task processing pipeline."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        SpikeGLXLFP=SpikeGLXLFPInterface,
        VirmenData=VirmenDataInterface,
    )

    def __init__(self, source_data, ttl_source: PathType):
        """
        Initialize the NWBConverter object.

        Parameters
        ----------
        ttl_source : PathType
            Path to data file containing the TTL signals to use for synchronizing.
        """
        super().__init__(source_data=source_data)
        recording = se.SpikeGLXRecordingExtractor(ttl_source)
        ttl, states = recording.get_ttl_events()
        rising_times = ttl[states == 1]

        assert len(rising_times) > 0, f"No TTL events found in ttl_source file ({ttl_source})."
        start_time = recording.frame_to_time(rising_times[0])

        for interface_name in ['SpikeGLXRecording', 'SpikeGLXLFP']:
            if interface_name in self.data_interface_objects:  # specified in source_data
                interface_extractor = self.data_interface_objects[interface_name].recording_extractor
                re_start_frame = int(interface_extractor.time_to_frame(start_time))
                self.data_interface_objects[interface_name].recording_extractor = se.SubRecordingExtractor(
                    parent_recording=interface_extractor,
                    start_frame=re_start_frame
                )

    def get_metadata(self):
        vermin_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = vermin_file_path.stem

        metadata = super().get_metadata()
        metadata['NWBFile'].update(
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
