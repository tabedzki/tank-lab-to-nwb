"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path
from typing import Optional, Union
import numpy as np
from datetime import datetime

from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface
import spikeextractors as se
from pynwb import NWBHDF5IO

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict

OptionalArrayType = Optional[Union[list, np.ndarray]]


def quick_write(session_id: str, session_description: str, save_path: str,
                sorting: se.SortingExtractor, overwrite: bool = False):
    """Write the sorting extractor and NWBFile through spikeextractors."""
    session_start = datetime.strptime(session_id[-16:-6], "%Y_%m_%d")
    nwbfile_kwargs = dict(
        session_id=session_id,
        session_description=session_description,
        session_start_time=session_start.astimezone()
    )
    se.NwbSortingExtractor.write_sorting(
        sorting=sorting,
        save_path=save_path,
        overwrite=overwrite,
        **nwbfile_kwargs
    )


class TowersProcessedNWBConverter(NWBConverter):
    """Primary conversion class for the Tank lab Towers task processing pipeline."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        SpikeGLXLFP=SpikeGLXLFPInterface,
        VirmenData=VirmenDataInterface,
    )

    def get_metadata(self):
        """Auto-populate as much metadata as possible."""
        vermin_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = vermin_file_path.stem
        date_text = [id_part for id_part in session_id.split('_') if id_part.isdigit()][0]
        session_start = datetime.strptime(date_text, "%Y%m%d")

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

    def run_conversion(self, metadata: dict, nwbfile_path: Optional[str] = None, save_to_file: bool = True,
                       conversion_options: Optional[dict] = None, overwrite: bool = False,
                       sorting: Optional[se.SortingExtractor] = None):
        """
        Build nwbfile object, auto-populate with minimal values if missing.

        Parameters
        ----------
        metadata : dict
        nwbfile_path : Optional[str], optional
        save_to_file : bool, optional
        conversion_options : Optional[dict], optional
        overwrite : bool, optional
        sorting : SortingExtractor, optional
            A SortingExtractor object to write to the NWBFile.
        recording_lfp : RecordingExtractor, optional
            A RecordingExtractor object to write to the NWBFile.
        timestamps : ArrayType, optional
            Array of timestamps obtained from tsync file.
        """
        run_conversion_kwargs = dict(
            metadata=metadata,
            conversion_options=conversion_options,
            save_to_file=False
        )
        if save_to_file:
            if nwbfile_path is None:
                raise TypeError("A path to the output file must be provided, but nwbfile_path got value None")

            if Path(nwbfile_path).is_file() and not overwrite:
                mode = "r+"
            else:
                mode = "w"

            with NWBHDF5IO(nwbfile_path, mode=mode) as io:
                if mode == "r+":
                    nwbfile = io.read()
                    run_conversion_kwargs.update(nwbfile=nwbfile)

                nwbfile = super().run_conversion(**run_conversion_kwargs)

                if sorting is not None:
                    se.NwbSortingExtractor.write_sorting(
                            sorting=sorting,
                            nwbfile=nwbfile
                    )

                io.write(nwbfile)
            print(f"NWB file saved at {nwbfile_path}!")
        else:
            return super().run_conversion(**run_conversion_kwargs)
