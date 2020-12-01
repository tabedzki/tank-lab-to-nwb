"""Authors: Alessio Buccino, Cody Baker, Szonja Weigl, and Ben Dichter."""
from pathlib import Path

from tank_lab_to_nwb import TowersRawNWBConverter

base_path = Path("D:/Neuropixels/")
spikeglx_file_path = base_path / "Neuropixels/A256_bank1_2020_09_30/A256_bank1_2020_09_30_g0/" \
                                 "A256_bank1_2020_09_30_g0_t0.imec0.ap.bin"
virmen_file_path = base_path / "TowersTask/PoissonBlocksReboot_cohort1_VRTrain6_E75_T_20181105.mat"
nwbfile_path = base_path / "TowersTask_stub.nwb"

if base_path.is_dir():
    source_data = dict(
        SpikeGLXRecording=dict(file_path=str(spikeglx_file_path.absolute())),
        VirmenData=dict(file_path=str(virmen_file_path.absolute()))
    )
    conversion_options = dict(
        SpikeGLXRecording=dict(stub_test=True)
    )

    converter = TowersRawNWBConverter(source_data)
    metadata = converter.get_metadata()
    metadata['NWBFile'].update(session_description="Enter session description here.")
    converter.run_conversion(
        nwbfile_path=str(nwbfile_path.absolute()),
        metadata=metadata,
        conversion_options=conversion_options
    )
