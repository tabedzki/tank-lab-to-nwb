"""Authors: Alessio Buccino, Cody Baker, Szonja Weigl, and Ben Dichter."""
from pathlib import Path

from tank_lab_to_nwb import TowersNWBConverter

base_path = Path("D:/Neuropixels/")
spikeglx_file_path = base_path / "Neuropixels/A256_bank1_2020_09_30/A256_bank1_2020_09_30_g0/" \
                                 "A256_bank1_2020_09_30_g0_t0.imec0.ap.bin"
virmen_file_path = base_path / "TowersTask/PoissonBlocksReboot_cohort1_VRTrain6_E75_T_20181105.mat"
nwbfile_path = base_path / "TowersTask_stub.nwb"

if base_path.is_dir():
    if not nwbfile_path.is_file():
        input_args = dict(
            SpikeGLXRecording=dict(
                file_path=spikeglx_file_path,
            ),
            VirmenData=dict(
                file_path=virmen_file_path
            )
        )

        converter = TowersNWBConverter(**input_args)
        metadata = converter.get_metadata()
        converter.run_conversion(
            nwbfile_path=str(nwbfile_path.absolute()),
            metadata_dict=metadata,
            stub_test=True
        )
