"""Authors: Alessio Buccino, Cody Baker, Szonja Weigl, and Ben Dichter."""
from pathlib import Path
from isodate import duration_isoformat
from datetime import timedelta

from tank_lab_to_nwb import TowersProcessedNWBConverter

# Point to the base folder path for both recording data and Virmen
base_path = Path("D:/Neuropixels/")

# Name the NWBFile and point to the desired save path
nwbfile_path = base_path / "TowersTask.nwb"

# Point to the various files for the conversion
recording_folder = base_path / "Neuropixels" / "A256_bank1_2020_09_30" / "A256_bank1_2020_09_30_g0"
spikeinterface_folder = recording_folder / "spikeinterface"
raw_data_file = recording_folder / "A256_bank1_2020_09_30_g0_t0.imec0.ap.bin"
raw_data_lfp_file = recording_folder / "A256_bank1_2020_09_30_g0_t0.imec0.lf.bin"
recording_pickle_file = spikeinterface_folder / "raw.pkl"
sorting_pickle_file = spikeinterface_folder / "sorter1.pkl"
virmen_file_path = base_path / "TowersTask" / "PoissonBlocksReboot_cohort1_VRTrain6_E75_T_20181105.mat"

# Enter Session and Subject information here - uncomment any fields you want to include
session_description = "Enter session description here."

subject_info = dict(
    description="Enter optional subject description here",
    # weight="Enter subject weight here",
    # age=duration_isoformat(timedelta(days=0)),  # Enter the age of the subject in days
    # species="Mus musculus",
    # genotype="Enter subject genotype here",
    # sex="Enter subject sex here"
)

# Set some global conversion options here
stub_test = True
overwrite = True  # False if appending a file with sorting data - True if writing a brand new file with Virmen/SpikeGLX



# Run the conversion
source_data = dict(
    SpikeGLXRecording=dict(file_path=str(raw_data_file)),
    SpikeGLXLFP=dict(file_path=str(raw_data_lfp_file)),
    VirmenData=dict(file_path=str(virmen_file_path))
)
conversion_options = dict(
    SpikeGLXRecording=dict(stub_test=stub_test),
    SpikeGLXLFP=dict(stub_test=stub_test)
)
converter = TowersProcessedNWBConverter(source_data)
metadata = converter.get_metadata()
metadata['NWBFile'].update(session_description=session_description)
metadata['Subject'].update(subject_info)
converter.run_conversion(
    nwbfile_path=str(nwbfile_path),
    metadata=metadata,
    conversion_options=conversion_options,
    overwrite=overwrite
)
