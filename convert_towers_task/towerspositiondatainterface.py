"""Authors: Cody Baker and Ben Dichter."""
from nwb_conversion_tools.utils import get_base_schema, get_schema_from_hdmf_class
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from pynwb.file import TimeIntervals
from pynwb.behavior import SpatialSeries, Position
from hdmf.backends.hdf5.h5_utils import H5DataIO
import os
import numpy as np
from scipy.io import loadmat
from .utils import check_module


class TankBehaviorInterface(BaseDataInterface):

    @classmethod
    def get_input_schema(cls):
        return dict(
            source_data=dict(
                required=['folder_path'],
                properties=dict(
                    folder_path=dict(type='string')
                )
            )
        )

    def __init__(self, **input_args):
        super().__init__(**input_args)

    def get_metadata_schema(self):
        metadata_schema = get_base_schema()

        # ideally most of this be automatically determined from pynwb docvals
        metadata_schema['properties']['SpatialSeries'] = get_schema_from_hdmf_class(SpatialSeries)
        required_fields = ['SpatialSeries']
        for field in required_fields:
            metadata_schema['required'].append(field)

        return metadata_schema

    def convert_data(self, nwbfile: NWBFile, metadata_dict: dict,
                     stub_test: bool = False, include_spike_waveforms: bool = False):
        session_path = self.input_args['folder_path']
        
        task_types = metadata_dict['task_types']

        subject_path, session_id = os.path.split(session_path)

        sleep_state_fpath = os.path.join(session_path, '{}--StatePeriod.mat'.format(session_id))

        exist_pos_data = any(os.path.isfile(os.path.join(session_path,
                                                         '{}__{}.mat'.format(session_id, task_type['name'])))
                             for task_type in task_types)

        if exist_pos_data:
            nwbfile.add_epoch_column('label', 'name of epoch')

        pos_obj = Position(name='_position')

        file = "..."
        matin = loadmat(file)
        pos_data = matin #[...]
        tt = matin #[...]
        conversion = 1 # need to change?
        exp_times = matin #[...]

        spatial_series_object = SpatialSeries(
            name='_{}_spatial_series',
            data=H5DataIO(pos_data, compression='gzip'),
            reference_frame='unknown', conversion=conversion,
            resolution=np.nan,
            timestamps=H5DataIO(tt, compression='gzip'))
        pos_obj.add_spatial_series(spatial_series_object)
    
        check_module(nwbfile, 'behavior', 'contains processed behavioral data').add_data_interface(pos_obj)
        for i, window in enumerate(exp_times):
            nwbfile.add_epoch(start_time=window[0], stop_time=window[1],
                              label='_' + str(i))

        trial_data = matin #[...]
        nwbfile.add_trial(start_time=trial_data[0], stop_time=trial_data[1],
                          error_run=trial_data[4], stim_run=trial_data[5], both_visit=trial_data[6])

