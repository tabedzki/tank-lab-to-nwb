import nwbwidgets
from nwbwidgets.placefield import PlaceFieldWidget
from nwbwidgets.utils.timeseries import get_timeseries_in_units, get_timeseries_tt
from nwbwidgets.utils.widgets import interactive_output
from nwbwidgets.base import vis2widget

import pynwb
import numpy as np

from ipywidgets import widgets, BoundedFloatText, Dropdown, Checkbox


class TowersTaskPlaceFieldWidget(PlaceFieldWidget):

    def __init__(self, spatial_series,
                 **kwargs):
        trials = spatial_series.get_ancestor('NWBFile').trials

        self.units = spatial_series.get_ancestor('NWBFile').units
        start = 0
        stop = None
        self.pos_tt = get_timeseries_tt(spatial_series, start, stop)
        self.pos, self.unit = get_timeseries_in_units(spatial_series)
        self.pixel_width = (np.nanmax(self.pos) - np.nanmin(self.pos)) / 1000

        if trials is not None:
            left_towers = trials.left_cue_onset[:]
            left_towers = left_towers[~np.isnan(left_towers)]
            right_towers = trials.right_cue_onset[:]
            right_towers = right_towers[~np.isnan(right_towers)]
            tt = self.pos_tt
            ss = np.zeros_like(tt)
            ss[np.searchsorted(tt, right_towers)] += 1
            ss[np.searchsorted(tt, left_towers)] -= 1
            starts = np.searchsorted(tt, trials.start_time[:])
            ends = np.searchsorted(tt, trials.stop_time[:])
            states = np.zeros_like(tt)
            for start, end in zip(starts, ends):
                states[start:end] = np.cumsum(ss[start:end])

            print(ss)
            self.pos[:, 0] = self.pos[:, 1]
            self.pos[:, 1] = states

        self.pos, self.unit = get_timeseries_in_units(spatial_series)

        self.pixel_width = (np.nanmax(self.pos) - np.nanmin(self.pos)) / 1000

        style = {'description_width': 'initial'}
        bft_gaussian_x = BoundedFloatText(value=0.0184, min=0, max=99999, description='gaussian sd (cm)', style=style)
        bft_gaussian_y = BoundedFloatText(value=0, min=0, max=99999, description='gaussian sd (cm)', style=style)
        bft_speed = BoundedFloatText(value=0.03, min=0, max=99999, description='speed threshold (cm/s)', style=style)
        dd_unit_select = Dropdown(options=np.arange(len(self.units)), description='unit')
        cb_velocity = Checkbox(value=False, description='use velocity', indent=False)

        self.controls = dict(
            gaussian_sd_x=bft_gaussian_x,
            gaussian_sd_y=bft_gaussian_y,
            speed_thresh=bft_speed,
            index=dd_unit_select,
            use_velocity=cb_velocity
        )

        out_fig = interactive_output(self.do_rate_map, self.controls)

        self.children = [
            widgets.VBox([
                bft_gaussian_x,
                bft_gaussian_y,
                bft_speed,
                dd_unit_select,
                cb_velocity,
            ]),
            vis2widget(out_fig)
        ]
