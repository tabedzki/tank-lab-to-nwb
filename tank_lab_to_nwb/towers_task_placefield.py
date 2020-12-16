from nwbwidgets.placefield import PlaceFieldWidget
from nwbwidgets.utils.timeseries import get_timeseries_in_units
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import BoundedFloatText, Dropdown, Checkbox


class TowersTaskPlaceFieldWidget(PlaceFieldWidget):

    def get_pixel_width(self):
        self.pixel_width = [(np.nanmax(self.pos) - np.nanmin(self.pos)) / 1000, 1]

    def get_position(self, spatial_series):
        self.pos, self.unit = get_timeseries_in_units(spatial_series)
        trials = spatial_series.get_ancestor('NWBFile').trials
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

            self.pos[:, 0] = self.pos[:, 1]
            self.pos[:, 1] = states

    def get_controls(self):
        style = {'description_width': 'initial'}
        bft_gaussian_x = BoundedFloatText(value=0.0184, min=0, max=99999, description='gaussian sd x (cm)', style=style)
        bft_gaussian_y = BoundedFloatText(value=0, min=0, max=99999, description='gaussian sd y (cm)', style=style)
        bft_speed = BoundedFloatText(value=0, min=0, max=99999, description='speed threshold (cm/s)', style=style)
        dd_unit_select = Dropdown(options=np.arange(len(self.units)), description='unit')
        cb_velocity = Checkbox(value=False, description='use velocity', indent=False)

        return bft_gaussian_x, bft_gaussian_y, bft_speed, dd_unit_select, cb_velocity


    def do_rate_map(self, index=0, speed_thresh=0.03, gaussian_sd_x=0.0184, gaussian_sd_y=0.0184, use_velocity=False):
        occupancy, filtered_firing_rate, [edges_x, edges_y] = self.compute_twodim_firing_rate(index=index,
                                                                                         speed_thresh=speed_thresh,
                                                                                         gaussian_sd_x=gaussian_sd_x,
                                                                                         gaussian_sd_y=gaussian_sd_y,
                                                                                         use_velocity=use_velocity)


        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(filtered_firing_rate,
                       extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
                       aspect='auto')

        ax.set_xlabel('x ({})'.format(self.unit))
        ax.set_ylabel('y (Evidence)')
        ax.set_ylim(np.nanmin(self.pos[:, 1] - 1), np.nanmax(self.pos[:, 1] + 2))
        ax.set_xlim(0, np.nanmax(self.pos[:, 0] + 0.01))
        cbar = plt.colorbar(im)
        im.set_clim(0, np.nanmean(filtered_firing_rate))
        cbar.ax.set_ylabel('firing rate (Hz)')

        return fig

