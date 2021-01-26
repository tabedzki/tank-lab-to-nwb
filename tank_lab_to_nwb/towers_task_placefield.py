from nwbwidgets.placefield import PlaceFieldWidget
from nwbwidgets.utils.timeseries import get_timeseries_in_units
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import BoundedFloatText, Dropdown, Checkbox


class TowersTaskPlaceFieldWidget(PlaceFieldWidget):

    def get_pixel_width(self, bin_num):
        self.pixel_width = [(np.nanmax(self.pos[:,0]) - np.nanmin(self.pos[:,0])) / bin_num, int(1)]

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
            bad_indices = np.isnan(self.pos[:, 0]) | np.isnan(self.pos[:, 1])
            good_indices = ~bad_indices
            good_tt = self.pos_tt[good_indices]
            self.pos_tt = good_tt
            good_x = self.pos[good_indices, 0]
            good_y = self.pos[good_indices, 1]
            self.pos = np.column_stack([good_x, good_y])

    def get_controls(self):
        style = {'description_width': 'initial'}
        bft_gaussian_x = BoundedFloatText(value=0.0184, min=0, max=99999, description='gaussian sd x (cm)', style=style)
        bft_gaussian_y = BoundedFloatText(value=0, min=0, max=99999, description='gaussian sd y (cm)', style=style)
        bft_bin_num = BoundedFloatText(value=15, min=0, max=99999, description='number of bins', style=style)
        bft_speed = BoundedFloatText(value=0.03, min=0, max=99999, description='speed threshold (m/s)', style=style)
        dd_unit_select = Dropdown(options=np.arange(len(self.units)), description='unit')
        cb_velocity = Checkbox(value=False, description='use velocity', indent=False, disabled=True)

        return bft_gaussian_x, bft_gaussian_y, bft_bin_num, bft_speed, dd_unit_select, cb_velocity


    def do_rate_map(self, index=0, speed_thresh=0.03, gaussian_sd_x=0.0184, gaussian_sd_y=0.0184, bin_num=15,
                    use_velocity=False):
        self.get_pixel_width(bin_num)
        occupancy, filtered_firing_rate, [edges_x, edges_y] = self.compute_twodim_firing_rate(self.pixel_width[0],
                                                                                              index=index,
                                                                                              speed_thresh=speed_thresh,
                                                                                              gaussian_sd_x=gaussian_sd_x,
                                                                                              gaussian_sd_y=gaussian_sd_y,
                                                                                              use_velocity=use_velocity)


        fig, ax = plt.subplots(figsize=(10, 10))
        filtered_firing_rate = filtered_firing_rate[:, edges_x[1:] > 0]
        edges_x = edges_x[edges_x >= 0]
        im = ax.imshow(filtered_firing_rate,
                       extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
                       aspect='auto')
        ax.set_xlabel('x ({})'.format(self.unit))
        ax.set_ylabel('y (Evidence)')

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('firing rate (Hz)')

        return fig

