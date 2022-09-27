# RFIND
# A tool to localize broadband RFI from interferometric visibilities.

from gettext import find
import logging
import pyuvdata
import numpy as np
from scipy import fft
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import find_peaks, savgol_filter
from matplotlib import pyplot as plt
from os import path

from rfind_utils import scale_visibility_data, mask_bright_sources, robust_rescale_1d

logger = logging.getLogger()
logger.setLevel("DEBUG")
logfile = logging.FileHandler(filename="RFIND.log")
logger.addHandler(logfile)

PLOTTYPE = "png"  # choose extension pdf or png for matplotlib
SPEED_OF_LIGHT = 2.99792458e8  # meters per second


class Observation(object):
    # @profile
    def __init__(
        self,
        filepath,
        outfileroot=None,
        num_grid_pts=100,
        GMRT_antenna_position_file=None,
        logger=logger,
        weights="uniform",
    ):
        """
        Defines an Observation class which holds the data from a phase calibrated CASA MS.

        Parameters:
        filepath: (str) Path to MS

        num_grid_pts: (int) Number of points to make a grid (corresponds to each size of the square grid)

        logger: Optional logger to be passed on to the class
        """
        self.logger = logger
        self.observation = pyuvdata.UVData()
        self.observation.read_ms(filepath, data_column="CORRECTED_DATA")
        if outfileroot is not None:
            self.outfileroot = outfileroot
        else:
            self.outfileroot = path.dirname(filepath)
        self.logger.info("Read MS file {:s}".format(filepath))
        self.num_grid_pts = num_grid_pts

        if GMRT_antenna_position_file is not None:
            assert GMRT_antenna_position_file.endswith(
                ".npz"
            ), "GMRT Antenna Position file must be a npz file."
            fh = np.load(GMRT_antenna_position_file, allow_pickle=True)

            self.observation.telescope_location = fh["telescope_location"]
            self.observation.telescope_location_lat_lon_alt = fh[
                "telescope_location_lat_lon_alt"
            ]
            self.observation.telescope_location_lat_lon_alt_degrees = fh[
                "telescope_location_lat_lon_alt_degrees"
            ]

            num_ants = len(fh["antenna_positions"])
            self.observation.antenna_positions[0:num_ants] = fh["antenna_positions"]

            self.logger.info(
                "Updated antenna positions and telescope position in the observation."
            )

            self.logger.info(
                "Rechecking the UVW array for consistency after updating positions."
            )
            self.observation.check()

        self.ant_lat, self.ant_lon, self.ant_alt = pyuvdata.utils.LatLonAlt_from_XYZ(
            self.observation.antenna_positions + self.observation.telescope_location
        )
        self.logger.info("Calculated antenna lat-lon-alt positions.")

        # make a unique baseline list.
        self.unique_bls = np.unique(self.observation.baseline_array)
        self.logger.info("Number of unique baselines: {}".format(len(self.unique_bls)))

        self.delay_spectrum_x_axis = fft.fftshift(
            fft.fftfreq(self.observation.Nfreqs, self.observation.channel_width)
        )

        self.w_delay = self.observation.uvw_array[:, 2] / SPEED_OF_LIGHT

        self.output_records = []

    def remove_baselines(self, baselines):
        """
        Remove a list of baselines from the unique baselines list.

        Parameters:
        baselines: (list, set): List or set of baselines to remove from existing baselines.

        """
        assert isinstance(baselines, (list, set))
        unique_bls = set(self.unique_bls)
        unique_bls.difference_update(baselines)

        self.unique_bls = np.array(list(unique_bls))
        self.logger.info("Removed baselines {}".format(baselines))
        self.logger.info("Number of unique baselines: {}".format(len(self.unique_bls)))

        return 0

    def remove_antenna(self, antenna):
        """
        Remove all baselines corresponding to an antenna

        Parameters:
        antenna: (int): Antenna to remove from the array

        """
        filt = np.where(
            (self.observation.ant_1_array == antenna)
            + (self.observation.ant_2_array == antenna)
        )
        bls_to_remove = set(self.observation.baseline_array[filt])
        self.logger.info(
            "Removing {} baselines corresponding to antenna {}".format(
                len(bls_to_remove), antenna
            )
        )
        self.remove_baselines(bls_to_remove)

        return 0

    def add_baselines(self, baselines):
        """
        Add a list of baselines to be processed

        Parameters:
        baselines: (list, set): List or set of baselines to add to existing baselines.

        """
        assert isinstance(baselines, (list, set))
        unique_bls = set(self.unique_bls)
        unique_bls.update(baselines)

        self.unique_bls = np.array(list(unique_bls))
        self.logger.info("Added baselines {}".format(baselines))
        self.logger.info("Number of unique baselines: {}".format(len(self.unique_bls)))

        return 0

    def add_antenna(self, antenna):
        """
        Add baselines corresponding to a single antenna to the list of baselines

        Parameters:
        antenna: (int): Antenna to add to the array
        """
        filt = np.where(
            (self.observation.ant_1_array == antenna)
            + (self.observation.ant_2_array == antenna)
        )
        bls_to_add = set(self.observation.baseline_array[filt])

        self.logger.info(
            "Adding {} baselines corresponding to antenna {}".format(
                len(bls_to_add), antenna
            )
        )
        self.add_baselines(bls_to_add)

        return 0

    def make_grid(self, lat_range=None, lon_range=None, plot=False):
        assert isinstance(lat_range, tuple) or lat_range is None
        assert isinstance(lon_range, tuple) or lon_range is None

        if lat_range is None:
            lat_min, lat_max = (
                np.rad2deg(min(self.ant_lat)),
                np.rad2deg(max(self.ant_lat)),
            )
            self.logger.debug(
                "Got lat range {}-{} from antenna positions".format(lat_min, lat_max)
            )
        else:
            lat_min, lat_max = lat_range
            self.logger.debug(
                "Got lat range {}-{} degrees from argument.".format(lat_min, lat_max)
            )

        if lon_range is None:
            lon_min, lon_max = (
                np.rad2deg(min(self.ant_lon)),
                np.rad2deg(max(self.ant_lon)),
            )
            self.logger.debug(
                "Got lon range {}-{} from antenna positions".format(lon_min, lon_max)
            )
        else:
            lon_min, lon_max = lon_range
            self.logger.debug(
                "Got lon range {}-{} from argument".format(lon_min, lon_max)
            )

        mean_alt = np.mean(self.ant_alt)

        lat_grid, lon_grid = np.meshgrid(
            np.linspace(lat_min, lat_max, self.num_grid_pts),
            np.linspace(lon_min, lon_max, self.num_grid_pts),
        )

        self.lat_grid = lat_grid.flatten()
        self.lon_grid = lon_grid.flatten()

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        self.grid_xyz = pyuvdata.utils.XYZ_from_LatLonAlt(
            np.deg2rad(self.lat_grid),
            np.deg2rad(self.lon_grid),
            mean_alt * np.ones_like(self.lat_grid),
        )

        if plot:
            plt.figure()
            plt.scatter(
                self.grid_xyz[:, 0], self.grid_xyz[:, 2], color="gray", alpha=0.5, s=1
            )
            ant_pos = (
                self.observation.antenna_positions + self.observation.telescope_location
            )
            plt.scatter(ant_pos[:, 0], ant_pos[:, 2], color="red", s=3)
            plt.xlabel("X (m)")
            plt.ylabel("Z (m)")
            plt.savefig(
                path.join(self.outfileroot, "antenna_position_grid_xyz." + PLOTTYPE)
            )
            plt.close()

            plt.figure()
            plt.scatter(self.lon_grid, self.lat_grid, color="gray", alpha=0.5, s=1)
            plt.scatter(
                np.rad2deg(self.ant_lon), np.rad2deg(self.ant_lat), color="red", s=3
            )
            plt.xlim([lon_min, lon_max])
            plt.ylim([lat_min, lat_max])
            plt.xlabel("Longitude (deg)")
            plt.ylabel("Latitude (deg)")
            plt.savefig(
                path.join(self.outfileroot, "antenna_position_grid_lonlat." + PLOTTYPE)
            )
            plt.close()

        return 0

    def make_intensity_map(self):
        # make the intensity map.
        self.logger.info("Creating a new intensity and weights map")
        self.rfi_intensity_map = np.zeros_like(self.lat_grid)
        self.weights_map = np.zeros_like(self.lat_grid)

        return 0

    def process_phased_data(self, plot=False, save=False):
        for i, bl in enumerate(self.unique_bls):
            self.process_single_phased_baseline(bl, plot=plot, save_1d_spec=save)

        return 0

    def make_delay_maps(self, plot=False):

        self.delay_maps = np.zeros((len(self.grid_xyz), len(self.unique_bls)))
        self.distance_weight_maps = np.zeros((len(self.grid_xyz), len(self.unique_bls)))

        for i, baseline in enumerate(self.unique_bls):

            ant1, ant2 = self.observation.baseline_to_antnums(baseline)
            ant1_pos = self.observation.antenna_positions[ant1, :]
            ant2_pos = self.observation.antenna_positions[ant2, :]

            d1 = np.sqrt(
                np.sum(
                    (self.grid_xyz - ant1_pos - self.observation.telescope_location)
                    ** 2,
                    axis=1,
                )
            )
            d2 = np.sqrt(
                np.sum(
                    (self.grid_xyz - ant2_pos - self.observation.telescope_location)
                    ** 2,
                    axis=1,
                )
            )

            self.delay_maps[:, i] = d1 - d2
            self.distance_weight_maps[:, i] = 1 / (d1 * d2)

            max_delay = np.max(np.abs(self.delay_maps[:, i]))

            if plot:
                fig, axes = plt.subplots(
                    nrows=2, ncols=1, figsize=(10, 14), squeeze=True
                )

                mesh = axes[0].pcolor(
                    self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                    self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                    self.delay_maps[:, i].reshape(
                        [self.num_grid_pts, self.num_grid_pts]
                    ),
                    cmap="RdBu",
                    vmin=-1 * max_delay,
                    vmax=max_delay,
                )
                plt.colorbar(mesh, ax=axes[0])
                axes[0].scatter(
                    np.rad2deg(self.ant_lon[ant1]),
                    np.rad2deg(self.ant_lat[ant1]),
                    marker=".",
                    color="r",
                )
                axes[0].scatter(
                    np.rad2deg(self.ant_lon[ant2]),
                    np.rad2deg(self.ant_lat[ant2]),
                    marker=".",
                    color="r",
                )
                axes[0].set_title(
                    "Delay map for baseline {:d}: Ants {:d}-{:d}".format(
                        baseline, ant1, ant2
                    )
                )

                mesh = axes[1].pcolor(
                    self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                    self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                    np.log10(
                        self.distance_weight_maps[:, i].reshape(
                            [self.num_grid_pts, self.num_grid_pts]
                        )
                    ),
                    cmap="viridis",
                )
                plt.colorbar(mesh, ax=axes[1])
                axes[1].scatter(
                    np.rad2deg(self.ant_lon[ant1]),
                    np.rad2deg(self.ant_lat[ant1]),
                    marker=".",
                    color="r",
                )
                axes[1].scatter(
                    np.rad2deg(self.ant_lon[ant2]),
                    np.rad2deg(self.ant_lat[ant2]),
                    marker=".",
                    color="r",
                )
                axes[1].set_title(
                    "Distance weight map for baseline {:d}: Ants {:d}-{:d}".format(
                        baseline, ant1, ant2
                    )
                )

                plt.savefig(
                    path.join(
                        self.outfileroot,
                        "delay_map_{:d}_{:02d}_{:02d}.{}".format(
                            baseline, ant1, ant2, PLOTTYPE
                        ),
                    )
                )
                plt.close()

        self.delay_maps /= SPEED_OF_LIGHT  # divide by speed of light

        return 0

    def process_single_phased_baseline(
        self, baseline, mask_sources=True, plot=False, save_1d_spec=False
    ):
        """
        Make and process the delays for a single baseline.

        Parameters:
        baseline: (int) Baseline number to process
        """
        assert (
            self.observation.phase_type == "phased"
        ), "Use this function for phased data, not for drift scan data"

        self.logger.info("Processing baseline {:d}".format(baseline))

        vis = self.observation.get_data(baseline, force_copy=True)

        freqs = self.observation.freq_array.squeeze()
        times = self.observation.get_times(baseline).squeeze()
        min_from_start = (times - np.min(times)) * 24 * 60  # times are in MJD

        vis = scale_visibility_data(
            vis,
            freqs,
            min_from_start,
            plot=False,
            niter=4,
            poly_freq_threshold=4,
            poly_freq_deg=5,
            poly_freq_fit_to_log=False,
        )

        ### not required since we are fitting and removing the bright sources.
        # subtract the mean along frequency axis.
        # r_vis = np.rollaxis(vis, 1)
        # r_vis -= r_vis.mean(axis=0)
        # vis = np.rollaxis(r_vis, 1)

        delay_spectrum = fft.fftshift(fft.fft(vis, axis=1))
        # sum along polarization
        delay_spectrum = np.sum(np.abs(delay_spectrum), axis=2)
        delay_spectrum -= np.median(
            delay_spectrum, axis=0
        )  # removes the bright sources

        light_curve = np.median(delay_spectrum, axis=1)
        mad_lightcurve = mad(delay_spectrum, axis=1)

        delay_spectrum = ((delay_spectrum.T-light_curve)/mad_lightcurve**2).T

        #divide the delay_spectrum by the median power

        delay_spectrum_1d = np.mean(delay_spectrum, axis=0)

        if mask_sources:
            delay_spectrum = mask_bright_sources(
                delay_spectrum, threshold=50, max_width=10, interp_range=20, expand=2
            )

        if plot:
            fig = plt.figure()
            plt.plot(
                self.delay_spectrum_x_axis / 1e-6,
                delay_spectrum_1d,
                label="original",
                alpha=0.5,
                drawstyle="steps-mid",
            )
            plt.plot(
                self.delay_spectrum_x_axis / 1e-6,
                np.mean(delay_spectrum, axis=0),
                label="masked",
                alpha=0.5,
                drawstyle="steps-mid",
            )
            ymin, ymax = plt.ylim()
            plt.ylim([0, ymax])
            plt.legend(loc="best")
            plt.xlabel(r"Delay ($\mu$s)")
            plt.savefig(
                path.join(
                    self.outfileroot,
                    "masked_spectrum_baseline_{:d}.{}".format(baseline, PLOTTYPE),
                )
            )
            plt.close(fig)

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), squeeze=True)
            ant1, ant2 = self.observation.baseline_to_antnums(baseline)

            axes[0].pcolor(
                freqs,
                min_from_start,
                np.log10(np.abs(np.sum(vis, axis=2))),
                cmap="viridis",
            )
            axes[0].set_xlabel("Frequency (MHz)")
            axes[0].set_ylabel("Time Elapsed (min)")
            axes[0].set_title(
                "Visibilities for baseline {:d} Ants-{:d}-{:d}".format(
                    baseline, ant1, ant2
                )
            )

            axes[1].pcolor(
                self.delay_spectrum_x_axis / 1e-6,
                min_from_start,
                np.log10(delay_spectrum),
                cmap="viridis",
            )
            axes[1].set_xlabel(r"Delay ($\mu$s)")
            axes[1].set_ylabel("Time Elapsed (min)")
            axes[1].set_title(
                "Delay spectrum for baseline {:d} Ants-{:d}-{:d}".format(
                    baseline, ant1, ant2
                )
            )

            plt.savefig(
                path.join(
                    self.outfileroot,
                    "vis_delay_spectrum_baseline_{:d}.{}".format(baseline, PLOTTYPE),
                )
            )
            plt.close(fig)

        # map the delays to the grid.
        self._threshold_and_save_peaks(
            baseline,
            delay_spectrum,
            plot=plot,
            save_1d_spec=save_1d_spec,
        )

        del delay_spectrum
        del vis

    def _threshold_and_add_phased_delay_spectrum_to_grid(
        self,
        baseline,
        delay_spectrum,
        threshold=6,
        min_peak_width=5,
        max_peak_width=50,
        plot=False,
    ):
        """
        Adds the sharp peaks above the threshold in the delay spectrum to the RFI map
        """
        self.logger.info(
            "Adding thresholded peaks to the grid baseline {:d}".format(baseline)
        )

        idx = np.where(self.unique_bls == baseline)

        current_delay_map = (self.delay_maps[:, idx]).squeeze()
        current_weight_map = (self.distance_weight_maps[:, idx]).squeeze()

        current_w_delay = self.w_delay[
            np.where(self.observation.baseline_array == baseline)
        ]

        self.logger.info("Shifting the delay spectrum")
        delayshifted_spectrum, x_axis = self._make_shifted_delay_spectrum(
            baseline, delay_spectrum, current_w_delay, plot=plot
        )

        self.logger.info(
            "Created a remapped delay spectrum with size {}, median = {:3.2e}, min_delay={:3.2e}, max_delay={:3.2e}".format(
                delayshifted_spectrum.shape,
                np.mean(delayshifted_spectrum),
                np.min(x_axis),
                np.max(x_axis),
            )
        )

        delayshifted_spectrum_1d = np.mean(delayshifted_spectrum, axis=0)
        self.logger.info(
            "Created a remapped 1d delay_spectrum with size {} and mean {:3.2e}".format(
                delayshifted_spectrum_1d.shape, np.mean(delayshifted_spectrum_1d)
            )
        )

        # rescale
        delayshifted_spectrum_1d = robust_rescale_1d(delayshifted_spectrum_1d)

        peaks = True
        try:
            peak_idxes, info = find_peaks(
                delayshifted_spectrum_1d,
                width=[min_peak_width, max_peak_width],
                height=10,  # minimum 6 sigma above for each pixel.
                prominence=10,
                distance=10,
            )
        except ValueError as e:
            peaks = False
            self.logger.error("Got error {}".format(e))

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
            axes[0].plot(
                x_axis / 1e-6,
                delayshifted_spectrum_1d,
                drawstyle="steps-mid",
                lw=1,
            )
            ymin, ymax = axes[0].get_ylim()
            axes[0].set_ylim([0, ymax])
            ymin = 0

            if peaks:
                axes[0].vlines(
                    x_axis[peak_idxes] / 1e-6,
                    ymin,
                    ymax,
                    color="r",
                    ls="dashed",
                    lw=1,
                )

            axes[1].semilogy(
                x_axis / 1e-6,
                delayshifted_spectrum_1d,
                drawstyle="steps-mid",
                lw=1,
            )
            ymin, ymax = axes[1].get_ylim()
            axes[1].set_ylim([1, ymax])

            if peaks:
                axes[1].vlines(
                    x_axis[peak_idxes] / 1e-6, ymin, ymax, color="r", ls="dashed", lw=1
                )

            axes[1].set_xlabel("Delay (us)")
            axes[0].set_ylabel("Amplitude")
            axes[1].set_ylabel("Amplitude")
            plt.title("Collapsed Delay Spectrum: {:d}".format(baseline))
            plt.savefig(
                path.join(
                    self.outfileroot,
                    "collapsed_shifted_delay_spectrum_{:d}.{}".format(
                        baseline, PLOTTYPE
                    ),
                )
            )
            plt.close(fig)

        # interpolate and add to the grid.
        # throwaway the very low values.
        # set 4-sigma threshold for now
        delayshifted_spectrum_1d[delayshifted_spectrum_1d < 4] = 0

        # self.output_dicts[baseline] = {
        #    "1d_delay_spectrum": delayshifted_spectrum_1d,
        #    "x_axis": x_axis,
        # }

        delay_1d_interp = interp1d(
            x_axis,
            delayshifted_spectrum_1d,
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )

        remapped_delay = delay_1d_interp(current_delay_map).squeeze()
        self.logger.debug("Remapped_delay.shape: {}".format(remapped_delay.shape))
        self.logger.debug(
            "Current_weight_map.shape: {}".format(current_weight_map.shape)
        )

        self.weights_map[np.isfinite(remapped_delay)] += 1
        self.rfi_intensity_map += np.nan_to_num(remapped_delay, 0)

        self.logger.info("Added to grid")

    def _threshold_and_save_peaks(
        self,
        baseline,
        delay_spectrum,
        plot=False,
        save_1d_spec=False,
        findpeaks_kwargs={
            "height": 6,
            "threshold": None,
            "distance": 10,
            "prominence": 7,
            "width": None,
            "wlen": None,
            "rel_height": 0.5,
            "plateau_size": None,
        },
        savgol_kwargs={
            "window_length": 61,
            "polyorder": 5,
            "mode": "constant",
            "cval": 0,
        },
    ):
        """
        Adds the sharp peaks above the threshold in the delay spectrum to the RFI map
        """
        self.logger.info(
            "Adding (delay, amplitude) pairs to the database {:d}".format(baseline)
        )

        current_w_delay = self.w_delay[
            np.where(self.observation.baseline_array == baseline)
        ]

        self.logger.info("Shifting the delay spectrum")
        delayshifted_spectrum, x_axis = self._make_shifted_delay_spectrum(
            baseline, delay_spectrum, current_w_delay, plot=plot
        )

        self.logger.info(
            "Created a remapped delay spectrum with size {}, median = {:3.2e}, min_delay={:3.2e}, max_delay={:3.2e}".format(
                delayshifted_spectrum.shape,
                np.mean(delayshifted_spectrum),
                np.min(x_axis),
                np.max(x_axis),
            )
        )

        delayshifted_spectrum_1d = np.mean(delayshifted_spectrum, axis=0)
        self.logger.info(
            "Created a remapped 1d delay_spectrum with size {} and mean {:3.2e}".format(
                delayshifted_spectrum_1d.shape, np.mean(delayshifted_spectrum_1d)
            )
        )

        # rescale
        delayshifted_spectrum_1d_rs = robust_rescale_1d(delayshifted_spectrum_1d)

        peaks = True
        try:
            smoothed_delayshifted_spectrum_1d = savgol_filter(
                delayshifted_spectrum_1d_rs, **savgol_kwargs
            )
            peak_idxes, info = find_peaks(
                smoothed_delayshifted_spectrum_1d, **findpeaks_kwargs
            )
        except ValueError as e:
            peaks = False
            self.logger.error("Got error {}".format(e))

        if save_1d_spec:
            np.savez(
                path.join(
                    self.outfileroot,
                    "collapsed_shifted_delay_spectrum_data_{:d}.npz".format(baseline),
                ),
                x_axis=x_axis,
                raw_ampl=delayshifted_spectrum_1d,
                scl_ampl=delayshifted_spectrum_1d_rs,
                smooth_ampl=smoothed_delayshifted_spectrum_1d,
                peaks=peak_idxes,
                peak_props=info,
            )

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
            axes[0].plot(
                x_axis / 1e-6,
                delayshifted_spectrum_1d_rs,
                drawstyle="steps-mid",
                lw=1,
            )
            axes[0].plot(
                x_axis / 1e-6,
                smoothed_delayshifted_spectrum_1d,
                drawstyle="steps-mid",
                lw=1,
            )
            ymin, ymax = axes[0].get_ylim()
            axes[0].set_ylim([0, ymax])
            ymin = 0

            if peaks:
                axes[0].vlines(
                    x_axis[peak_idxes] / 1e-6,
                    ymin,
                    ymax,
                    color="r",
                    ls="dashed",
                    lw=1,
                )

            axes[1].semilogy(
                x_axis / 1e-6,
                delayshifted_spectrum_1d_rs,
                drawstyle="steps-mid",
                lw=1,
            )
            axes[1].plot(
                x_axis / 1e-6,
                smoothed_delayshifted_spectrum_1d,
                drawstyle="steps-mid",
                lw=1,
            )
            ymin, ymax = axes[1].get_ylim()
            axes[1].set_ylim([1, ymax])

            if peaks:
                axes[1].vlines(
                    x_axis[peak_idxes] / 1e-6, ymin, ymax, color="r", ls="dashed", lw=1
                )

            axes[1].set_xlabel("Delay (us)")
            axes[0].set_ylabel("Integrations")
            axes[1].set_ylabel("Integrations")
            plt.title("Collapsed Delay Spectrum: {:d}".format(baseline))
            plt.savefig(
                path.join(
                    self.outfileroot,
                    "collapsed_shifted_delay_spectrum_{:d}.{}".format(
                        baseline, PLOTTYPE
                    ),
                )
            )
            plt.close(fig)

        # interpolate and add to the grid.
        # throwaway the very low values.
        # set 4-sigma threshold for now

        time_segment = 0
        for i, peak_idx in enumerate(peak_idxes):
            # add the peaks
            t = x_axis[peak_idx]
            amp = info["peak_heights"][i]
            width = info.get("widths", 1.0)
            self.output_records.append([time_segment, baseline, t, amp, width])
            self.logger.debug(
                "Appended [{:d}, {:d}, {:f}, {:f}, {:f}] to records.".format(
                    time_segment, baseline, t, amp, width
                )
            )

        self.logger.info(
            "Appended {:d} peaks to the record array for baseline {:d}".format(
                len(peak_idxes), baseline
            )
        )

    def _add_phased_delay_spectrum_to_grid(self, baseline, delay_spectrum, plot=False):
        """
        Add the delay spectrum to the grid based on the delay map and the w-delays
        """

        self.logger.info("Adding to the grid baseline {:d}".format(baseline))

        idx = np.where(self.unique_bls == baseline)

        current_delay_map = self.delay_maps[:, idx]

        current_w_delay = self.w_delay[
            np.where(self.observation.baseline_array == baseline)
        ]

        self._make_shifted_delay_spectrum(
            baseline, delay_spectrum, current_w_delay, plot=plot
        )

        for i in range(self.observation.Ntimes):
            delay_1d_interp = interp1d(
                self.delay_spectrum_x_axis - current_w_delay[i],
                delay_spectrum[i, :],
                kind="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            remapped_delay = delay_1d_interp(current_delay_map).squeeze()
            self.rfi_intensity_map += np.nan_to_num(remapped_delay, 0)
            self.weights_map[np.isfinite(remapped_delay)] += 1

        self.logger.info("Added to grid")

    def _make_shifted_delay_spectrum(
        self, baseline, delay_spectrum, current_w_delay, plot=True
    ):
        """
        Makes a 2d plot by realigning the dely spectrum by the w delay terms.

        Parameters:
        baseline: (int) Baseline number being processed

        delay_2d: (np.array, dtype = np.float) Power spectrum in delay space with shape [Nints, Nfreqs]. Nfreqs = Ndelays
        This has been summed over polarization.

        current_w_delay: (np.array, dtype = np.float) W-delay for current baseline with shape [Nints].
        """

        min_w_delay, max_w_delay = np.min(current_w_delay), np.max(current_w_delay)

        dt = self.delay_spectrum_x_axis[1] - self.delay_spectrum_x_axis[0]
        nfreqs = delay_spectrum.shape[1]
        num_delay_samp = np.int(np.ceil((max_w_delay - min_w_delay) / dt)) + nfreqs

        remapped_delay = np.zeros(
            [delay_spectrum.shape[0], num_delay_samp]
        ) + np.median(delay_spectrum)

        for i in range(delay_spectrum.shape[0]):
            shift = np.int(np.round((current_w_delay[i] - min_w_delay) / dt))
            remapped_delay[i, shift : shift + nfreqs] = delay_spectrum[i, :]

        x_axis = (
            np.arange(num_delay_samp) * dt + min_w_delay + self.delay_spectrum_x_axis[0]
        )

        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=True)
            plt.pcolor(
                x_axis / 1e-6,
                np.arange(self.observation.Ntimes),
                np.log10(remapped_delay),
                cmap="viridis",
            )
            plt.xlabel("Delay (us)")
            plt.ylabel("Integrations")
            plt.colorbar()
            plt.title("Shifted Delay Spectrum: {:d}".format(baseline))
            plt.savefig(
                path.join(
                    self.outfileroot,
                    "shifted_delay_spectrum_{:d}.{}".format(baseline, PLOTTYPE),
                )
            )

        return remapped_delay, x_axis

    def plot_rfi_and_weights_map(self, log=True):
        """
        Plots the RFI_intensity_maps and weight map.
        """
        plt.figure(figsize=(10, 10), dpi=300)

        if log:
            plt.pcolor(
                self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                np.log10(
                    self.rfi_intensity_map.reshape(
                        [self.num_grid_pts, self.num_grid_pts]
                    )
                ),
                cmap="viridis",
            )
        else:
            plt.pcolor(
                self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                self.rfi_intensity_map.reshape([self.num_grid_pts, self.num_grid_pts]),
                cmap="viridis",
            )

        plt.colorbar()

        plt.scatter(
            np.rad2deg(self.ant_lon), np.rad2deg(self.ant_lat), marker=".", color="r"
        )

        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")

        plt.title("RFI Map")
        plt.savefig(path.join(self.outfileroot, "rfi_map." + PLOTTYPE))
        plt.close()

        plt.figure(figsize=(10, 10), dpi=300)

        reshaped_rfi_map = self.rfi_intensity_map.reshape(
            [self.num_grid_pts, self.num_grid_pts]
        )

        prewitt_x = ndimage.prewitt(reshaped_rfi_map, axis=0)
        prewitt_y = ndimage.prewitt(reshaped_rfi_map, axis=1)
        prew = np.hypot(prewitt_x, prewitt_y)

        if log:
            plt.pcolor(
                self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                np.log10(prew),
                cmap="viridis",
            )
        else:
            plt.pcolor(
                self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
                prew,
                cmap="viridis",
            )

        plt.colorbar()

        plt.scatter(
            np.rad2deg(self.ant_lon), np.rad2deg(self.ant_lat), marker=".", color="r"
        )

        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")

        plt.title("RFI Edges Map")
        plt.savefig(path.join(self.outfileroot, "rfi_edges_map." + PLOTTYPE))
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.pcolor(
            self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
            self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
            self.weights_map.reshape([self.num_grid_pts, self.num_grid_pts]),
            cmap="viridis",
        )

        plt.colorbar()

        plt.scatter(
            np.rad2deg(self.ant_lon), np.rad2deg(self.ant_lat), marker=".", color="r"
        )
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")

        plt.title("Weight Map")
        plt.savefig(path.join(self.outfileroot, "weight_map." + PLOTTYPE))
        plt.close()

    def save_rfi_and_weights_map(self):
        """
        Saves the RFI_intensity_maps and weight map.
        """
        np.savez(
            path.join(self.outfileroot, "rfi_map.npz"),
            rfi_intensity_map=self.rfi_intensity_map.reshape(
                [self.num_grid_pts, self.num_grid_pts]
            ),
            weights_map=self.weights_map.reshape(
                [self.num_grid_pts, self.num_grid_pts]
            ),
            lon_grid=self.lon_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
            lat_grid=self.lat_grid.reshape([self.num_grid_pts, self.num_grid_pts]),
        )

        return 0

    # @profile
    def save_output_records(self):

        output = np.rec.fromrecords(
            self.output_records, names="timesegment, baseline, delay, amplitude"
        )

        enu_positions = pyuvdata.utils.ENU_from_ECEF(
            self.observation.antenna_positions + self.observation.telescope_location,
            self.observation.telescope_location_lat_lon_alt[0],
            self.observation.telescope_location_lat_lon_alt[1],
            self.observation.telescope_location_lat_lon_alt[2],
        )

        baselines = self.unique_bls

        np.savez(
            path.join(self.outfileroot, "output_records.npz"),
            positions=enu_positions,
            baselines=baselines,
            delays=output,
        )
        self.logger.info(
            "Wrote {}".format(path.join(self.outfileroot, "output_records.npz"))
        )
