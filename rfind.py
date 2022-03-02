# RFIND
# A tool to localize broadband RFI from interferometric visibilities.

import logging
import pyuvdata
import numpy as np
from scipy import fft
from scipy.interpolate import interp1d
from scipy import ndimage
from matplotlib import pyplot as plt
from os import path

logger = logging.getLogger()
logger.setLevel("DEBUG")
logfile = logging.FileHandler(filename="RFIND.log")
logger.addHandler(logfile)

PLOTTYPE = 'png' # choose extension pdf or png for matplotlib
SPEED_OF_LIGHT = 2.99792458e8  # meters per second


class Observation(object):
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
        self.observation.read_ms(filepath, data_column="DATA")
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
            # self.observation.telescope_name = fh['telescope_name']
            # self.observation.antenna_diameters = fh['antenna_diameters']
            # self.observation.antenna_names = fh['antenna_names']
            # self.observation.antenna_numbers = fh['antenna_numbers']
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

        # self.logger.info('Converting data to drift-scan mode.')
        # self.observation.unphase_to_drift()
        # self.logger.info('Converted data to drift-scan mode.')

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
            plt.savefig(path.join(self.outfileroot, "antenna_position_grid_xyz."+PLOTTYPE))
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
            plt.savefig(path.join(self.outfileroot, "antenna_position_grid_lonlat."+PLOTTYPE))
            plt.close()

        return 0

    def make_intensity_map(self):
        # make the intensity map.
        self.logger.info("Creating a new intensity and weights map")
        self.rfi_intensity_map = np.zeros_like(self.lat_grid)
        self.weights_map = np.zeros_like(self.lat_grid)

        return 0

    def process_phased_data(self):
        # make a delay map
        self.make_delay_map()

        return 0

    def process_drift_data(self, plot=False):
        for i, bl in enumerate(self.unique_bls):
            self.process_single_drift_baseline(bl, plot=plot)

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
                        "delay_map_{:d}_{:02d}_{:02d}.{}".format(baseline, ant1, ant2, PLOTTYPE),
                    )
                )
                plt.close()

        self.delay_maps /= SPEED_OF_LIGHT  # divide by speed of light

        return 0

    def process_single_phased_baseline(self, baseline, plot=False):
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

        vis /= np.mean(np.abs(vis), axis=0)
        # vis = (vis.T - np.mean(vis.T, axis=0)).T # subtract the DC component along the frequency axis.
        # vis = (vis.T / np.mean(np.abs(vis.T), axis=0)).T

        delay_spectrum = fft.fftshift(fft.fft(vis, axis=1))

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), squeeze=True)
            ant1, ant2 = self.observation.baseline_to_antnums(baseline)

            axes[0].pcolor(
                self.observation.freq_array,
                np.arange(self.observation.Ntimes),
                np.log10(np.abs(np.sum(vis, axis=2))),
                cmap="viridis",
            )
            axes[0].set_xlabel("Frequency (MHz)")
            axes[0].set_ylabel("Integrations")
            axes[0].set_title(
                "Visibilities for baseline {:d} Ants-{:d}-{:d}".format(
                    baseline, ant1, ant2
                )
            )

            axes[1].pcolor(
                self.delay_spectrum_x_axis / 1e-6,
                np.arange(self.observation.Ntimes),
                np.log10(np.abs(np.sum(delay_spectrum, axis=2))),
                cmap="viridis",
            )
            axes[1].set_xlabel(r"Delay ($\mu$s)")
            axes[1].set_ylabel("Integrations")
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
            plt.close()

        # map the delays to the grid.
        self._add_phased_delay_spectrum_to_grid(baseline, delay_spectrum, plot=plot)

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

        # delay_1d = np.abs(np.sum(np.sum(delay_spectrum, axis = 2), axis = 0))

        # sum along polarization
        delay_2d = np.sum(np.abs(delay_spectrum), axis=2)
        delay_2d -= np.median(delay_2d)

        if plot:
            self._make_shifted_delay_spectrum_plot(baseline, delay_2d, current_w_delay)

        for i in range(self.observation.Ntimes):
            delay_1d_interp = interp1d(
                self.delay_spectrum_x_axis + current_w_delay[i],
                delay_2d[i, :],
                kind="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            remapped_delay = delay_1d_interp(current_delay_map).squeeze()
            self.rfi_intensity_map += np.nan_to_num(remapped_delay, 0)
            self.weights_map[np.isfinite(remapped_delay)] += 1

        self.logger.info("Added to grid")

    def _make_shifted_delay_spectrum_plot(self, baseline, delay_2d, current_w_delay):
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
        nfreqs = delay_2d.shape[1]
        num_delay_samp = np.int(np.ceil((max_w_delay - min_w_delay) / dt)) + nfreqs

        remapped_delay = np.zeros([delay_2d.shape[0], num_delay_samp])

        for i in range(delay_2d.shape[0]):
            shift = np.int(np.round((current_w_delay[i] - min_w_delay) / dt))
            remapped_delay[i, shift : shift + nfreqs] = delay_2d[i, :]

        x_axis = np.arange(
            self.delay_spectrum_x_axis[0] - min_w_delay,
            self.delay_spectrum_x_axis[-1] + max_w_delay - min_w_delay,
            dt,
        )

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=True)
        plt.pcolor(np.log10(remapped_delay), cmap="viridis")
        plt.xlabel("Delay - {:3.2f} (microsecond)".format(min_w_delay / 1e-6))
        plt.ylabel("Integrations")
        plt.savefig(
            path.join(
                self.outfileroot, "shifted_delay_spectrum_{:d}.{}".format(baseline, PLOTTYPE)
            )
        )

        return 0

    def process_single_drift_baseline(self, baseline, plot=False):
        """
        Make and process the delays for a single baseline.

        Parameters:
        baseline: (int) Baseline number to process
        """
        assert (
            self.observation.phase_type == "drift"
        ), "Use this function for drift scan data, not for phased data"

        self.logger.info("Processing baseline {:d}".format(baseline))

        vis = self.observation.get_data(baseline, force_copy=True)

        vis /= np.mean(np.abs(vis), axis=0)
        vis = (
            vis.T - np.mean(vis.T, axis=0)
        ).T  # subtract the DC component along the frequency axis.
        # vis = (vis.T / np.mean(np.abs(vis.T), axis=0)).T

        delay_spectrum = fft.fftshift(fft.fft(vis, axis=1))

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), squeeze=True)
            ant1, ant2 = self.observation.baseline_to_antnums(baseline)

            axes[0].pcolor(
                self.observation.freq_array,
                np.arange(self.observation.Ntimes),
                np.log10(np.abs(np.sum(vis, axis=2))),
                cmap="viridis",
            )
            axes[0].set_xlabel("Frequency (MHz)")
            axes[0].set_ylabel("Integrations")
            axes[0].set_title(
                "Visibilities for baseline {:d} Ants-{:d}-{:d}".format(
                    baseline, ant1, ant2
                )
            )

            axes[1].pcolor(
                self.delay_spectrum_x_axis / 1e-6,
                np.arange(self.observation.Ntimes),
                np.log10(np.abs(np.sum(delay_spectrum, axis=2))),
                cmap="viridis",
            )
            axes[1].set_xlabel(r"Delay ($\mu$s)")
            axes[1].set_ylabel("Integrations")
            axes[1].set_title(
                "Delay spectrum for baseline {:d} Ants-{:d}-{:d}".format(
                    baseline, ant1, ant2
                )
            )

            plt.savefig(
                path.join(
                    self.outfileroot,
                    "vis_delay_spectrum_baseline_{:d}.".format(baseline, PLOTTYPE),
                )
            )
            plt.close()

        # map the delays to the grid.
        self._add_drift_delay_spectrum_to_grid(baseline, delay_spectrum, plot=plot)

    def _add_drift_delay_spectrum_to_grid(self, baseline, delay_spectrum, plot=False):
        """
        Add the delay spectrum to the grid based on the delay map and the w-delays
        """

        self.logger.info("Adding to the grid baseline {:d}".format(baseline))

        idx = np.where(self.unique_bls == baseline)

        current_delay_map = (self.delay_maps[:, idx]).squeeze()
        current_weight_map = (self.distance_weight_maps[:, idx]).squeeze()

        # if the data is phased to drift_scan, the w_delay will be constant.
        current_w_delay = np.mean(
            self.w_delay[np.where(self.observation.baseline_array == baseline)]
        )

        # collapse along time and polarization.

        delay_1d = np.sum(np.sum(np.abs(delay_spectrum), axis=2), axis=0)
        delay_1d -= np.median(delay_1d)

        delay_1d_interp = interp1d(
            self.delay_spectrum_x_axis + current_w_delay,
            delay_1d,
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )

        remapped_delay = (
            delay_1d_interp(current_delay_map).squeeze() * current_weight_map
        )
        self.logger.debug("Remapped_delay.shape: {}".format(remapped_delay.shape))
        self.logger.debug(
            "Current_weight_map.shape: {}".format(current_weight_map.shape)
        )

        self.weights_map[np.isfinite(remapped_delay)] += 1
        self.rfi_intensity_map += np.nan_to_num(remapped_delay, 0)

        self.logger.info("Added to grid")

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
        plt.savefig(path.join(self.outfileroot, "rfi_map."+PLOTTYPE))
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
        plt.savefig(path.join(self.outfileroot, "rfi_edges_map."+PLOTTYPE))
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
        plt.savefig(path.join(self.outfileroot, "weight_map."+PLOTTYPE))
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
