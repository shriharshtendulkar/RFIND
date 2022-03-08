from rfind import Observation

filename = "/DATA/shriharsh/GMRT_RFI/TST2524/EMPTY2split.ms"
outfileroot = "/DATA/shriharsh/GMRT_RFI/TST2524/rfifind_trial_"
GMRT_antenna_position_file = "/home/shriharsh/Code/RFIND/GMRT_antenna_positions.npz"

obs = Observation(
    filename,
    outfileroot=outfileroot,
    num_grid_pts=700,
    GMRT_antenna_position_file=GMRT_antenna_position_file,
)
obs.logger.setLevel("INFO")
obs.make_grid(lat_range=(19.050, 19.125), lon_range=(74.000, 74.100), plot=False)
# obs.make_grid()
obs.make_delay_maps(plot=False)
obs.make_intensity_map()

# obs.observation.unphase_to_drift()

# for i in range(14,30):
#    obs.remove_antenna(i)

# obs.process_drift_data(plot=False)
# obs.process_single_drift_baseline(67586, plot=False) # 0-1
# obs.process_single_drift_baseline(67587, plot=False) # 0-2
# obs.process_single_drift_baseline(69635, plot=False) # 1-2

# obs.process_single_drift_baseline(69647, plot=True) # 1-14
# obs.process_single_drift_baseline(69646, plot=True) # 1-13
# obs.process_single_drift_baseline(94223, plot=True) # 13-14

# obs.process_single_phased_baseline(90128, plot=True)

# choose three antennas. 14, 19, 24 (nearest arm antennas)

obs.process_single_phased_baseline(96276, plot=True)  # 14-19
obs.process_single_phased_baseline(106521, plot=True)  # 19-24
obs.process_single_phased_baseline(96281, plot=True)  # 14-24

obs.process_single_phased_baseline(67586, plot=True)  # 0-1
obs.process_single_phased_baseline(67587, plot=True)  # 0-2
obs.process_single_phased_baseline(69635, plot=True)  # 1-2

obs.process_single_phased_baseline(69647, plot=True)  # 1-14
obs.process_single_phased_baseline(69646, plot=True)  # 1-13
obs.process_single_phased_baseline(94223, plot=True)  # 1-2
# obs.process_phased_data(plot=True)
obs.plot_rfi_and_weights_map(log=False)
