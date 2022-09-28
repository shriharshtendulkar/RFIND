from rfind import Observation

filename = "/DATA/shriharsh/GMRT_RFI/Jitendra_reduction/Empty23.ms"
outfileroot = "/DATA/shriharsh/GMRT_RFI/Jitendra_reduction/empty23_analysis"
GMRT_antenna_position_file = "/home/shriharsh/Code/RFIND/GMRT_antenna_positions.npz"

PLOT=False
save=False

obs = Observation(
    filename,
    outfileroot=outfileroot,
    num_grid_pts=700,
    GMRT_antenna_position_file=GMRT_antenna_position_file,
)
obs.logger.setLevel("INFO")
#obs.make_grid(lat_range=(19.050, 19.125), lon_range=(74.000, 74.100), plot=False)
# obs.make_grid()
#obs.make_delay_maps(plot=False)
#obs.make_intensity_map()

# obs.observation.unphase_to_drift()

for i in [3, 22, 30, 31]:
    obs.remove_antenna(i)


# choose three antennas. 14, 19, 24 (nearest arm antennas)

obs.process_single_phased_baseline(96276, plot=PLOT, save_1d_spec=save)  # 14-19
obs.process_single_phased_baseline(106521, plot=PLOT, save_1d_spec=save)  # 19-24
obs.process_single_phased_baseline(96281, plot=PLOT, save_1d_spec=save)  # 14-24

obs.process_single_phased_baseline(67586, plot=PLOT, save_1d_spec=save)  # 0-1
obs.process_single_phased_baseline(67587, plot=PLOT, save_1d_spec=save)  # 0-2
obs.process_single_phased_baseline(69635, plot=PLOT, save_1d_spec=save)  # 1-2

obs.process_single_phased_baseline(69647, plot=PLOT, save_1d_spec=save)  # 1-14
obs.process_single_phased_baseline(69646, plot=PLOT, save_1d_spec=save)  # 1-13
obs.process_single_phased_baseline(94223, plot=PLOT, save_1d_spec=save)  # 13-14

# random baselines
#obs.process_single_phased_baseline(67610, plot=PLOT, save_1d_spec=save)  # 0-25
#obs.process_single_phased_baseline(94234, plot=PLOT, save_1d_spec=save)  # 13-25
#obs.process_single_phased_baseline(94229, plot=PLOT, save_1d_spec=save)  # 13-20
#obs.process_single_phased_baseline(98329, plot=PLOT, save_1d_spec=save)  # 15-24
#obs.process_single_phased_baseline(90128, plot=PLOT, save_1d_spec=save)


#obs.process_phased_data(plot=False)
#obs.plot_rfi_and_weights_map(log=False)
#obs.save_rfi_and_weights_map()
obs.save_output_records()
