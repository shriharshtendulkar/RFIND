import pyuvdata
import numpy as np

filename = "/Users/shriharsh/Documents/Work/GMRT_Transients/04BCJ01_0142_170903.LTA_RRLL.RRLLFITS.fits"

data = pyuvdata.UVData()
data.read_uvfits(filename)

ant_lat, ant_lon, ant_alt = pyuvdata.utils.LatLonAlt_from_XYZ(
    data.antenna_positions + data.telescope_location, check_acceptability=True
)

telescope_location = data.telescope_location
telescope_location_lat_lon_alt = data.telescope_location_lat_lon_alt
telescope_location_lat_lon_alt_degrees = data.telescope_location_lat_lon_alt_degrees
telescope_name = data.telescope_name
antenna_diameters = data.antenna_diameters
antenna_names = data.antenna_names
antenna_numbers = data.antenna_numbers
antenna_positions = data.antenna_positions

np.savez(
    "GMRT_antenna_positions.npz",
    telescope_location=telescope_location,
    telescope_location_lat_lon_alt=telescope_location_lat_lon_alt,
    telescope_location_lat_lon_alt_degrees=telescope_location_lat_lon_alt_degrees,
    telescope_name=telescope_name,
    antenna_diameters=antenna_diameters,
    antenna_names=antenna_names,
    antenna_numbers=antenna_numbers,
    antenna_positions=antenna_positions,
)
