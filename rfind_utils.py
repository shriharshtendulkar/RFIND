import numpy as np
from os import path
from pyuvdata import utils
from astropy.modeling import models, fitting
from astropy.stats import median_absolute_deviation as mad
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import minimize

SPEED_OF_LIGHT = 2.99792458e8  # meters per second


def robust_rescale_1d(data):
    # subtract median and divide by mad-derived std.

    data -= np.median(data)
    data /= 1.4826 * mad(data)
    return data


def scale_visibility_data(
    vis,
    freqs,
    times,
    plot=False,
    plotfileroot="./",
    poly_freq_deg=2,
    poly_freq_threshold=3,
    niter=4,
    poly_freq_fit_range=None,
    poly_freq_fit_to_log=False,
):
    """
    Scales and prepares visibility data for FFT and dynamic spectra.
    """

    assert vis.shape == (len(times), len(freqs), 2)

    dyn = np.sum(np.abs(vis), axis=2)

    # plot the raw dynamic spectrum
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=True)
        plt.pcolor(freqs, times, np.log10(dyn), cmap="viridis")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Integrations")
        plt.title("Raw Spectrum")
        plt.savefig(path.join(plotfileroot, "raw_spectrum.png"))
        plt.close(fig)

    # remove the bad frequency channels.

    mean = np.mean(dyn, axis=0)
    std = np.std(dyn, axis=0)

    med_mean = np.median(mean)
    mad_mean = mad(mean)

    # set the threshold to 4 and make bad channels zero.
    threshold = 4
    filt = np.where((mean > med_mean + threshold * mad_mean))

    med_vis = np.median(vis[:, filt, :])
    vis[:, filt, :] = med_vis

    # plot the data with mask.
    dyn = np.sum(np.abs(vis), axis=2)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=True)
        plt.pcolor(freqs, times, np.log10(dyn), cmap="viridis")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Integrations")
        plt.title("Masked Spectrum")
        plt.savefig(path.join(plotfileroot, "masked_spectrum.png"))
        plt.close(fig)

    # fit each time integration with low order polynomial and divide.
    x = np.arange(len(freqs))
    for i in range(len(times)):
        fitted_curve, mask = fit_with_outlier_rejection(
            dyn[i, :],
            deg=poly_freq_deg,
            sigma=poly_freq_threshold,
            niter=niter,
            fit_range=poly_freq_fit_range,
            fit_to_log=poly_freq_fit_to_log,
        )
        if poly_freq_fit_to_log:
            linear_fitted_curve = np.power(10, fitted_curve(x))
            dyn[i, :] /= linear_fitted_curve
            # divide the power out for both polynomials.
            vis[i, :, 0] /= linear_fitted_curve
            vis[i, :, 1] /= linear_fitted_curve
        else:
            dyn[i, :] /= fitted_curve(x)
            # divide the power out for both polynomials.
            vis[i, :, 0] /= fitted_curve(x)
            vis[i, :, 1] /= fitted_curve(x)

        vis[i, mask, 0] = 0
        vis[i, mask, 1] = 0

    dyn = np.sum(np.abs(vis), axis=2)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=True)
        plt.pcolor(freqs, times, np.log10(dyn), cmap="viridis")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Integrations")
        plt.title("Masked Spectrum")
        plt.savefig(path.join(plotfileroot, "divided_masked_spectrum.png"))
        plt.close(fig)

    return vis


def fit_with_outlier_rejection(
    data, deg, niter=0, sigma=4, fit_to_log=True, fit_range=None
):
    """
    Implements a wrapper around astropy.modeling that will fit with iterative outlier rejection.

    """

    x = np.arange(len(data))

    if fit_range is not None:
        _min, _max = fit_range

    else:
        _min = 0
        _max = len(x)

    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(
        fit, sigma_clip, niter=niter, sigma=sigma
    )

    poly_init = models.Polynomial1D(degree=deg)

    if fit_to_log:
        fitted_curve, mask = or_fit(poly_init, x[_min:_max], np.log10(data[_min:_max]))
    else:
        fitted_curve, mask = or_fit(poly_init, x[_min:_max], data[_min:_max])

    return fitted_curve, mask


def mask_bright_sources(
    delay_spectrum, threshold=50, max_width=2, interp_range=20, expand=2
):

    delay_spectrum_1d = np.sum(delay_spectrum, axis=0)

    delay_spectrum_1d = robust_rescale_1d(delay_spectrum_1d)

    peaks, info = find_peaks(
        delay_spectrum_1d, width=[1, max_width], prominence=threshold
    )

    print(peaks, info)

    num_peaks = len(peaks)

    for i in range(num_peaks):
        left = int(np.floor(info["left_ips"][i]) - expand)
        right = int(np.ceil(info["right_ips"][i]) + expand)

        # interpolation range
        intp_left = max(0, left - interp_range)
        intp_right = min(delay_spectrum_1d.shape[0], right + interp_range)

        # mask out the bad data between left to right.
        x = np.hstack([np.arange(intp_left, left), np.arange(right, intp_right)])
        y = np.hstack(
            [delay_spectrum[:, intp_left:left], delay_spectrum[:, right:intp_right]]
        )

        print("x.shape = {} y.shape = {}".format(x.shape, y.shape))

        intp = interp1d(x, y, kind="quadratic", axis=1)

        delay_spectrum[:, left:right] = intp(np.arange(left, right))

    return delay_spectrum


def hyperbola(delay_s, p0, p1, range=10000, pts=200):
    """
    Plots a hyperbola with positions p0 = (x0, y0) and p1 = (x1, y1) as the foci.
    The delay is defined in seconds of light travel time such that positive --> closer to p0.
    """
    x0, y0 = p0  # position in meteres
    x1, y1 = p1  # position in meters

    angle = np.pi - np.arctan2((y1 - y0), (x1 - x0))

    c = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) / 2
    a = delay_s * SPEED_OF_LIGHT / 2  # convert from seconds to meters

    xy = cartesian_hyperbola(c, a, range, pts)

    # shift the focus from (c, 0) to origin.
    xy[0, :] -= c

    # rotate to p1 -> p0 angle.
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    xyp = np.matmul(rot, xy)

    # shift origin to p0
    xyp[0, :] += x0
    xyp[1, :] += y0

    return xyp


def cartesian_hyperbola(c, a, range, pts):
    """
    Returns a hyperbola around the focus at (c,0). The other focus is at (-c,0)
    """
    y = np.linspace(-range, range, pts, endpoint=True)
    b = np.sqrt(c**2 - a**2)
    x = a * np.sqrt(1 + y**2 / b**2)
    return np.vstack([x, y])


def simple_baseline_to_ant_pair(bl):
    ant1 = bl % 256 - 1
    ant2 = bl - (ant1 + 1) // 256 - 1
    return ant1, ant2

def solve_3d_hyperbola(ant_pos, delays, bounds):
    """
    Solves the intersection of 3 hyperbolas in 3d space.

    Definitions:
    ant_pos(6,3): antenna positions P1,.. P6 where P = (x, y, z) in cartesian coordinates.
    delays (3)
    We solve for P where
    (x1-x)**2+(y1-y)**2+(z1-z)**2-(x2-x)**2-(y2-y)**2-(z2-z)**2 = delays[1]**2
    (x3-x)**2+(y3-y)**2+(z3-z)**2-(x4-x)**2-(y4-y)**2-(z4-z)**2 = delays[2]**2
    (x5-x)**2+(y5-y)**2+(z5-z)**2-(x6-x)**2-(y6-y)**2-(z6-z)**2 = delays[3]**2
    """
    d1, d2, d3 = delays
    x1, y1, z1 = ant_pos[0,:]
    x2, y2, z2 = ant_pos[1,:]
    x3, y3, z3 = ant_pos[2,:]
    x4, y4, z4 = ant_pos[3,:]
    x5, y5, z5 = ant_pos[4,:]
    x6, y6, z6 = ant_pos[5,:]

    func = lambda p: (((x1-p[0])**2+(y1-p[1])**2+(z1-p[2])**2-(x2-p[0])**2-(y2-p[1])**2-(z2-p[2])**2 - d1**2)**2 
                    + ((x3-p[0])**2+(y3-p[1])**2+(z3-p[2])**2-(x4-p[0])**2-(y4-p[1])**2-(z4-p[2])**2 - d2**2)**2 
                    + ((x5-p[0])**2+(y5-p[1])**2+(z5-p[2])**2-(x6-p[0])**2-(y6-p[1])**2-(z6-p[2])**2 - d3**2)**2)

    x0 = np.mean(ant_pos, axis=0)

    p = minimize(func, x0, bounds=bounds)

    return p

def calculate_likely_intersections(positions, delays, bounds, limit):
    """
    Chooses all combinations of three unique baselines, chooses all combinations of three unique delays from each of them.
    Calculates a possible intersection. Saves the location of the intersection and the function value.
    """

    unique_bls = np.unique(delays['baseline'])

    start=True
    dtype=np.dtype([('x', np.float64),('y', np.float64),('z', np.float64),('w', np.float64)])
    ret_array = np.zeros((1,), dtype=dtype)
    for i in range(len(unique_bls)-2):
        for j in range(i+1, len(unique_bls)-1):
            for k in range(j+1, len(unique_bls)):
                print("i, j, k: ({}, {}, {}) from {}".format(i,j,k, len(unique_bls)))
                print("bls: {}, {}, {}".format(unique_bls[i], unique_bls[j], unique_bls[k]))
                filt1 = np.where(delays['baseline']==unique_bls[i])
                filt2 = np.where(delays['baseline']==unique_bls[j])
                filt3 = np.where(delays['baseline']==unique_bls[k])

                if start:
                    ret_array = calculate_likely_intersections_baselines(positions, delays[filt1], delays[filt2], delays[filt3], bounds=bounds, limit=limit)
                    start=False
                else:
                    temp_array = calculate_likely_intersections_baselines(positions, delays[filt1], delays[filt2], delays[filt3], bounds=bounds, limit=limit)
                    ret_array = np.vstack([ret_array, temp_array])
    return ret_array

def calculate_likely_intersections_baselines(positions, delays1, delays2, delays3, bounds, limit):
    """
    Chooses every combination of a delay from each baseline and then calculates a possible intersection.
    """

    num_delays1 = len(delays1)
    num_delays2 = len(delays2)
    num_delays3 = len(delays3)

    bl1 = delays1['baseline'][0]
    bl2 = delays2['baseline'][0]
    bl3 = delays3['baseline'][0]

    ant1, ant2 = utils.baseline_to_antnums(bl1, 30)
    ant3, ant4 = utils.baseline_to_antnums(bl2, 30)
    ant5, ant6 = utils.baseline_to_antnums(bl3, 30)

    ant_pos = np.zeros((6,3))

    ant_pos[0,:] = positions[ant1,:]
    ant_pos[1,:] = positions[ant2,:]
    ant_pos[2,:] = positions[ant3,:]
    ant_pos[3,:] = positions[ant4,:]
    ant_pos[4,:] = positions[ant5,:]
    ant_pos[5,:] = positions[ant6,:]

    start=True
    dtype=np.dtype([('x', np.float64),('y', np.float64),('z', np.float64),('w', np.float64)])
    ret_array = np.zeros((1,), dtype=dtype)
    for i in range(num_delays1):
        for j in range(num_delays2):
            for k in range(num_delays3):
                print("Delays: {}/{} {}/{} {}/{}".format(i, num_delays1, j, num_delays2, k, num_delays3))
                delay_dist1 = delays1[i]['delay']*SPEED_OF_LIGHT
                delay_dist2 = delays2[j]['delay']*SPEED_OF_LIGHT
                delay_dist3 = delays3[k]['delay']*SPEED_OF_LIGHT

                sol = solve_3d_hyperbola(ant_pos, [delay_dist1, delay_dist2, delay_dist3], bounds)
                if sol['fun'] < limit and sol['success']:
                    if start:
                        ret_array['x'][0] = sol['x'][0]
                        ret_array['y'][0] = sol['x'][1]
                        ret_array['z'][0] = sol['x'][2]
                        ret_array['w'][0] = sol['fun']
                        start=False

                    else:
                        temp_array = np.zeros((1,), dtype=dtype)
                        temp_array['x'][0] = sol['x'][0]
                        temp_array['y'][0] = sol['x'][1]
                        temp_array['z'][0] = sol['x'][2]
                        temp_array['w'][0] = sol['fun']
                        ret_array = np.vstack((ret_array, temp_array))

    return ret_array