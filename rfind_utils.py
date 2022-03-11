from turtle import delay
import numpy as np
from os import path
from astropy.modeling import models, fitting
from astropy.stats import median_absolute_deviation as mad
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

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

    vis[:, filt, :] = 0

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
    b = np.sqrt(c ** 2 - a ** 2)
    x = a * np.sqrt(1 + y ** 2 / b ** 2)
    return np.vstack([x, y])
