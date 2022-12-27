import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SUPPORTED_TISSUES = [
    'air',
    'blood',
    'blood_vessel',
    'body_fluid',
    'bone_cancellous',
    'bone_cortical',
    'bone_marrow',
    'brain_grey_matter',
    'brain_white_matter',
    'cerebellum',
    'cerebro_spinal_fluid',
    'dura', 'fat',
    'muscle',
    'skin_dry',
    'skin_wet'
    ]
SUPPORTED_POLARIZATIONS = ['parallel', 'perpendicular']
SUPPORTED_LIMITS = ['icnirp', 'ieee']
SUPPORTED_EXPOSURE = ['occupational', 'general public']


def load_tissue_properties(tissue, frequency):
    """Return conductivity, relative permitivity, loss tangent and
    penetration depth of a given tissue based on a given frequency.
    
    Parameters
    ----------
    tissue : str
        type of human tissue
    frequency : float
        radiation frequency
        
    Returns
    -------
    tuple
        Values for conductivity, relative permitivity, loss tangent and
        penetration depth of a given tissue at corresponding frequency.
    """
    tissue = tissue.lower()
    if tissue not in SUPPORTED_TISSUES:
        raise ValueError(
            f'Unsupported tissue. Choose from: {SUPPORTED_TISSUES}.'
            )
    if 1e9 > frequency > 100e9:
        raise ValueError('Invalid frequency. Choose in [1, 100] GHz range.')
    tissue_diel_properties_path = os.path.join(
        os.pardir, 'data', 'tissue_properties.csv'
        )
    with open(tissue_diel_properties_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if str(row[0]) == tissue and float(row[1]) == frequency:
                conductivity = float(row[2])
                relative_permitivity = float(row[3])
                loss_tangent = float(row[4])
                penetration_depth = float(row[5])
    return (conductivity, relative_permitivity, loss_tangent, penetration_depth)


def reflection_coefficient(eps_r, theta_i=0, polarization='parallel'):
    """Return reflection coefficient for oblique plane wave incidence.
    
    Parameters
    ----------
    eps_r : float
        Relative permittivity.
    theta_i : float
        Angle of incidence in °.
    polarization : str
        Either parallel or perpendicular/normal polarization.
    
    Returns
    -------
    float
        Reflection coefficient.
    """
    polarization = polarization.lower()
    if polarization not in SUPPORTED_POLARIZATIONS:
        raise ValueError(
            f'Unsupported tissue. Choose from: {SUPPORTED_POLARIZATIONS}.'
            )
    scaler = np.sqrt(eps_r - np.sin(theta_i) ** 2)
    if polarization == 'parallel':
        return np.abs(
            (-eps_r * np.cos(theta_i) + scaler)
            / (eps_r * np.cos(theta_i) + scaler)
        )
    return np.abs(
        (np.cos(theta_i) - scaler)
        / (np.cos(theta_i) + scaler)
    )


def incident_power_density(frequency, limits, exposure):
    """Return inciden power density value at a given frequency.
    
    Parameters
    ----------
    frequency : float
        Frequency at the [6, 300] GHz range.
    limits : str
        Supported exposure limits are: 'icnirp' and 'ieee'.
    exposure: str
        Supported exposure scenarios are 'occupational' and
        'general public'.

    Returns
    -------
    float
        Incident power density, either peak or spatially averaged.
    """
    if (frequency < 6e9) | (frequency > 300e9):
        raise ValueError('Frequency out of the supported range.')
    limits = limits.lower()
    assert limits in SUPPORTED_LIMITS, 'Limits not supported.'
    exposure = exposure.lower()
    assert exposure in SUPPORTED_EXPOSURE, 'Exposure not supported.'
    exposure = exposure.lower()
    if frequency == 6e9:
        if exposure == 'occupational':
            return 200
        return 40
    elif 6e9 < frequency < 300e9:
        if exposure == 'occupational':
            if limits == 'icnirp':
                return 275 * (frequency / 1e9) ** (-0.177)
            return 274.8 * (frequency / 1e9) ** (-0.177)
        return 55 * (frequency / 1e9) ** (-0.177)
    if exposure == 'occupational':
        return 100
    return 20


def cart2sph(x, y, z):
    """Return spherical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """Return Cartesian given Spherical coordinates."""
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def sph_normals(r, theta, phi):
    """Return unit vector field components normal to spherical
    surface."""
    nx = r ** 2 * np.cos(phi) * np.sin(theta) ** 2 
    ny = r ** 2 * np.sin(phi) * np.sin(theta) ** 2
    nz = r ** 2 * np.cos(theta) * np.sin(theta)
    return nx, ny, nz


def cyl2cart(r, theta, z):
    """Return Cartesian given Cylndrical coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def cyl_normals(r, theta, z):
    """Return unit vector field components normal to cylndrical
    surface."""
    nx = np.cos(theta)
    ny = np.sin(theta)
    nz = np.zeros_like(z)
    return nx, ny, nz


def set_axes_equal(ax):
    """Return 3-D axes with equal scale.
    Note: This function is implemented as in:
    https://stackoverflow.com/a/31364297/15005103 because there is no
    support setting that would enable `ax.axis('equal')` in 3-D.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D axes subplot with scale settings set to `auto`.
    
    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Axes as if the scale settings were defined as `equal`.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # bounding box is a sphere in the sense of the infinity norm
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax


def update_matplotlib_rc_parameters():
    """Run and configure visualization parameters."""
    sns.set(style='ticks', font='serif', font_scale=1.25)
    plt.rcParams.update({
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif'
        })
