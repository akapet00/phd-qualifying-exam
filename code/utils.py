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