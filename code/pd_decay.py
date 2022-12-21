import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import epsilon_0 as eps_0, mu_0
import seaborn as sns

from utils import incident_power_density
from utils import update_matplotlib_rc_parameters
from utils import load_tissue_properties
from utils import reflection_coefficient


def take_sample(sample_size,
                ipd,
                transmission_coefficient,
                depth,
                penetration_depth):
    uniform_scaler = np.random.rand(sample_size, ipd.size)
    S_surface = uniform_scaler * ipd * transmission_coefficient
    return S_surface * np.exp(-2 * depth / penetration_depth)
 

def run_sampling(number_of_runs, **kwargs):
    S_avg_list = []
    S_std_list = []
    for _ in range(number_of_runs):
        S_sample = take_sample(**kwargs)
        S_avg_list.append(np.mean(S_sample, axis=0))
        S_std_list.append(np.std(S_sample, ddof=1, axis=0))
    return np.asarray(S_avg_list), np.asarray(S_std_list)


def main():
    f = np.array([10, 30, 60, 100]) * 1e9  # in Hz
    sigma, eps_r, _, penetration_depth = np.vectorize(
        load_tissue_properties
        )('skin_dry', f)
    eps_i = sigma / (2 * np.pi * f * eps_0)
    eps = eps_r - 1j * eps_i
    gamma = reflection_coefficient(eps)  # reflection coefficient
    transmission_coefficient = 1 - gamma ** 2  # power transmission  coefficient
    ipd_icnirp_gp = np.vectorize(
        incident_power_density)(f, 'icnirp', 'general public')
    depths = [0, 1 / 1000]  # 0 and 1 mm in meters


    # visualize
    update_matplotlib_rc_parameters()
    cs = sns.color_palette('rocket', len(depths))
    fig, ax = plt.subplots()
    for i, depth in enumerate(depths):
        S_avg, S_std = run_sampling(number_of_runs=1000,
                                    sample_size=1000,
                                    ipd=ipd_icnirp_gp,
                                    transmission_coefficient=transmission_coefficient,
                                    depth=depth,
                                    penetration_depth=penetration_depth)
        S_avg_avg = np.mean(S_avg, axis=0)
        S_err_avg = np.mean(S_std, axis=0)

        if depth == 0:
            label = 'at the skin surface'
            ls = '-'
        else:
            label = f'{depth * 1000:.0f} mm deep in the skin'
            ls = '--'
        ax.errorbar(f, S_avg_avg, yerr=S_err_avg, c=cs[i], marker='o', ls=ls,
                    elinewidth=1.25, capsize=5, label=label)
        ax.legend(loc='upper right', frameon=False)
    ax.set(
        xlabel='frequency [GHz]',
        ylabel='power density [W/m$^2$]',
        xticks=f,
        xticklabels=(f / 1e9).astype(int),
        yticks=[0, 6, 12, 18],
        yticklabels=[0, 6, 12, 18],
        )
    fig.tight_layout()
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'pd_decay.pdf'
    #     ), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
