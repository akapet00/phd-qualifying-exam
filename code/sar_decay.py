import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0 as eps_0, mu_0
import seaborn as sns

from utils import update_matplotlib_rc_parameters
from utils import load_tissue_properties
from utils import reflection_coefficient


def main():
    f = np.array([6, 30]) * 1e9  # frequency in Hz
    _, eps_r, _, penetration_depth = np.vectorize(  # power penetration depth
        load_tissue_properties
        )('skin_dry', f)
    L = penetration_depth * 1000  # penetration depth in mm
    gamma = reflection_coefficient(eps_r)
    T_tr = 1 - gamma ** 2  # power transmission coefficient
    ipd = 1000  # incident power density in W/m2
    rho_avg = 1109  # average skin density in kg/m3
    rho_std = 14  # std of skin density in kg/m3
    sar_sur = ipd * T_tr / (rho_avg * L)
    sar_dec = sar_sur / np.exp(1)
    z = np.linspace(0, 5, 101)
    sar_6 = sar_sur[0] * np.exp(- 2 * z / L[0])
    sar_30 = sar_sur[1] * np.exp(- 2 * z / L[1])
    idx_sar_6_dec = np.where(np.isclose(sar_dec[0], sar_6, 1e-2))[0][0]
    idx_sar_30_dec = np.where(np.isclose(sar_dec[1], sar_30, 1e-1))[0][0]

    # visualize
    update_matplotlib_rc_parameters()
    cs = sns.color_palette('rocket', 2)
    fig, ax = plt.subplots()
    ax.plot(z, sar_6, '-', c=cs[0], lw=3, label='$6$ GHz')
    ax.plot(z[idx_sar_6_dec], sar_dec[0], 'o', c=cs[0], label='value post $1/e$ decay at $6$ GHz')
    ax.plot(z, sar_30, '--', c=cs[1], lw=3, label='$30$ GHz')
    ax.plot(z[idx_sar_30_dec], sar_dec[1], '^', c=cs[1], label='value post $1/e$ decay at $30$ GHz')
    ylim = ax.axes.get_ylim()
    ax.vlines(z[idx_sar_6_dec], ylim[0], sar_dec[0],
            lw=1.5, color='k', zorder=0)
    ax.vlines(z[idx_sar_30_dec], ylim[0], sar_dec[1],
            lw=1.5, color='k', zorder=0)
    ax.set(
        #xscale='log',
        xlabel='depth of a 10-g cube of skin [mm]',
        ylabel='specific absorption rate [W/kg]',
        xticks=[0, z[idx_sar_30_dec], z[idx_sar_6_dec], 5],
        xticklabels=[0, z[idx_sar_30_dec].round(2), z[idx_sar_6_dec].round(2), 5],
        yticks=[0, sar_6.max(), sar_30.max()],
        yticklabels=[0, sar_6.max().round(2), sar_30.max().round(2)],
        ylim=ylim
        )
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'sar_decay.pdf'
    #     ), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()