import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import update_matplotlib_rc_parameters
from utils import incident_power_density


def main():
    f = np.linspace(6, 300, 1001) * 1e9
    ipd_icnirp_occupational = np.vectorize(
        incident_power_density
        )(f, 'icnirp', 'occupational')
    ipd_icnirp_gp = np.vectorize(
        incident_power_density
        )(f, 'icnirp', 'general public')
    ipd_ieee_occupational = np.vectorize(
        incident_power_density)(f, 'ieee', 'occupational')
    ipd_ieee_gp = np.vectorize(
        incident_power_density
        )(f, 'ieee', 'general public')
    ipd_wb_occupational = np.full(f.shape, 50)
    ipd_wb_gp = np.full(f.shape, 10)

    # visualize
    update_matplotlib_rc_parameters()
    cs = sns.color_palette('rocket', 4)
    fig, ax = plt.subplots()
    ax.plot(f, ipd_icnirp_occupational, '-', c=cs[0], lw=3,
            label='ICNIRP, occupational')
    ax.plot(f, ipd_ieee_occupational, '--', c=cs[2], lw=3,
            label='IEEE, occupational')
    ax.plot(f, ipd_icnirp_gp, '-.', c=cs[1], lw=3,
            label='ICNIRP, general public')
    ax.plot(f, ipd_ieee_gp, ':', c=cs[3], lw=3,
            label='IEEE, general public')
    ax.hlines(ipd_wb_occupational, xmin=f.min(), xmax=f.max(),
            lw=1.25, color='.15')
    ax.text(9e9, ipd_wb_occupational[0] * 1.1,
            s='whole body exposure limit, occupational')
    ax.hlines(ipd_wb_gp, xmin=f.min(), xmax=f.max(),
            lw=1.25, color='.15')
    ax.text(9e9, ipd_wb_gp[0] * 1.1,
            s='whole body exposure limit, general public')
    ax.set(
        xscale='log',
        yscale='log',
        xlabel='frequency [GHz]',
        ylabel='incident power density [W/m$^2$]',
        xticks=np.array([6, 30, 301]) * 1e9,
        xticklabels=[6, 30, 300],
        yticks=[10, 20, 40, 50, 100, 200],
        yticklabels=[10, 20, 40, 50, 100, 200],
        ylim=[9.1, 800]
        )
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'reference_levels.pdf'
    #     ), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
