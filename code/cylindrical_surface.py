import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
import seaborn as sns
from scipy import spatial

from utils import update_matplotlib_rc_parameters
from utils import cyl2cart
from utils import cyl_normals


def main():
    # averaging surface 
    edge_length = 0.02  # in m
    target_area_origin = (-edge_length/2, -edge_length/2)
    N = 11  # number of grid points

    # control evaluation plane coordinates
    x = np.linspace(-edge_length/2, edge_length/2, N)
    y = 0
    z = np.linspace(-edge_length/2, edge_length/2, N)
    Xt, Zt = np.meshgrid(x, z)
    x_pln = Xt.ravel()
    z_pln = Zt.ravel()

    # cylndrical averaging surface coordinates
    r = 0.05  # in m
    alpha = 2 * np.arcsin(edge_length/2/r)  # angle from secant
    theta = np.linspace(np.pi/2-alpha/2, np.pi/2+alpha/2, N)
    Theta, Z = np.meshgrid(-theta, z)
    x_cyl, y_cyl, z_cyl = cyl2cart(r, Theta.ravel(), Z.ravel())
    y_cyl -= y_cyl.min()+ y_cyl.ptp() / 2
    nx_cyl, ny_cyl, nz_cyl = cyl_normals(r, Theta.ravel(), Zt.ravel())

    # visualize
    update_matplotlib_rc_parameters(is_3d=True)
    cs = sns.color_palette('rocket', 2)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    plane = Rectangle(target_area_origin,
                      width=edge_length, height=edge_length,
                      ec=cs[0], fc=cs[1], alpha=0.375,
                      label='control evaluation plane')
    ax.add_patch(plane)
    pathpatch_2d_to_3d(plane, z=y, zdir='y')
    ax.scatter(x_cyl, y_cyl, z_cyl, color=cs[0], depthshade=True,
               label='integration grid')
    ax.quiver(x_cyl, y_cyl, z_cyl,
              nx_cyl, ny_cyl, nz_cyl,
              normalize=True, arrow_length_ratio=0.33, length=0.25/1000,
              lw=1.25, color=cs[0], label='unit normal vector')
    ax.set_box_aspect([1, 1, 1])
    ax.set(xlabel='$x$ [mm]', ylabel='$y$ [mm]', zlabel='$z$ [mm]',
           xticks=[x_pln.min(), 0.0, x_pln.max()],
           yticks=[y_cyl.min(), 0.0, y_cyl.max()],
           zticks=[z_pln.min(), 0.0, z_pln.max()],
           xticklabels=[round(x_pln.min()*1000), 0, round(x_pln.max()*1000)],
           yticklabels=[round(y_cyl.min()*1000, 1), 0, round(y_cyl.max()*1000, 1)],
           zticklabels=[round(z_pln.min()*1000), 0, round(z_pln.max()*1000)],
           xlim=[x_pln.min()*1.5, x_pln.max()*1.5],
           ylim=[y_cyl.min()-abs(y_cyl.max()*0.5),
                 y_cyl.max()+abs(y_cyl.max()*0.5)],
           zlim=[z_pln.min()*1.5, z_pln.max()*1.5]
           )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(20, -70)
    fig.legend()
    fig.tight_layout()
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'cylindrical_surface.pdf'
    #     ))
    plt.show()


if __name__ == '__main__':
    main()
