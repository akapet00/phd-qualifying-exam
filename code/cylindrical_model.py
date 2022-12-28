import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from scipy import spatial

from utils import update_matplotlib_rc_parameters
from utils import cyl2cart
from utils import set_axes_equal


def main():
    # coordinates
    r = 0.05  # in m
    theta = np.linspace(0, 2 * np.pi, 16)
    z = np.linspace(-r/2, r/2, 2)
    Theta, Z = np.meshgrid(theta, z)
    x, y, z = cyl2cart(r, Theta.ravel(), Z.ravel())

    # smooth spherical surface
    hull = spatial.ConvexHull(np.c_[x, y, z])

    # traingular mesh
    hull_triangle_coords = hull.points[hull.simplices]

    # visualize
    update_matplotlib_rc_parameters(is_3d=True)
    cs = sns.color_palette('rocket', 2)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    triangles = Poly3DCollection(hull_triangle_coords, lw=0.5,
                                 color=cs[1], alpha=0.25)
    ax.add_collection3d(triangles)
    ax.scatter(x, y, z, color=cs[0])
    ax.set_box_aspect([1, 1, 1])
    ax = set_axes_equal(ax)
    xyticks = [-r, 0, r]
    zticks = [z.min(), 0, z.max()]
    xyticklabels = [-round(r * 1000), 0, round(r * 1000)]
    zticklabels = [round(z.min() * 1000), 0, round(z.max() * 1000)]
    xylim = [-r * 1.2, r * 1.2]
    zlim = [-r/2 * 1.2, r/2 * 1.2]
    ax.set(xlabel='$x$ [mm]', ylabel='$y$ [mm]', zlabel='$z$ [mm]',
           xticks=xyticks, yticks=xyticks, zticks=zticks,
           xticklabels=xyticklabels,
           yticklabels=xyticklabels,
           zticklabels=zticklabels,
           xlim=xylim, ylim=xylim, zlim=zlim
           )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(20, -60)
    fig.tight_layout()
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'cylindrical_model.pdf'
    #     ))
    plt.show()


if __name__ == '__main__':
    main()
