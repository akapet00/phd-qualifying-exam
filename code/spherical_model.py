import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from scipy import spatial

from utils import update_matplotlib_rc_parameters
from utils import sph2cart
from utils import set_axes_equal


def main():
    # coordinates
    r = 50 / 1000  # in m
    theta = np.linspace(0, 2 * np.pi, 17)
    phi = np.linspace(0, np.pi, 9)
    Theta, Phi = np.meshgrid(theta, phi)
    x, y, z = sph2cart(r, Theta.ravel(), Phi.ravel())
    hull = spatial.ConvexHull(np.c_[x, y, z])

    # visualize
    update_matplotlib_rc_parameters()
    cs = sns.color_palette('rocket', 2)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter(x, y, z, ec='k', color=cs[0])
    hull_triangle_coords = hull.points[hull.simplices]
    triangles = Poly3DCollection(hull_triangle_coords,
                                color=cs[1], ec='k', lw=0.1, alpha=0.25)
    ax.add_collection3d(triangles)
    ax.set_box_aspect([1, 1, 1])
    ax = set_axes_equal(ax)
    ax.set(xlabel='$x$ [mm]', ylabel='$y$ [mm]', zlabel='$z$ [mm]',
        xticks=[-r, 0, r],
        yticks=[-r, 0, r],
        zticks=[-r, 0, r],
        xticklabels=[-int(r * 1000), 0, int(r * 1000)],
        yticklabels=[-int(r * 1000), 0, int(r * 1000)],
        zticklabels=[-int(r * 1000), 0, int(r * 1000)],
        xlim=[-r * 1.2, r * 1.2],
        ylim=[-r * 1.2, r * 1.2],
        zlim=[-r * 1.2, r * 1.2],
        )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 7
    ax.zaxis.labelpad = 7
    ax.view_init(25, -55)
    # fig.savefig(os.path.join(
    #     os.pardir, 'artwork', 'spherical_model.pdf'
    #     ), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()