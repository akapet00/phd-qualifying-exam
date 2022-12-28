import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
import seaborn as sns
from scipy import spatial

from utils import update_matplotlib_rc_parameters
from utils import sph2cart
from utils import sph_normals


def main():
	# averaging surface configuration
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

	# spherical averaging surface coordinates
	r = 0.05  # in m
	alpha = 2 * np.arcsin(edge_length/2/r)  # angle from secant
	theta = np.linspace(np.pi/2-alpha/2, np.pi/2+alpha/2, N)
	phi = np.linspace(np.pi-alpha/2, np.pi+alpha/2, N)
	Theta, Phi = np.meshgrid(theta, phi)
	y_sph, x_sph, z_sph = sph2cart(r, Theta.ravel(), Phi.ravel())
	y_sph -= y_sph.min() + y_sph.ptp() / 2
	ny_sph, nx_sph, nz_sph = sph_normals(r, Theta.ravel(), Phi.ravel())

	# visualize
	update_matplotlib_rc_parameters(is_3d=True)
	cs = sns.color_palette('rocket', 2)
	fig = plt.figure()
	ax = plt.axes(projection ='3d')
	plane = Rectangle(target_area_origin,
					  width=edge_length, height=edge_length,
					  ec=cs[0], fc=cs[1], alpha=0.25,
					  label='control evaluation plane')
	ax.add_patch(plane)
	pathpatch_2d_to_3d(plane, z=y, zdir='y')
	ax.scatter(x_sph, y_sph, z_sph, color=cs[0], depthshade=True,
		       label='integration grid')
	ax.quiver(x_sph, y_sph, z_sph,
		      nx_sph, ny_sph, nz_sph,
			  normalize=True, arrow_length_ratio=0.33, length=0.75/1000,
			  lw=1.25, color=cs[0], label='unit normal vector')
	ax.set_box_aspect([1, 1, 1])
	ax.set(xlabel='$x$ [mm]', ylabel='$y$ [mm]', zlabel='$z$ [mm]',
		   xticks=[x_pln.min(), 0.0, x_pln.max()],
		   yticks=[y_sph.min(), 0.0, y_sph.max()],
		   zticks=[z_pln.min(), 0.0, z_pln.max()],
		   xticklabels=[round(x_pln.min()*1000), 0, round(x_pln.max()*1000)],
		   yticklabels=[round(y_sph.min()*1000, 2), 0, round(y_sph.max()*1000, 2)],
		   zticklabels=[round(z_pln.min()*1000), 0, round(z_pln.max()*1000)],
		   xlim=[x_pln.min()*1.5, x_pln.max()*1.5],
		   ylim=[y_sph.min()-abs(y_sph.max()*0.5),
				 y_sph.max()+abs(y_sph.max()*0.5)],
		   zlim=[z_pln.min()*1.5, z_pln.max()*1.5]
		   )
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	ax.view_init(20, -70)
	fig.legend()
	fig.tight_layout()
	# fig.savefig(os.path.join(
	#     os.pardir, 'artwork', 'spherical_surface.pdf'
	#     ))
	plt.show()


if __name__ == '__main__':
	main()
