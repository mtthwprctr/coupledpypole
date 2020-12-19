# import matplotlib as mpl

# nice_fonts = {
#         "text.usetex": False,
#         "font.family": "serif",
#         "axes.labelsize": 12,
#         "font.size": 12,
#         "legend.fontsize": 12,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "axes.titlesize": 12,
# }

# mpl.rcParams.update(nice_fonts)

import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
from utils import flat_hex_corner_edges, square_edges

import lattices
   
def plot_lattice(lattice, r = None, uc_only = False): 
    fig, ax = plt.subplots(1, figsize = (4, 4), dpi = 200)

    a1, a2 = lattice.get_lattice_vectors()
    if r == None:
        r = lattice.R0 / 3
    uc = lattice.unit_cell()

    ax.plot([0, a1[0]], [0, a1[1]], zorder = 4, c='r')
    ax.plot([0, a2[0]], [0, a2[1]], zorder = 4, c='b')

    for n in np.arange(-2, 3):
        for m in np.arange(-2, 3):
            centre = n * a1 + m * a2

            for position in uc:
                fc = (0.7, 0.7, 0.7, 1)
                pos = centre + position
                circle=plt.Circle((pos[0], pos[1]), radius=r, fc=fc, ec=(0,0,0,1), 
                        rasterized=True, zorder=3, lw=0.5)
                ax.add_artist(circle)

            if isinstance(lattice, lattices.Square):
                uc_edge = np.array([square_edges(centre, lattice.a0 / np.sqrt(2), i) for i in np.arange(5)])

            elif isinstance(lattice, lattices.Triangular):
                uc_edge = np.array([flat_hex_corner_edges(centre, lattice.a0 / sqrt(3), i) for i in np.arange(7)])
            
            uc_edge_x, uc_edge_y = uc_edge[:, 0], uc_edge[:, 1] 
            ax.plot(uc_edge_x, uc_edge_y, lw=0.2, c='grey', zorder=3, ls='-')

            if n == 0 and m == 0:
                ax.fill(uc_edge_x, uc_edge_y, lw=0.2, fc=(0, 1, 0, 0.2), zorder=1, ls='-')

    ax.axis('equal')
    plt.show()
