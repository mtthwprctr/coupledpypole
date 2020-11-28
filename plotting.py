import numpy as np
from scipy import constants
from matplotlib import pyplot as plt

import lattices
import solver
import particles

EV = constants.e / constants.h

def plot_bands(system, grid = 64):
    in_eigs, out_eigs = system.eigvals(grid)

    fig, ax = plt.subplots(1, figsize = (4, 3), dpi=150)

    in_freqs = system.element.eig_to_freq(in_eigs, 'xy') / EV
    out_freqs = system.element.eig_to_freq(out_eigs, 'z') / EV

    ax.plot(in_freqs, c='r',)
    ax.plot(out_freqs, c='b',)

    labels = l.get_bz_labels(grid)
    ax.set_xticks(list(labels.keys()))
    ax.set_xticklabels(labels.values())

    ax.set_xlim(0, grid-1)

    ax.set_ylabel('Eigenvalue')

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    l = lattices.Honeycomb(30E-9 * np.sqrt(3))
    silver = particles.Metal()
    np = particles.Particle({'radius':10E-9, 'height':10E-9, 'material':silver})
    print(l)
    print(np)
    # qs = solver.Quasistatic(l, np, neighbours = 10) 
    # plot_bands(qs)

   