import numpy as np

import itertools as it

from multiprocessing import Pool

import lattices

class Quasistatic:
    def __init__(self, lattice, element = None, neighbours = 4):
        self.lattice = lattice
        self.element = element
        self.neighbours = neighbours

    def gf(self, r, R, q):
        rho = r - R[:, None]

        r_norm = np.linalg.norm(rho, axis=2)
        
        if np.linalg.norm(q) == 0:
            b1, b2 = self.lattice.b1, self.lattice.b2
            q = q + b1

        bloch = np.exp(-1j * np.dot(q, R.T))
        
        outer = np.einsum('...j,...k->...jk', rho, rho)
        identity = np.broadcast_to(np.identity(3), (r_norm.shape[0], r_norm.shape[1], 3, 3))

        g = bloch[:, None, None, None] * (3 * outer/r_norm[:, :, None, None]**5 - identity/r_norm[:, :, None, None]**3)

        return np.sum(g, axis=0)

    def interaction_matrix(self, q):
        a1, a2 = self.lattice.get_lattice_vectors()
        uc = self.lattice.unit_cell()
        N = self.lattice.size

        n_range = np.arange(-self.neighbours, self.neighbours + 1)
        neighbouring_indices = it.product(n_range, n_range)

        if N == 1:
            R = np.array([a1 * i[0] + a2 * i[1] for i in neighbouring_indices])
            R_excl = R[np.where(np.linalg.norm(R, axis=1) != 0)]
    
            g_excl = self.gf([np.array([0, 0, 0])], R_excl, q)
            H = g_excl.reshape((3, 3))

            H_xy = H[0:2, 0:2]
            H_z = H[2][2]
        else:
            if N == 2:
                r = uc[1] - uc[0]
            else:
                combos = np.vstack(np.triu_indices(N, k=1)).T  # quicker way of creating (n, 2) combinations
                r = uc[combos[:, 0]] - uc[combos[:, 1]]
        
            # sum including origin
            R  = np.array([a1 * i[0] + a2 * i[1] for i in neighbouring_indices])
            g_incl = self.gf(r, R, q)

            # sub excluding origin
            R_excl = R[np.where(np.linalg.norm(R, axis=1) != 0)]    
            g_excl = self.gf([np.array([0, 0, 0])], R_excl, q)

            # filling the interaction matrix            
            H = np.zeros((N, N, 3, 3), dtype=np.complex128)
            if N == 2:
                H[0, 1] = g_incl
            else:
                H[combos[:, 0], combos[:, 1]] = g_incl

            H = H.transpose(0, 2, 1, 3).reshape(3 * H.shape[0], 3 * H.shape[0])
            H = H + np.conj(H).T
            
            g_excl = g_excl.reshape(3, 3)
            H += np.kron(np.eye(N, dtype=int), g_excl)

            N_range = np.arange(3 * N)
            z_components = N_range[2::3]
            xy_components = np.delete(N_range, z_components)

            ii = np.array(list(it.product(z_components, z_components)))
            H_z = H[ii[:, 0], ii[:, 1]]
            H_z = H_z.reshape((N, N))

            ii = np.array(list(it.product(xy_components, xy_components)))
            H_xy = H[ii[:, 0], ii[:, 1]]
            H_xy = H_xy.reshape((2*N, 2*N))

        return H_xy, H_z

    def eigvals(self, size: int) -> tuple:
        bz = self.lattice.get_brillouin_zone(size)

        p = Pool()
        result = p.map(self._mp_eigvals, bz)
        p.close()

        result = np.array(result, dtype=object)

        in_eigs = np.array([i.real for i in result[:, 0]])
        out_eigs =  np.array([i.real for i in result[:, 1]])

        return in_eigs, out_eigs

    def bands(self, size: int) -> tuple:
        in_eigs, out_eigs = self.eigvals(size)

        in_freqs = self.element.eig_to_freq(in_eigs, 'xy') 
        out_freqs = self.element.eig_to_freq(out_eigs, 'z')

        return in_freqs, out_freqs

    def _mp_eigvals(self, q):
        uc = self.lattice.unit_cell()
        N = self.lattice.size

        H_xy, H_z = self.interaction_matrix(q)
        in_plane = np.linalg.eigvalsh(H_xy)

        if N == 1:
            out_plane = H_z
        else:
            out_plane = np.linalg.eigvalsh(H_z)

        return in_plane, out_plane
