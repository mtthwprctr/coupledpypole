import numpy as np
import itertools as it
import time

class Quasistatic:
    def __init__(self, lattice, element = None, neighbours = 40):
        self.lattice = lattice
        self.element = element
        self.neighbours = neighbours

    def gf(self, r, R, k_xy):
        rho = r - R[:, None]

        r_norm = np.linalg.norm(rho, axis=2)
        
        if np.linalg.norm(k_xy) == 0:
            b1, b2 = self.lattice.b1, self.lattice.b2
            k_xy = k_xy + b1

        bloch = np.exp(-1j * np.dot(k_xy, R.T))
        
        outer = np.einsum('...j,...k->...jk', rho, rho)
        identity = np.broadcast_to(np.identity(3), (r_norm.shape[0], r_norm.shape[1], 3, 3))

        g_a = (3 * outer/r_norm[:, :, None, None]**5 - identity/r_norm[:, :, None, None]**3)

        g = bloch[...,None,None,None] * g_a
        total = np.sum(g, axis=1)

        return total

    def interaction_matrix(self, k_xy):
        a1, a2 = self.lattice.get_lattice_vectors()
        uc = self.lattice.unit_cell()
        N = self.lattice.size

        n_range = np.arange(-self.neighbours, self.neighbours + 1)
        neighbouring_indices = it.product(n_range, n_range)

        if N == 1:
            R = np.array([a1 * i[0] + a2 * i[1] for i in neighbouring_indices])
            R_excl = R[np.where(np.linalg.norm(R, axis=1) != 0)]
    
            g_excl = self.gf([np.array([0, 0, 0])], R_excl, k_xy)
            H = g_excl.reshape((len(k_xy), 3, 3))

            H_xy = H[:, 0:2, 0:2]
            H_z = H[:, 2, 2]
        else:
            if N == 2:
                r = uc[1] - uc[0]
            else:
                combos = np.vstack(np.triu_indices(N, k=1)).T  # quicker way of creating (n, 2) combinations
                r = uc[combos[:, 0]] - uc[combos[:, 1]]
        
            # sum including origin
            R  = np.array([a1 * i[0] + a2 * i[1] for i in neighbouring_indices])
            g_incl = self.gf(r, R, k_xy)

            # sub excluding origin
            R_excl = R[np.where(np.linalg.norm(R, axis=1) != 0)]    
            g_excl = self.gf([np.array([0, 0, 0])], R_excl, k_xy)

            # filling the interaction matrix            
            H = np.zeros((len(k_xy), N, N, 3, 3), dtype=np.complex128)
            if N == 2:
                g_incl = g_incl.reshape(len(k_xy), 3, 3)
                H[:, 0, 1] = g_incl
            else:
                H[:, combos[:, 0], combos[:, 1]] = g_incl

            H = H.transpose(0, 1, 3, 2, 4).reshape(len(k_xy), 3 * H.shape[1], 3 * H.shape[1])
            H = H + np.conj(H.transpose(0, 2, 1))
            
            g_excl = g_excl.reshape(len(k_xy), 3, 3)
            H += np.kron(np.eye(N, dtype=int), g_excl)

            N_range = np.arange(3 * N)
            z_components = N_range[2::3]
            xy_components = np.delete(N_range, z_components)

            ii = np.array(list(it.product(z_components, z_components)))
            H_z = H[:, ii[:, 0], ii[:, 1]]
            H_z = H_z.reshape((len(k_xy), N, N))

            ii = np.array(list(it.product(xy_components, xy_components)))
            H_xy = H[:, ii[:, 0], ii[:, 1]]
            H_xy = H_xy.reshape((len(k_xy), 2*N, 2*N))

        return H_xy, H_z

    def eigvals(self, size: int) -> tuple:
        start = time.time()
        k_xy = self.lattice.get_brillouin_zone(size)

        H_xy, H_z = self.interaction_matrix(k_xy)
        xy_eigvals =  np.empty((len(k_xy), H_xy.shape[1]))
        z_eigvals = np.empty((len(k_xy), H_z.shape[1]))
        
        for i in np.arange(size):
            xy_eigvals[i] = np.linalg.eigvalsh(H_xy[i])
            z_eigvals[i] = np.linalg.eigvalsh(H_z[i])

        print(f't = {time.time() - start:.2f}s')

        return xy_eigvals, z_eigvals

    def bands(self, size: int) -> tuple:
        xy_eigvals, z_eigvals = self.eigvals(size)

        xy_freqs = self.element.eig_to_freq(xy_eigvals, 'xy') 
        z_freqs = self.element.eig_to_freq(z_eigvals, 'z')

        return xy_freqs, z_freqs
