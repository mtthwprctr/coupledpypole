#! /usr/bin/python3

import numpy as np
import scipy as sp
from numpy import sqrt, pi, exp
from scipy.special import erfc
from scipy import linalg, constants, integrate, optimize

import itertools as it
from multiprocessing import Pool

import time

EV = sp.constants.e/sp.constants.h

class Interaction:
    def __init__(self, lattice, element, eps_m = 1, element2 = None):
        self.lattice = lattice   
        self.element = element
        self.eps_m = eps_m  # epsilon_medium
        self.element2 = element2

    def spatial_part(self, f, k_xy, r0, excl=False, neighbours = 1):
        k0 = 2 * pi * f / sp.constants.c * sqrt(self.eps_m)

        a_1, a_2 = self.lattice.get_lattice_vectors()
        A = self.lattice.area
        eta = sqrt(pi/A)  # Ewald parameter

        n_range = np.arange(-neighbours, neighbours + 1)
        neighbouring_indices = it.product(n_range, n_range)

        R_xy  = np.array([a_1 * i[0] + a_2 * i[1] for i in neighbouring_indices])

        sep = r0 - R_xy[:, None]
        rho = np.linalg.norm(r0 - R_xy[:, None], axis=2)

        bloch = exp(-1j * np.dot(k_xy, R_xy.T))

        x = -sep[:, :, 0]/rho
        y = -sep[:, :, 1]/rho

        # Reshape arrays and vectors to allow for vectorized calculation.
        rho = rho[:, :, None]
        bloch = bloch[:, None, None]
        x = x[:, :, None]
        y = y[:, :, None]
        
        F_plus = exp(1j * k0 * rho) * sp.special.erfc(1j * k0/(2 * eta) + rho * eta)
        F_minus = exp(-1j * k0 * rho) * sp.special.erfc(-1j * k0/(2 * eta) + rho * eta) 
        
        fx = F_plus + F_minus

        df = -4 * eta / sqrt(pi) * exp(k0**2 / (4*eta**2) - rho**2 * eta**2) \
                    + 1j * k0 * (F_plus - F_minus)
        d2f = 8 * eta**3 * rho / sqrt(pi) * exp(k0**2 / (4*eta**2) - rho**2 * eta**2) \
                    - k0**2 * fx

        f1 = -fx / rho**3 + df / rho**2
        f2 = d2f / rho - 3 * df / rho**2 + 3 * fx / rho**3

        s0_spat  = 0.5 * fx / rho * bloch
        sxx_spat = 0.5 * (f1 + f2 * x**2) * bloch
        sxy_spat = 0.5 * (   + f2 * x * y) * bloch
        syy_spat = 0.5 * (f1 + f2 * y**2) * bloch
        szz_spat = 0.5 * (f1           ) * bloch

        # Remove NaN terms from sums including origin
        if np.linalg.norm(r0) == 0:
            for sum_term in [s0_spat, sxx_spat, sxy_spat, syy_spat, szz_spat]:
                sum_term[np.isnan(sum_term)] = 0

        S0_spat  = np.sum(s0_spat, axis=0)
        Sxx_spat = S0_spat +   1 / k0**2 * np.sum(sxx_spat, axis=0) 
        Sxy_spat =             1 / k0**2 * np.sum(sxy_spat, axis=0) 
        Syy_spat = S0_spat +   1 / k0**2 * np.sum(syy_spat, axis=0) 
        Szz_spat = S0_spat +   1 / k0**2 * np.sum(szz_spat, axis=0) 

        # Regularize the origin sum.
        if np.linalg.norm(r0) == 0:
            fp0   = -4 * eta / sqrt(pi) * exp((k0 / (2 * eta))**2)- 2 * k0 * (sp.special.erfc(0.5j * k0 / eta).imag)
            fppp0 = 8 * eta**3 / sqrt(pi) * exp((k0 / (2 * eta))**2) - k0**2 * fp0
            S0_spat  = S0_spat + 0.5 * (fp0 - 2j * k0)
            Sxx_spat = Sxx_spat + 0.5 * (fp0 - 2j * k0) + 0.5 * 1./ 3 * (1 / k0**2 * fppp0 + 2j * k0)
            Syy_spat = Syy_spat + 0.5 * (fp0 - 2j * k0) + 0.5 * 1./ 3 * (1 / k0**2 * fppp0 + 2j * k0)
            Szz_spat = Szz_spat + 0.5 * (fp0 - 2j * k0) + 0.5 * 1./ 3 * (1 / k0**2 * fppp0 + 2j * k0)

        spatial_sum = np.zeros((r0.shape[0], 3, 3, len(k0)), dtype=complex)
        
        spatial_sum[:, 0, 0, :] = Sxx_spat        
        spatial_sum[:, 0, 1, :] = Sxy_spat
        spatial_sum[:, 1, 1, :] = Syy_spat
        spatial_sum[:, 1, 0, :] = Sxy_spat
        spatial_sum[:, 2, 2, :] = Szz_spat 

        return k0**2 * spatial_sum

    def spectral_part(self, f, k_xy, r0, excl=False, neighbours = 1):
        k0 = 2 * np.pi * f / sp.constants.c * sqrt(self.eps_m)

        a_1, a_2 = self.lattice.get_lattice_vectors()
        b_1, b_2 = self.lattice.get_reciprocal_vectors(a_1, a_2)

        A = self.lattice.area
        eta = sqrt(np.pi / A)

        n_range = np.arange(-neighbours, neighbours + 1)
        neighbouring_indices = it.product(n_range, n_range)
        G_xy  = np.array([b_1 * i[0] + b_2 * i[1] for i in neighbouring_indices])

        beta_xy = k_xy + G_xy

        bloch = exp(-1j * np.dot(beta_xy, r0.T))

        beta = np.linalg.norm(beta_xy, axis = 1)
        betx = beta_xy[:, 0]
        bety = beta_xy[:, 1]

        k0 = k0[:, None]

        test = (beta / k0 > 1)
        kz = np.where(test, 
                sqrt(beta**2 - k0**2), 
                -1j * sqrt(k0**2 - beta**2))

        kz = kz[:, :, None]
        betx = betx[:, None]
        bety = bety[:, None]

        z = r0[:, 2]

        F_plus = exp(kz * z) * sp.special.erfc(kz / (2 * eta) + z * eta)
        F_minus = exp(-kz * z) * sp.special.erfc(kz / (2 * eta) - z * eta) 
        
        g = F_plus + F_minus
        d2g = kz**2 * g - kz * 4 * eta / sqrt(pi) * exp(-z**2 * eta**2 - kz**2 / (4 * eta**2))
        s0_spec = g/kz * bloch

        sxx_spec = -(betx * betx) * s0_spec
        sxy_spec = -(betx * bety) * s0_spec
        syy_spec = -(bety * bety) * s0_spec
        szz_spec = d2g/kz * bloch

        S0_spec  = pi / A * np.sum(s0_spec, axis=1) 
        Sxx_spec = S0_spec + 1 / k0**2 * pi / A * np.sum(sxx_spec, axis=1) 
        Sxy_spec =           1 / k0**2 * pi / A * np.sum(sxy_spec, axis=1) 
        Syy_spec = S0_spec + 1 / k0**2 * pi / A * np.sum(syy_spec, axis=1) 
        Szz_spec = S0_spec + 1 / k0**2 * pi / A * np.sum(szz_spec, axis=1) 
        
        spectral_sum = np.zeros((r0.shape[0], 3, 3, len(k0)), dtype=complex)

        spectral_sum[:, 0, 0] = Sxx_spec.T
        spectral_sum[:, 0, 1] = Sxy_spec.T
        
        spectral_sum[:, 1, 0] = Sxy_spec.T
        spectral_sum[:, 1, 1] = Syy_spec.T

        spectral_sum[:, 2, 2] = Szz_spec.T
          
        return k0[:, 0]**2 * spectral_sum

    def interaction_matrix(self, f, k_xy):
        k0 = 2 * np.pi * f / sp.constants.c * sqrt(self.eps_m)

        uc = np.array(self.lattice.unit_cell())
        N = len(uc)      
     
        R0 = np.array([0, 0, 0])
        origin = np.array([R0])
     
        if N == 1:       
            H = self.spatial_part(f, k_xy, origin, excl=True) + \
                    self.spectral_part(f, k_xy, origin, excl=True)

            H = H.reshape(3, 3, len(k0))

        else:
            H = np.zeros((N, N, 3, 3, len(k0)), dtype=complex)

            combos = np.vstack(np.triu_indices(N, k=1)).T 
            combos_lower = np.array([combos[:, 1], combos[:, 0]]).T

            r0 = uc[combos[:, 0]] - uc[combos[:, 1]]

            g_incl_upper = self.spatial_part(f, k_xy, r0) + \
                        self.spectral_part(f, k_xy, r0) 

            g_incl_lower = self.spatial_part(f, k_xy, -r0) + \
                        self.spectral_part(f, k_xy, -r0) 

            H[combos[:, 0], combos[:, 1], :, :] = g_incl_upper
            H[combos_lower[:, 0], combos_lower[:, 1], :, :] = g_incl_lower

            g_excl = self.spatial_part(f, k_xy, origin, excl=True) + \
                    self.spectral_part(f, k_xy, origin, excl=True)
            
            H = H.transpose(0, 2, 1, 3, 4).reshape(3*H.shape[0], 3*H.shape[0], len(k0))

            g_excl = g_excl.reshape(3, 3, len(k0))

            identity = np.eye(N, dtype=int).reshape(N, N, 1)
            diag = np.kron(identity, g_excl)

            H += diag

        return H

    def sf_mp(self, k_xy, f_range):
        H = self.interaction_matrix(f_range, k_xy)

        k0 = 2 * np.pi * f_range / sp.constants.c * sqrt(self.eps_m)
        N = self.lattice.size
        
        a_xy, a_z = self.element.polarisability(f_range)
        
        N_range = np.arange(3 * N)
        z_components = N_range[2::3]  # (0, 0, z).. (0, 0, z)..
        xy_components = np.delete(N_range, z_components)  # (x, y, 0).. (x, y, 0)..

        interaction_matrix = -H
        interaction_matrix[z_components, z_components] += 1/a_z
        interaction_matrix[xy_components, xy_components] += 1/a_xy

        ext = np.empty(len(f_range))
        for i in range(len(f_range)):
            eigs = sp.linalg.eigvals(interaction_matrix[:, :, i])
            ext[i] = np.sum((1/eigs)).imag
        return ext

    def spectral_function(self, f_range) -> np.array:
        k_xy = self.lattice.get_brillouin_zone(len(f_range))
        test = it.product(k_xy, [f_range])
        
        t_start = time.time()
        p = Pool()
        result = p.starmap(self.sf_mp, test)
        p.close()
        print(f'spectral function: t = {time.time() - t_start}s')

        return np.array(result).T
        
class OutPlane(Interaction):
    def __init__(self, lattice, element):
        super().__init__(lattice, element)

    def sf_mp(self, k_xy, f_range):
        H = self.interaction_matrix(f_range, k_xy)

        k0 = 2 * np.pi * f_range / sp.constants.c * sqrt(self.eps_m)
        N = self.lattice.size
        
        a_xy, a_z = self.element.polarisability(f_range)
        
        N_range = np.arange(3 * N)
        z_components = N_range[2::3]  # (0, 0, z).. (0, 0, z)..
        xy_components = np.delete(N_range, z_components)  # (x, y, 0).. (x, y, 0)..

        interaction_matrix = -H
        interaction_matrix[z_components, z_components] += 1/a_z
        interaction_matrix[xy_components, xy_components] += 1/a_xy

        H_z = interaction_matrix[z_components, :][:, z_components]

        ext = np.empty(len(f_range))
        for i in range(len(f_range)):
            eigs = sp.linalg.eigvals(H_z[:, :, i])
            ext[i] = np.sum((1/eigs)).imag
        return ext        


class InPlane(Interaction):
    def __init__(self, lattice, element):
        super().__init__(lattice, element)

    def sf_mp(self, k_xy, f_range):
        H = self.interaction_matrix(f_range, k_xy)

        k0 = 2 * np.pi * f_range / sp.constants.c * sqrt(self.eps_m)
        N = self.lattice.size
        
        a_xy, a_z = self.element.polarisability(f_range)
        
        N_range = np.arange(3 * N)
        z_components = N_range[2::3]  # (0, 0, z).. (0, 0, z)..
        xy_components = np.delete(N_range, z_components)  # (x, y, 0).. (x, y, 0)..

        interaction_matrix = -H
        interaction_matrix[z_components, z_components] += 1/a_z
        interaction_matrix[xy_components, xy_components] += 1/a_xy

        H_xy = interaction_matrix[xy_components, :][:, xy_components]

        ext = np.empty(len(f_range))
        for i in range(len(f_range)):
            eigs = sp.linalg.eigvals(H_xy[:, :, i])
            ext[i] = np.sum((1/eigs)).imag

        return ext       

class LayerInteraction(Interaction):
    def __init__(self, lattice, element, eps_m, e2 = 1, ex = None, d = 45E-9, dx = 10E-9, element2 = None):
        super().__init__(lattice, element, eps_m, element2)
        self.e2 = e2  # epsilon_substrate
        self.ex = ex  # epsilon_layer (function)
        self.d = d  # z pos of layer (-d -> -d-dx)
        self.dx = dx  # layer thickness

    def scattered_part(self, f, k_xy, r0, excl=False):
        r0 = np.array(r0)
        
        k0 = 2 * np.pi * f / sp.constants.c
        k1 = k0 * sqrt(self.eps_m)
        k2 = k0 * sqrt(self.e2)

        e1 = self.eps_m
        e2 = self.e2
        ex = self.ex(f)
        d = self.d
        dx = self.dx

        a_1, a_2 = self.lattice.get_lattice_vectors()
        b_1, b_2 = self.lattice.get_reciprocal_vectors()

        A = np.linalg.norm(np.cross(a_1, a_2))

        neighbours = 10
        n_range = np.arange(-neighbours, neighbours + 1)
        neighbouring_indices = it.product(n_range, n_range)
        G_xy  = np.array([b_1 * i[0] + b_2 * i[1] for i in neighbouring_indices])
        zeros = np.zeros((G_xy.shape[0], 1))
        G_xy = np.hstack((G_xy, zeros))

        beta_xy = k_xy + G_xy
        betx = beta_xy[:, 0][:, None]
        bety = beta_xy[:, 1][:, None]
        beta = np.linalg.norm(beta_xy, axis = 1)[:, None]

        bloch = np.exp(1j * np.dot(beta_xy, r0.T))

        # Note: Different branch cut convention to Ewald summation
        gam1 = np.where((beta/k0)**2 < e1, 
                    sqrt(e1 - beta**2 / k0**2), +1j * sqrt(beta**2 / k0**2 - e1))

        gamx = np.where((beta/k0)**2 < ex, 
                    sqrt(ex - beta**2 / k0**2), +1j * sqrt(beta**2 / k0**2 - ex))

        gam2 = np.where((beta/k0)**2 < e2, 
                    sqrt(e2 - beta**2 / k0**2), +1j * sqrt(beta**2 / k0**2 - e2))

        kz1 = (gam1 * k0)
        kzex = (gamx * k0)

        # Fresnel coefficients
        rs1x = ((gam1 - gamx) / (gam1 + gamx))
        rsx2 = ((gamx - gam2) / (gamx + gam2))
        rp1x = ((ex * gam1 - e1 * gamx) / (ex * gam1 + e1 * gamx))
        rpx2 = ((e2 * gamx - ex * gam2) / (e2 * gamx + ex * gam2))

        rs = (rs1x + rsx2 * exp(2j * kzex * dx)) / (1 + rs1x * rsx2 * exp(2j * kzex * dx))
        rp = (rp1x + rpx2 * exp(2j * kzex * dx)) / (1 + rp1x * rpx2 * exp(2j * kzex * dx))

        g0    = -1j / (2 * kz1) * exp(2j * kz1 * d) * bloch

        gxx   = g0 * (+ bety**2 / beta**2 * rs - kz1**2 / k1**2 * betx**2 / beta**2 * rp)
        gxy   = g0 * (-betx * bety / beta**2 * rs - kz1**2 / k1**2 * betx * bety / beta**2 * rp);
        gxz   = g0 * (-kz1 / k1**2 * betx * rp)
        gyy   = g0 * (+betx**2 / beta**2 * rs - kz1**2 / k1**2 * bety**2 / beta**2 * rp)
        gyz   = g0 * (-kz1 / k1**2 * bety * rp)
        gzz   = g0 * (+beta**2 / k1**2 * rp)

        beta_zero_index = np.where(beta == 0)[0]
        
        gxx = np.delete(gxx, beta_zero_index)
        gxy = np.delete(gxy, beta_zero_index)
        gyy = np.delete(gyy, beta_zero_index)

        Gxx = 1/A * np.sum(gxx, axis = 0)
        Gxy = 1/A * np.sum(gxy, axis = 0)
        Gxz = 1/A * np.sum(gxz, axis = 0)
        Gyy = 1/A * np.sum(gyy, axis = 0)
        Gyz = 1/A * np.sum(gyz, axis = 0)
        Gzz = 1/A * np.sum(gzz, axis = 0)

        if np.linalg.norm(r0) == 0:
            def sqrt_bc(a, b):
                if a >= b:
                    return sqrt(a - b)
                else:
                    return 1j * sqrt(b - a)

            def integral_xx_s(q, part='re'):
                gam1 = np.where(q**2 < e1, sqrt(e1 - q**2), +1j * sqrt(q**2 - e1))
                gamx = np.where(q**2 < ex, sqrt(ex - q**2), +1j * sqrt(q**2 - ex))
                gam2 = np.where(q**2 < e2, sqrt(e2 - q**2), +1j * sqrt(q**2 - e2))

                rs1x = (gam1 - gamx) / (gam1 + gamx)
                rsx2 = (gamx - gam2) / (gamx + gam2)
                rs = (rs1x + rsx2 * exp(2j * gamx * k0 * dx)) \
                        / (1 + rs1x * rsx2 * exp(2j * gamx * k0 * dx))

                out = q / gam1 * rs * exp(2j * gam1 * k0 * d)

                if 're' in part:
                    return out.real
                elif 'im' in part:
                    return out.imag

            def integral_xx_p(q, part='re'):
                gam1 = np.where(q**2 < e1, sqrt(e1 - q**2), +1j * sqrt(q**2 - e1))
                gamx = np.where(q**2 < ex, sqrt(ex - q**2), +1j * sqrt(q**2 - ex))
                gam2 = np.where(q**2 < e2, sqrt(e2 - q**2), +1j * sqrt(q**2 - e2))

                rp1x = (ex * gam1 - e1 * gamx) / (ex * gam1 + e1 * gamx)
                rpx2 = (e2 * gamx - ex * gam2) / (e2 * gamx + ex * gam2)
                rp = (rp1x + rpx2 * exp(2j * gamx * k0 * dx)) \
                        / (1 + rp1x * rpx2 * exp(2j * gamx * k0 * dx))

                out = q * gam1 * rp * exp(2j * gam1 * k0 * d)

                if 're' in part:
                    return out.real
                elif 'im' in part:
                    return out.imag

            def integral_zz_p(q, part='re'):
                gam1 = np.where(q**2 < e1, sqrt(e1 - q**2), +1j * sqrt(q**2 - e1))
                gamx = np.where(q**2 < ex, sqrt(ex - q**2), +1j * sqrt(q**2 - ex))
                gam2 = np.where(q**2 < e2, sqrt(e2 - q**2), +1j * sqrt(q**2 - e2))

                rp1x = (ex * gam1 - e1 * gamx) / (ex * gam1 + e1 * gamx)
                rpx2 = (e2 * gamx - ex * gam2) / (e2 * gamx + ex * gam2)
                rp = (rp1x + rpx2 * exp(2j * gamx * k0 * dx)) \
                        / (1 + rp1x * rpx2 * exp(2j * gamx * k0 * dx))  

                out = q**3 / gam1 * rp * exp(2j * gam1 * k0 * d)

                if 're' in part:
                    return out.real
                elif 'im' in part:
                    return out.imag
            
            int_max = 10/(2 * k0 * self.d)#4 * sqrt(e1)#
            pre = 0.9999999
            post = 1.0000001
 
            N = 50000

            q_min, q_max = 0, 5/(2 * k0 * self.d)
            dq = q_max - q_min
            q_range = np.linspace(q_min, q_max, N, endpoint=True)

            ixxs = sp.integrate.trapz(integral_xx_s(q_range, 're'), dx = dq/N) \
                    + 1j * sp.integrate.trapz(integral_xx_s(q_range, 'im'), dx = dq/N)
            ixxp = sp.integrate.trapz(integral_xx_p(q_range, 're'), dx = dq/N) \
                    + 1j * sp.integrate.trapz(integral_xx_p(q_range, 'im'), dx = dq/N)
            izzp = sp.integrate.trapz(integral_zz_p(q_range, 're'), dx = dq/N) \
                    + 1j * sp.integrate.trapz(integral_zz_p(q_range, 'im'), dx = dq/N)
            
            Gxx = Gxx - 1j/(8 * pi) * k0 * (ixxs - ixxp / e1)
            Gyy = Gyy - 1j/(8 * pi) * k0 * (ixxs - ixxp / e1)
            Gzz = Gzz - 1j/(4 * pi) * k0 / e1 * izzp

        scattered_sum = np.zeros((r0.shape[0], 3, 3), dtype=np.complex128)
        scattered_sum[:, 0, 0] = Gxx
        scattered_sum[:, 0, 1] = Gxy
        scattered_sum[:, 0, 2] = Gxz
        
        scattered_sum[:, 1, 0] = Gxy
        scattered_sum[:, 1, 1] = Gyy
        scattered_sum[:, 1, 2] = Gyz

        scattered_sum[:, 2, 0] = -Gxz
        scattered_sum[:, 2, 1] = -Gyz
        scattered_sum[:, 2, 2] = Gzz

        return  -4 * np.pi * k1**2 * scattered_sum

    def interaction_matrix(self, f, k_xy):
        k0 = 2 * np.pi * f / sp.constants.c * sqrt(self.e1)

        uc = np.array(self.lattice.unit_cell())
        N = len(uc)
        k_xy = np.concatenate((k_xy, [0]))        
     
        R0 = np.array([0, 0, 0])
        origin = np.array([R0])
     
        if N == 1:       
            H = self.spatial_part(f, k_xy, origin, excl=True) \
                    + self.spectral_part(f, k_xy, origin, excl=True) \
                        + self.scattered_part(f, k_xy, origin)
            H = H.reshape(3, 3)

        else:
            H = np.zeros((N, N, 3, 3), dtype=np.complex128)

            combos = np.vstack(np.triu_indices(N, k=1)).T 
            combos_lower = np.array([combos[:, 1], combos[:, 0]]).T

            r_combos = uc[combos[:, 0]] - uc[combos[:, 1]]
            zeros = np.zeros((r_combos.shape[0], 1))
            r0 = np.hstack((r_combos, zeros))

            g_incl_upper = self.spatial_part(f, k_xy, r0) \
                        + self.spectral_part(f, k_xy, r0) \
                        + self.scattered_part(f, k_xy, r0)

            g_incl_lower = self.spatial_part(f, k_xy, -r0) \
                        + self.spectral_part(f, k_xy, -r0) \
                        + self.scattered_part(f, k_xy, -r0)

            H[combos[:, 0], combos[:, 1], :, :] = g_incl_upper
            H[combos_lower[:, 0], combos_lower[:, 1], :, :] = g_incl_lower

            g_excl = self.spatial_part(f, k_xy, origin, excl=True) \
                    + self.spectral_part(f, k_xy, origin, excl=True) \
                    + self.scattered_part(f, k_xy, origin)
            
            H = H.transpose(0, 2, 1, 3).reshape(3*H.shape[0], 3*H.shape[0])
            g_excl = g_excl.reshape(3, 3)

            diag =  np.kron(np.eye(N, dtype=int), g_excl)

            H += diag

        return H


