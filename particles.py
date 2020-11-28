import numpy as np
from numpy import pi, sqrt, arctanh, arcsin, log
import scipy as sp
from scipy import constants

from dataclasses import dataclass
EV = sp.constants.e / sp.constants.h

@dataclass
class Metal:
    omega_p: float = 8.9 * EV
    loss: float = 0.038 * EV
    eps_inf: float = 5
    eps_m: float = 1

    def permittivity(freq) -> float:
        return eps_inf - ((omega_p * omega_p) / 
                                (freq * freq + 1j * loss * freq))

class Particle:    
    def __init__(self, radius = 10E-9, height = 10E-9, material = Metal()):
        self.radius = radius
        self.height = height
        self.material = material

        self.omega_p = self.material.omega_p
        self.loss = self.material.loss
        self.eps_inf = self.material.eps_inf
        self.eps_m = self.material.eps_m

        r = self.radius
        h0 = self.height
        self.volume = 4 / 3 * pi * r * r * h0

        if r > h0:
            ecc = sqrt((r**2 - h0**2)/r**2)
            
            self.L_z = (1 / ecc**2) * (1 - sqrt(1 - ecc**2)/ecc *arcsin(ecc))
            self.L_xy = 0.5 * (1 - self.L_z) 

            self.D_z = (3./4) * ((1 - 2 * ecc**2) * self.L_z + 1)
            self.D_xy = r / (2 * h0) * (3 / ecc * sqrt(1 - ecc**2) * arcsin(ecc) - self.D_z)

            self.omega_lsp = self.omega_p/np.sqrt(self.eps_inf - 1 + 1/self.L_xy),\
                                self.omega_p/np.sqrt(self.eps_inf - 1 + 1/self.L_z)

        elif r < h0:
            ecc = sqrt((h0**2 - r**2)/h0**2)  

            self.L_z = ((1 - ecc**2)/(ecc**3)) * (-ecc + 0.5*log((1 + ecc)/(1 - ecc)))
            self.L_xy = 0.5 * (1 - self.L_z)  

            self.D_z = (3./4) * (((1 + ecc**2)/(1 - ecc**2)) * self.L_z + 1)
            self.D_xy = (3 * r)/(4 * h0 * ecc) * 2 * arctanh(ecc) - (r * self.D_z)/(2 * h0)

            self.omega_lsp = self.omega_p/np.sqrt(self.eps_inf - 1 + 1/self.L_xy),\
                                self.omega_p/np.sqrt(self.eps_inf - 1 + 1/self.L_z)

        elif r == h0:
            self.L_z, self.L_xy = 1/3, 1/3
            self.D_z, self.D_xy = 1, 1
            self.omega_lsp = self.omega_p / np.sqrt(self.eps_inf + 2),\
                                self.omega_p / np.sqrt(self.eps_inf +2) 

    def __str__(self):
        r = self.radius / 1E-9
        h = self.height / 1E-9
        w_p = self.omega_p / EV
        loss = self.loss / EV

        return f'Particle: r = {r} nm, h = {h} nm, w_p = {w_p} eV, loss = {loss} eV, eps_inf = {self.eps_inf}, eps_m = {self.eps_m}'

    def get_omega_lsp(self, pol = 'z'):
        if pol == 'xy':
            return self.omega_lsp[0]
        elif pol == 'z':
            return self.omega_lsp[1]

    def polarisability(self, freq, static=False) -> tuple:
        k = 2 * pi * freq / sp.constants.c * sqrt(self.eps_m)

        h0 = self.height
        r = self.radius
        V = self.volume

        permittivity = self.material.permittivity(freq) / self.eps_m

        alpha_z_static = V/(4*pi) * ((permittivity - 1) / (1 + self.L_z * (permittivity - 1)))
        alpha_xy_static = V/(4*pi) * ((permittivity - 1) / (1 + self.L_xy * (permittivity - 1)))

        alpha_z_radiative = alpha_z_static/(1 - ((k**2 * self.D_z * alpha_z_static)/h0) - 1j * 2.  *k**3 * alpha_z_static/3) 
        alpha_xy_radiative = alpha_xy_static/(1 - (k**2 * self.D_xy * alpha_xy_static)/r - 1j * 2. * k**3 * alpha_xy_static/3) 

        if static:
            return alpha_xy_static, alpha_z_static
        else:
            return alpha_xy_radiative, alpha_z_radiative

    def eig_to_freq(self, eig, pol = 'z'):
        w_p = self.omega_p
        eps_inf = self.eps_inf
        V = self.volume
        L_z = self.L_z
        L_xy = self.L_xy
        
        if pol == 'xy':
            num = L_xy * w_p**2 - eig * V/(4*np.pi) * w_p**2
            den = 1 + L_xy * eps_inf - L_xy - eig * V/(4*np.pi) * (eps_inf - 1)
            
        elif pol == 'z':
            num = L_z * w_p**2 - eig * V/(4*np.pi) * w_p**2
            den = 1 + L_z * eps_inf - L_z - eig * V/(4*np.pi) * (eps_inf - 1)

        return np.sqrt(num/den)


