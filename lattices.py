import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import norm

def rot(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])

class Lattice:
    def __init__(self, a1 : np.array, a2 : np.array):
        self.a1, self.a2 = a1, a2

    def unit_cell(self):
        return NotImplementedError()

    def get_reciprocal_vectors(self, a1, a2) -> tuple:
        b1 = 2 * pi * np.dot(rot(pi / 2), a2) / np.dot(a1, np.dot(rot(pi / 2), a2))
        b2 = 2 * pi * np.dot(rot(pi / 2), a1) / np.dot(a2, np.dot(rot(pi / 2), a1))
        return b1, b2

    def get_lattice_vectors(self) -> tuple:
        return self.a1, self.a2

    @property
    def size(self) -> int: 
        return len(self.unit_cell())

    @property
    def area(self) -> float:
        return np.linalg.norm(np.cross(self.a1, self.a2))

class Triangular(Lattice):
    def __init__(self, lattice_constant:float):
        self.a0 = lattice_constant

        self.a1 = np.array([1.5 / sqrt(3), 0.5, 0]) * self.a0 
        self.a2 = np.array([1.5 / sqrt(3), -0.5, 0]) * self.a0

        self.b1, self.b2 = self.get_reciprocal_vectors(self.a1, self.a2)

        Gamma = np.array([0, 0, 0])
        K = np.array([(2 * sqrt(3) * pi) / (3 * self.a0), (2 * pi) / (3 * self.a0), 0])
        K_prime = np.array([(2 * sqrt(3) * pi)/(3 * self.a0), -(2 * pi)/(3 * self.a0), 0])
        M = np.array([(2 * sqrt(3) * pi) / (3 * self.a0), 0, 0])

        self.bz = {'Gamma': Gamma, 'K': K, 'K_prime': K_prime, 'M': M}

    def __str__(self):
        return f'Triangular lattice: a0 = {self.a0/1E-9:.2f} nm'

    def unit_cell(self) -> np.array:
        return np.array([np.array([0, 0, 0])])

    def get_bz_path(self, N:int) -> np.array:
        paths = np.array([norm(self.bz['M'] - self.bz['Gamma']),
                        norm(self.bz['Gamma'] - self.bz['K']),
                        norm(self.bz['K'] - self.bz['M'])])
        n_paths = (paths / np.sum(paths) * N).astype(int)

        return n_paths

    def get_bz_labels(self, N:int) -> dict:
        label_index = np.cumsum(self.get_bz_path(N))
        label_index = np.insert(label_index, 0, [0])

        labels = [r'M', r'$\Gamma$', r'K', r'M']
        labels_dict = dict(zip(label_index, labels))

        return labels_dict

    def get_brillouin_zone(self, N:int) -> np.array:
        n_paths = self.get_bz_path(N)

        M_Gamma = np.linspace(self.bz['M'], self.bz['Gamma'], n_paths[0], endpoint = False)
        Gamma_K = np.linspace(self.bz['Gamma'], self.bz['K'], n_paths[1], endpoint = False)
        K_M = np.linspace(self.bz['K'], self.bz['M'], N - np.sum(n_paths[:-1]))

        path = np.vstack((M_Gamma, Gamma_K, K_M))

        return path

class Kagome(Triangular):
    def __init__(self, lattice_constant:float, scale:float = 0, rotation:float = 0):
        super().__init__(lattice_constant)

        self.R0 = lattice_constant / (2 * sqrt(3))
        self.s = scale
        self.rot = rotation

    def __str__(self):
        return f'Kagome lattice: a0 = {self.a0/1E-9} nm, R0 = {self.a0/1E-9/2} nm'

    def unit_cell(self) -> np.array:
        r = self.R0 * (1 + self.s)
        
        angles = np.array([0, 2*pi/3, 4*pi/3]) + self.rot
        x = r * cos(angles)
        y = r * sin(angles)
        z = np.zeros(len(x))

        positions = np.vstack((x, y, z)).T
    
        return positions

class BreathingHoneycomb(Triangular):
    def __init__(self, lattice_constant:float, scale:float = 1, rotation:float = 0):
        super().__init__(lattice_constant)

        self.R0 = lattice_constant / 3
        self.s = scale
        self.rot = rotation

    def __str__(self):
        return f'Breathing honeycomb lattice: a0 = {self.a0/1E-9:.2f} nm, R0 = {self.R0/1E-9:.2f} nm'

    def unit_cell(self) -> np.array:
        r = self.R0 * self.s
        
        angles = np.arange(0, 2*np.pi, np.pi/3) + np.pi/6 + self.rot
        
        x = r * cos(angles)
        y = r * sin(angles)
        z = np.zeros(len(x))

        positions = np.vstack((x, y, z)).T
    
        return positions

class Honeycomb(Triangular):
    def __init__(self, lattice_constant:float):
        super().__init__(lattice_constant)

        self.R0 = lattice_constant / np.sqrt(3) 

    def __str__(self):
        return f'Breathing honeycomb lattice: a0 = {self.a0/1E-9:.2f} nm, R0 = {self.R0/1E-9:.2f} nm'

    def unit_cell(self) -> np.array:
        positions = np.array([np.array([0, 0, 0]), np.array([self.R0, 0, 0])])
    
        return positions

class Square(Lattice):
    def __init__(self, lattice_constant:float):
        self.a0 = lattice_constant

        self.a1 = np.array([1, 0, 0]) * self.a0
        self.a2 = np.array([0, 1, 0]) * self.a0

        self.b1, self.b2 = self.get_reciprocal_vectors(self.a1, self.a2)

        Gamma = np.array([0, 0, 0])
        X = np.array([pi / self.a0, 0, 0])
        M = np.array([pi / self.a0, pi / self.a0, 0])

        self.bz = {'Gamma': Gamma, 'X': X, 'M': M}

    def __str__(self):
        return f'Square lattice: a0 = {self.a0/1E-9:.2f} nm'

    def unit_cell(self) -> np.array:
        return np.array([np.array([0, 0, 0])])

    def get_bz_path(self, N:int) -> np.array:
        paths = np.array([norm(self.bz['Gamma'] - self.bz['X']),
                        norm(self.bz['X'] - self.bz['M']),
                        norm(self.bz['M'] - self.bz['Gamma'])])
        n_paths = (paths / np.sum(paths) * N).astype(int)

        return n_paths

    def get_bz_labels(self, N:int) -> dict:
        label_index = np.cumsum(self.get_bz_path(N))
        label_index = np.insert(label_index, 0, [0])

        labels = [r'$\Gamma$', r'X', r'M', r'$\Gamma$']
        labels_dict = dict(zip(label_index, labels))

        return labels_dict

    def get_brillouin_zone(self, N:int) -> np.array:
        n_paths = self.get_bz_path(N)

        Gamma_X = np.linspace(self.bz['Gamma'], self.bz['X'], n_paths[0], endpoint = False)
        X_M = np.linspace(self.bz['X'], self.bz['M'], n_paths[0], endpoint = False)
        M_Gamma =  np.linspace(self.bz['M'], self.bz['Gamma'], N - np.sum(n_paths[:-1]))

        path = np.vstack((Gamma_X, X_M, M_Gamma))

        return path

