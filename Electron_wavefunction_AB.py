import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.sparse.linalg import bicgstab


class WaveFunctionAB(object):

    def __init__(self, x, y, psi_0, V, dt, A_x, A_y, hbar, m=1, t0=0.0, q=1.602e-19):
        self.x = x
        self.y = y
        self.A_x = A_x
        self.A_y = A_y
        self.size_x = len(x)
        self.size_y = len(y)
        dimension = self.size_x * self.size_y
        self.psi = np.array(psi_0, dtype=np.complex128)
        self.V = V
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.pml_width = 10  # Set the width of the perfectly matched layer (PML) boundary to 10 grid points.
        self.pml_strength = 0.5  # Set the strength of the PML boundary to 0.5, a measure of the absorption rate of the boundary.
        self.pml_coefficients_x, self.pml_coefficients_y = self.compute_pml_coefficients(self.pml_width, self.pml_strength)  # Compute the PML coefficients for the x and y directions using the provided width and strength.
        alpha = dt / (4 * self.dx**2)  # Compute the alpha constant, which is used in the matrix equations, based on the time step (dt) and spatial grid spacing (dx)
        self.alpha = alpha
        q_hbar= q/hbar
        # Building the first matrix to solve the system (A from Ax_{n+1}=Mx_{n})
        N = (self.size_x - 1) * (self.size_y - 1)  # Calculate the number of non-zero elements in the sparse matrix system for A * x_{n+1} = M * x_n.
        size = 5 * N + 2 * self.size_x + 2 * (self.size_y - 2)
        I = np.zeros(size)  # Initialize the row indices (I), column indices (J), and values (K) of the sparse matrices.
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0  # Initialize the index counter for the sparse matrix elements.
        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                # Conditions at the reflecting y-boundaries
                if i == 0 or i == (self.size_y - 1):  # Check if the current grid point (i, j) is on the top or bottom boundary of the simulation domain.
                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 1
                    k += 1

                # Conditions at the reflecting x-boundaries
                elif j == 0 or j == (self.size_x - 1):
                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 1
                    k += 1

                # Points inside the domain
                else:
                    # Central point (i,j)
                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 1.0j - 4 * alpha - V[i + j * self.size_y] * dt / 2
                    k += 1

                    # Point (i-1,j)
                    I[k] = i + j * self.size_y
                    J[k] = (i - 1) + j * self.size_y
                    K[k] = alpha + 0.5j * q_hbar * (self.A_x[i + j * self.size_y] * dt / (self.dx) - self.A_y[i + j * self.size_y] * dt / (self.dy))
                    k += 1

                    # Point (i+1,j)
                    I[k] = i + j * self.size_y
                    J[k] = (i + 1) + j * self.size_y
                    K[k] = alpha - 0.5j * q_hbar * (self.A_x[i + j * self.size_y] * dt / (self.dx) + self.A_y[i + j * self.size_y] * dt / (self.dy))
                    k += 1

                    # Point (i,j-1)
                    I[k] = i + j * self.size_y
                    J[k] = i + (j - 1) * self.size_y
                    K[k] = alpha + 0.5j * q_hbar * (self.A_y[i + j * self.size_y] * dt / (self.dy) - self.A_x[i + j * self.size_y] * dt / (self.dx))
                    k += 1

                    # Point (i,j+1)
                    I[k] = i + j * self.size_y
                    J[k] = i + (j + 1) * self.size_y
                    K[k] = alpha - 0.5j * q_hbar * (self.A_y[i + j * self.size_y] * dt / (self.dy) + self.A_x[i + j * self.size_y] * dt / (self.dx))
                    k += 1

        self.Mat1 = sparse.coo_matrix((K, (I, J)), shape=(dimension, dimension)).tocsc()

        # Building the second matrix to solve the system (M from Ax_{n+1}=Mx_{n})
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0, self.size_y):
            start_time = time.time()
            for j in range(0, self.size_x):

                if i == 0 or i == (self.size_y - 1):
                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 0
                    k += 1

                elif j == 0 or j == (self.size_x - 1):
                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 0
                    k += 1

                else:

                    I[k] = i + j * self.size_y
                    J[k] = i + j * self.size_y
                    K[k] = 1.0j + 4 * alpha + V[i + j * self.size_y] * dt / 2
                    k += 1

                    # Point (i-1,j)
                    I[k] = i + j * self.size_y
                    J[k] = (i - 1) + j * self.size_y
                    K[k] = -alpha - 0.5j * q_hbar * (self.A_x[i + j * self.size_y] * dt / (self.dx) - self.A_y[i + j * self.size_y] * dt / (self.dy))
                    k += 1

                    # Point (i+1,j)
                    I[k] = i + j * self.size_y
                    J[k] = (i + 1) + j * self.size_y
                    K[k] = -alpha + 0.5j * q_hbar * (self.A_x[i + j * self.size_y] * dt / (self.dx) + self.A_y[i + j * self.size_y] * dt / (self.dy))
                    k += 1

                    # Point (i,j-1)
                    I[k] = i + j * self.size_y
                    J[k] = i + (j - 1) * self.size_y
                    K[k] = -alpha - 0.5j * q_hbar * (self.A_y[i + j * self.size_y] * dt / (self.dy) - self.A_x[i + j * self.size_y] * dt / (self.dx))
                    k += 1

                    # Point (i,j+1)
                    I[k] = i + j * self.size_y
                    J[k] = i + (j + 1) * self.size_y
                    K[k] = -alpha + 0.5j * q_hbar * (self.A_y[i + j * self.size_y] * dt / (self.dy) + self.A_x[i + j * self.size_y] * dt / (self.dx))
                    k += 1
            if (i == (self.size_y - 1)):
                end_time = time.time()
                elapsed_time = end_time - start_time

        self.Mat2 = sparse.coo_matrix((K, (I, J)), shape=(dimension, dimension)).tocsc()

    def get_prob(self):
        """
        Calculate and return the probability density (magnitude squared) of the wave function.
        """
        return (abs(self.psi)) ** 2

    def step(self):
        """
        Perform a single time step in the simulation using bicgstab solver.
        Update the wave function by applying PML coefficients and solving the linear system.
        """

        for i in range(self.size_y):
            for j in range(self.size_x):
                self.psi[i + j * self.size_y] *= self.pml_coefficients_x[j] * self.pml_coefficients_y[i]

        self.psi, info = bicgstab(self.Mat1, self.Mat2.dot(self.psi), x0=self.psi, tol=1e-6)
        self.t += self.dt

    def compute_pml_coefficients(self, pml_width, pml_strength):
        """
        Compute the PML coefficients for x and y directions based on the given width and strength.
        These coefficients are used to absorb boundary reflections.
        """
        pml_coefficients_x = np.ones(self.size_x)
        pml_coefficients_y = np.ones(self.size_y)

        for j in range(self.size_x):
            if j < pml_width:
                pml_coefficients_x[j] = 1 - pml_strength * (1 - j / pml_width) ** 2
            elif j >= self.size_x - pml_width:
                pml_coefficients_x[j] = 1 - pml_strength * (1 - (self.size_x - j - 1) / pml_width) ** 2

        for i in range(self.size_y):
            if i < pml_width:
                pml_coefficients_y[i] = 1 - pml_strength * (1 - i / pml_width) ** 2
            elif i >= self.size_y - pml_width:
                pml_coefficients_y[i] = 1 - pml_strength * (1 - (self.size_y - i - 1) / pml_width) ** 2

        return pml_coefficients_x, pml_coefficients_y



