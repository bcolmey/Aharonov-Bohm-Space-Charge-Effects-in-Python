import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.sparse.linalg import bicgstab
from scipy.constants import epsilon_0


class WaveFunction(object):
    

    def __init__(self, x, y, psi_0, V, dt, hbar=1, m=1, t0=0.0):
        start = time.time()
                 
        self.x = np.array(x)
        self.y = np.array(y)
        self.size_x = len(x)
        self.size_y = len(y)
        self.psi = np.array(psi_0, dtype=np.complex128)
        self.V = np.array(V, dtype=np.complex128)
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.pml_width = 10
        self.pml_strength = 0.5
        self.pml_coefficients_x, self.pml_coefficients_y = self.compute_pml_coefficients(self.pml_width, self.pml_strength)

        
        alpha = dt/(4*self.dx**2)
        self.alpha = alpha
        self.size_x = len(x)
        self.size_y = len(y)
        dimension = self.size_x*self.size_y
        

        #Building the first matrix to solve the system (A from Ax_{n+1}=Mx_{n})
        N = (self.size_x-1)*(self.size_y-1)
        size = 5*N + 2*self.size_x + 2*(self.size_y-2)
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)
        
        #print(f"dimension: {dimension}")
        #print(f"psi_0 shape: {psi_0.shape}")
        #print(f"V_xyz shape: {V_xy.shape}")
        
        k = 0
        for i in range(0,self.size_y):
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j - 4*alpha - V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = alpha
                    k += 1

        self.Mat1 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()

        #Building the second matrix to solve the system (M from Ax_{n+1}=Mx_{n})
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0,self.size_y):
            start_time= time.time()
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j + 4*alpha + V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = -alpha
                    k += 1
            if (i == (self.size_y -1)): 
                #print("made it here")
                end_time = time.time()
                elapsed_time = end_time - start_time
                #print(f"one full loop time: {elapsed_time:.2f} seconds")        
        self.Mat2 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()
        
    def get_prob(self):
        return (abs(self.psi))**2

    def compute_norm(self):
        return np.trapz(np.trapz((self.get_prob()).reshape(self.size_y,self.size_x), self.x).real, self.y).real
    
    def step(self):
    # The rest of the step method remains unchanged

        for i in range(self.size_y):
            for j in range(self.size_x):
                self.psi[i + j * self.size_y] *= self.pml_coefficients_x[j] * self.pml_coefficients_y[i]
              
        # Update the state
        self.psi, info = bicgstab(self.Mat1, self.Mat2.dot(self.psi), x0=self.psi, tol=1e-6)
        
        #self.psi = spsolve(self.Mat1, self.Mat2.dot(self.psi))
        
        # Update time
        self.t += self.dt
    
        
    def compute_pml_coefficients(self, pml_width, pml_strength):
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