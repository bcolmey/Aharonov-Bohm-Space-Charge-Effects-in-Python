################################################
#     Creating relevant functions              #
################################################
import numpy as np
from scipy.constants import epsilon_0
e = 1.602e-19  # Elementary charge in C
# Define our gaussian electron packet
def gauss_xy(x, y, delta_x, delta_y, x0, y0, kx0, ky0):
    # Compute the Gaussian function with given parameters and return the result
    return 1/(2*delta_x**2*np.pi)**(1/4) * 1/(2*delta_y**2*np.pi)**(1/4) * np.exp(-((x-x0)**2)/(2*delta_x**2)) * np.exp(-((y-y0)**2)/(2*delta_y**2)) * np.exp( 1.j * (kx0*x + ky0*y))

# Define a gaussian potential function
def potential(V0, x0, sigma_x, y0, sigma_y, x, y):
    # Initialize V with the correct shape
    V = np.zeros(len(x) * len(y))
    # Store the length of y
    size_y = len(y)
    # Loop over y values
    for i, yi in enumerate(y):
        # Loop over x values
        for j, xj in enumerate(x):
            # Compute the potential for each xj and yi and store it in V
            V[i + j * size_y] = V0 * np.exp(-((xj - x0) ** 2 / (2 * sigma_x ** 2) + (yi - y0) ** 2 / (2 * sigma_y ** 2)))
    # Return the potential V
    return V

# Define the function to compute the Coulomb potential
def colomb_potential(rho, x, y, dx, dy):
    # Define Bohr radius in meters
    a0 = 5.29177210903e-11
    # Convert dx to meters
    dx_m = dx * a0
    # Convert dy to meters
    dy_m = dy * a0
    
    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x, y, indexing='ij')
    # Compute the x differences
    x_diff = (xx * a0)[:, :, np.newaxis, np.newaxis] - (xx * a0)
    # Compute the y differences
    y_diff = (yy * a0)[:, :, np.newaxis, np.newaxis] - (yy * a0)

    # Compute the distance matrix r
    r = np.sqrt(x_diff**2 + y_diff**2)

    # Set the diagonal elements of r to a very small number
    r[r == 0] = np.finfo(float).eps
    # Calculate potential in Joules
    V = np.sum(rho * dx_m * dy_m / (4 * np.pi * epsilon_0 * r), axis=(2, 3))
    # Convert potential to electron volts
    V_eV = (V/e).T.flatten()
    # Return the potential in electron volts
    return V_eV

# Define the potential function using the Heaviside step function
def potential_heaviside(V0, x0, xf, y0, yf, x, y):
    # Initialize V with the correct shape
    V = np.zeros(len(x)*len(y))
    # Store the length of y
    size_y = len(y)
    # Loop over y values
    for i,yi in enumerate(y):
        # Loop over x values
        for j,xj in enumerate(x):
            # Check if xj and yi are within the specified limits
            if (xj >= x0) and (xj <= xf) and (yi >= y0) and (yi <= yf):
                # Set the potential to V0 for the current xj and yi
                V[i+j*size_y] = V0
            else:
                # Set the potential to 0 for the current xj and yi
                V[i+j*size_y] = 0
    # Return the potential V
    return V

# Define the vector potential function for a solenoid
def vector_potential_solenoid(I, R, x, y, x0=0, y0=0):
    # Define the vacuum permeability constant
    mu_0 = 4 * np.pi * 1e-7
    # Initialize A_x and A_y with the correct shapes
    A_x = np.zeros((len(y), len(x)))
    A_y = np.zeros((len(y), len(x)))
    # Loop over x values
    for i, xi in enumerate(x):
        # Loop over y values
        for j, yi in enumerate(y):
            # Compute the distance from the center of the solenoid
            r = np.sqrt((xi - x0)**2 + (yi - y0)**2)
            # Only compute the vector potential outside the solenoid
            if r > R:
                # Calculate the radial component of the vector potential
                A_r = (mu_0 * I * R**2) / (4 * np.pi * r**3)
                # Calculate the x component of the vector potential
                A_x[j, i] = -A_r * (yi - y0)
                # Calculate the y component of the vector potential
                A_y[j, i] = A_r * (xi - x0)
    # Return the vector potential components A_x and A_y
    print(A_x.max())
    return A_x.ravel(), A_y.ravel()

# Define the function to calculate the total charge
def total_charge(rho, dx, dy):
    # Define Bohr radius in meters
    a0 = 5.29177210903e-11
    # Convert dx to meters
    dx_m = dx * a0
    # Convert dy to meters
    dy_m = dy * a0

    # Calculate the total charge by summing the charge density times the area elements
    total_q = np.sum(rho * dx_m * dy_m)
    # Return the total charge
    return total_q

