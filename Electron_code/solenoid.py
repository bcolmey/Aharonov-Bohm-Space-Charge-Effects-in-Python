import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def vector_potential_solenoid(I, R, x, y, x0=0, y0=0):
    epsilon_0 = 8.85e-12  # permittivity of free space
    c = 3e8  # speed of light in vacuum
    A_x = np.zeros((len(y), len(x)))
    A_y = np.zeros((len(y), len(x)))

    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            r_prime = np.sqrt((xi - x0)**2 + (yi - y0)**2)
            if r_prime > R:  # Only compute the vector potential outside the solenoid
                A_r = (I * R**2) / (2*epsilon_0 * c**2 * r_prime)
                A_x[j, i] = -A_r * (yi - y0) / r_prime
                A_y[j, i] = A_r * (xi - x0) / r_prime

    return A_x.ravel(), A_y.ravel()


# Define your x and y arrays, and the solenoid parameters
I = 100000
R = 0.1

print()

x_min, x_max, dx = -3, 3, 0.05
y_min, y_max, dy = -3, 3, 0.05
x, y = np.arange(x_min, x_max+dx, dx), np.arange(y_min, y_max+dy, dy)

# Calculate the vector potential
x0, y0 = 0.1, 0  # Desired center coordinates

A_x, A_y = vector_potential_solenoid(I, R, x, y, x0, y0)
X, Y = np.meshgrid(x, y)
n = 5 # Keep every nth arrow
biggest_A_value = np.max(np.abs(A_x))
print(f"The biggest A value is: {biggest_A_value}")
# Reshape A_x and A_y back to 2D arrays
A_x_2d = A_x.reshape(len(y), len(x))
A_y_2d = A_y.reshape(len(y), len(x))

x_desired = 0.1
y_desired = 0.25
A_x_desired = np.interp(x_desired, x, A_x_2d[:, np.argmin(np.abs(y - y_desired))])
A_y_desired = np.interp(y_desired, y, A_y_2d[np.argmin(np.abs(x - x_desired)), :])
print(f"Vector potential A at x={x_desired}, y={y_desired} is A_x={A_x_desired:.3e} T*m and A_y={A_y_desired:.3e} T*m")

# Plot the vector field using quiver
plt.figure()
plt.quiver(X.ravel()[::n], Y.ravel()[::n], A_x[::n] * (biggest_A_value), A_y[::n] * ( biggest_A_value))

# Add labels, a title, and a circle representing the solenoid
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Potential (A_x, A_y) around a Solenoid')

# Draw the solenoid as a filled circle, color inside where the magnetic field B is present
sol = plt.Circle((x0, y0), R, color='r', fill=True, alpha=0.3)
plt.text(x0, y0, 'B field', fontsize=12, ha='center', va='center', color='black')
plt.gca().add_artist(sol)

# Show the plot
plt.show()