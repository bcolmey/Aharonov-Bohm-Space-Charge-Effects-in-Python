
import seaborn as sns
sns.set(style="whitegrid")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from matplotlib.animation import FFMpegWriter
import time
import datetime
from Electron_wavefunction_AB import WaveFunctionAB
from electron_wavefunction_test import WaveFunction
from potential_functions import colomb_potential
from potential_functions import gauss_xy
from potential_functions import potential 
from potential_functions import potential_heaviside
from potential_functions import total_charge
from potential_functions import vector_potential_solenoid



#####################################
#       1) Create the system        #
#####################################
start_time = time.time()
nb_frame = 100
Aharanov_Bohm= True
Space_charge = False
Two_slit = True

dt = 0.005
t = 0
num_electrons = 1 #number of electrons in wavepacket
hbar = 1.05e-34  # Planck's constant divided by 2π (J·s)
m = 9.1e-31  # Mass of an electron (kg)
e = 1.6e-19  # Elementary charge in Coulombs
epsilon_0 = 8.85e-12 #m-3 kg-1 s4 A2
x_min, x_max, dx = -12, 12, 0.1
y_min, y_max, dy = -12, 12, 0.1
x, y = np.arange(x_min, x_max + dx, dx), np.arange(y_min, y_max + dy, dy)
#ni = 250
# Set parameters for initial Gaussian wave packet
x0 = -3
y0 = 0
kx0 = 400
ky0 = 0
delta_x = 2
delta_y = 2
x_desired = x_max-6
 
# Create the initial wave packet
size_x = len(x)
size_y = len(y)
xx, yy = np.meshgrid(x, y)
psi_0 = gauss_xy(xx, yy, delta_x, delta_y, x0, y0, kx0, ky0).transpose().reshape(size_x * size_y)
psi_0_2D = psi_0.reshape(size_y, size_x)
initial_norm = np.sqrt(np.trapz(np.trapz((abs(psi_0_2D))**2, x), y))
psi_0 = psi_0 / initial_norm


#####################################
#       2) Defining Potentials      #
#####################################

if (Two_slit == True):
    # Define parameters for the double slit (potential barrier) 
    x01, xf1, y01, yf1 = 0, 0.05, y.min(), -1.00  # Adjusted slit width
    x02, xf2, y02, yf2 = x01, xf1, -yf1, y.max()
    x03, xf3, y03, yf3 = x01, xf1, yf1 + 0.7, -yf1 - 0.7
    V0 = 4000
    V_xy = potential_heaviside(V0, x01, xf1, y01, yf1, x, y) + potential_heaviside(V0, x02, xf2, y02, yf2, x, y) + potential_heaviside(V0, x03, xf3, y03, yf3, x, y)
else:
    V_xy = np.zeros(len(x)*len(y))


if (Aharanov_Bohm== True):
    I = 1e-8  # Current in amps
    R = 0.2  # Radius of solenoid
    solenoid_y0, solenoid_x0 = 0, 0.5
    A_x, A_y = vector_potential_solenoid(I, R, x, y, solenoid_y0, solenoid_x0)
    S = WaveFunctionAB(x=x, y=y, psi_0=psi_0, V=V_xy, dt=dt, A_x=A_x, A_y=A_y, hbar=hbar, m=1)
else:
    S = WaveFunction(x=x, y=y, psi_0=psi_0, V=V_xy, dt=dt, hbar=hbar, m=1)

z = S.get_prob().reshape(size_x, size_y).transpose()

if (Space_charge == True):
    # Calculate the normalization factor for charge density
    norm_factor = np.sum(z * dx * dy)
    z_normalized = z * (e * num_electrons / norm_factor)
    total_charge = np.sum(z_normalized * dx * dy)
    #print(total_charge)
    rho = z_normalized
    colomb = colomb_potential(rho, x, y, dx, dy)
    vmax=colomb.max()
    Vcolomb = V_xy + colomb_potential(rho, x, y, dx, dy)
else:
    Vcolomb = V_xy
      

#####################################
#       3) Setting up plots         #
#####################################
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.5)  

#fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(12, 6))
#plt.subplots_adjust(wspace=0.4)  

# Set up the first subplot (ax1) for probability density plot
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '3%', '3%')
# Create colormap and normalized probability density for ax1
color_map = plt.get_cmap('magma')
#normalized_z = z / np.max(z)
scatter = ax1.scatter(xx, yy, c=z, cmap=color_map)
cbar1 = fig.colorbar(scatter, cax=cax1)


if (Space_charge == True):
    #Display the potentials on the second subplot (ax2)
    Vcolomb_2D = Vcolomb.reshape(size_x, size_y).T
    potential_contour = ax2.contourf(xx, yy, Vcolomb_2D)
    cbar2 = fig.colorbar(potential_contour, ax=ax2)
    cbar2.set_label('Electric Potential (V)', fontsize=14)
    cbar2.ax.yaxis.label.set_color('red')


# Find the index of x-coordinate closest to the desired value
k = abs(x - x_desired).argmin()
# Initialize an empty array to store probability density values
coupe = np.zeros((nb_frame, len(z[:, k])))

vmin=0
zmax= 0
#####################################
#   4) Performing time evolution    #
#####################################
def animate(i):
    global z, S, rho, Vcolomb, potential_contour, cbar2,zmax # Make S a global variable to access it outside the function
    if (Space_charge ==True and Aharanov_Bohm ==False):#Both methods require updating the wavefunction at each step but aharanov bohm
    #requires using the different wavefunction file WavefunctionAB which includes Ax, Ay
        V_new = colomb_potential(rho, x, y, dx, dy)+V_xy
        #potential(V0,a,b,c,d,x,y)
        S = WaveFunction(x=x, y=y, psi_0=S.psi, V=V_new, dt=dt, hbar=hbar, m=m)
        #S.psi = S.psi / np.sqrt(np.sum(np.abs(S.psi)**2) * dx * dy)
        
    if (Space_charge ==True and Aharanov_Bohm ==True):
        V_new = colomb_potential(rho, x, y, dx, dy)+V_xy
        S = WaveFunctionAB(x=x, y=y, psi_0=psi_0, V=V_new, dt=dt, A_x=A_x, A_y=A_y, hbar=hbar, m=1)
        #S.psi = S.psi / np.sqrt(np.sum(np.abs(S.psi)**2) * dx * dy)
        
    # Update the wavefunction at each time step
    S.step()
    z = S.get_prob().reshape(size_x, size_y).transpose()
    
    #print(f"Total distribution: {total_distribution}")
    coupe[i] = z[:, k]
    
    # Clear the existing plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    ax1.set_title('Probability Density Distribution', fontsize=15)
    ax2.set_title('Vector Field and Electric Potential', fontsize=15)
    ax3.set_title(f"Cross-section of probability distribution at x = {x_desired:.2f} after {i} frames")
    
    
    if (Space_charge == True):
        norm_factor = np.sum(z * dx * dy)
        z_normalized = z * (e * num_electrons / norm_factor)
        total_charge = np.sum(z_normalized * dx * dy)
        rho = z_normalized
    
        # Update the electric potential and plot the new potential contour
        Vcolomb = colomb_potential(rho, x, y, dx, dy)
        Vcolomb_2D = Vcolomb.reshape(size_x, size_y).T
        max_value = vmax
        levels = np.linspace(0, max_value, 20)
        potential_contour = ax2.contourf(xx, yy, Vcolomb_2D, levels=levels, cmap= "magma")
        
        
        # Update colorbar for the electric potential plot
        if cbar2 is not None:
            cbar2.remove()
        cbar2 = fig.colorbar(potential_contour, ax=ax2)
        cbar2.set_label('Electric Potential (V)')
    else:  
        levels = np.linspace(0, 1, 20)    
        potential_contour = ax2.contourf(xx, yy, np.zeros(len(x)*len(y)).reshape(size_x, size_y).T, levels=levels, cmap= "magma")        
    ax2.set_xlabel(r"x ($a_0$)", fontsize=14)
    ax2.set_ylabel(r"y ($a_0$)", fontsize=14)
    
    if (Aharanov_Bohm == True):
        n = 6 
        sol = plt.Circle((solenoid_x0, solenoid_y0), R, color='blue', fill=True, alpha=0.5)
        #ax2.text(solenoid_y0, solenoid_x0 + R + 0.2, 'B field', fontsize=14, ha='center', va='bottom', color='white')
        ax2.add_artist(sol)
        ax2.quiver(yy.ravel()[::n], xx.ravel()[::n], A_y[::n].T, A_x[::n].T, color='white', alpha=0.6)
    
    
    z_normalized = z / np.sum(z)
    total_distribution = np.sum(z * dx * dy)
    # Update the probability density plot
    if i == 0:
        zmax= z_normalized.max()#choosing initial probability density to be fixed maximum of colorbar 
        
    scatter = ax1.scatter(xx, yy, c=z_normalized, cmap=color_map, vmin=0, vmax = zmax)
    #scatter = ax1.scatter(xx, yy, c=z_normalized, cmap=color_map, vmin=0, vmax=z_normalized.max())

    ax1.set_xlabel(r"x ($a_0$)", fontsize=14)
    ax1.set_ylabel(r"y ($a_0$)", fontsize=14)
    
    
    
    if (Two_slit == True):
        #plotting the double slits on ax1
        ax1.vlines(x01, y01, yf1, colors='white', linewidth=0.5, zorder=2)
        ax1.vlines(xf1, y01, yf1, colors='white', linewidth=0.5, zorder=2)
        ax1.vlines(x02, y02, yf2, colors='white', linewidth=0.5, zorder=2)
        ax1.vlines(xf2, y02, yf2, colors='white', linewidth=0.5, zorder=2)
        ax1.hlines(yf1, x01, xf1, colors='white', linewidth=0.5, zorder=2)
        ax1.hlines(y02, x01, xf1, colors='white', linewidth=0.5, zorder=2)
        ax1.vlines(x03, y03, yf3, colors='white', linewidth=0.5, zorder=2)
        ax1.vlines(xf3, y03, yf3, colors='white', linewidth=0.5, zorder=2)
        ax1.hlines(yf3, x03, xf3, colors='white', linewidth=0.5, zorder=2)
        ax1.hlines(y03, x03, xf3, colors='white', linewidth=0.5, zorder=2)
                
    # Update colorbar for the probability density plot
    cbar1 = fig.colorbar(scatter, cax=cax1)
    ticks = np.around(np.linspace(0, z.max(), 10), 3)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.ax.set_ylabel(r"$|\psi(y,t)|^2$", fontsize=14)
    ax1.axvline(x=x_desired, linestyle=':', color='red')

    # Plot the cross-section of the probability distribution at a specific x-coordinate
    
    max_prob_index = np.unravel_index(np.argmax(z, axis=None), z.shape)
    k_final = max_prob_index[1]
    final_z_slice = z_normalized[:, k_final]
    #ax3.set_title(f"Cross-section of probability distribution at x = {x[k_final]:.2f} after {i} frames")
    #ax3.plot(yy[:, k_final], final_z_slice)
    
    if i>140: #only taking cross sections for later frames when the wavefunction will have crossed the threshold
        k_final = np.argmin(np.abs(x - x_desired))
        final_z_slice = z_normalized[:, k_final]
        ax3.plot(y, final_z_slice)


   
    
    #ax1.axvline(x=x[k_final], linestyle=':', color='red')
    # Update the appearance of the cross-section plot
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_xlabel(r"y ($a_0$)")
    ax3.set_ylabel(r"$|\psi(y,t)|^2$")
    ax3.grid(True)
    print(i)

# Set the animation interval
interval = 0.001


# Create the animation object
anim = animation.FuncAnimation(fig, animate, nb_frame, interval=interval*1e+3, blit=False)
now = datetime.datetime.now()
filename = '2D_2slit_dx={0}_dt={1}_k={2}_{3}.gif'.format(dx, dt, kx0, now.strftime('%Y-%m-%d_%H-%M-%S'))

# Save the animation as a GIF
anim.save(filename, dpi=80, writer='imagemagick')

# Save the cross-section data to a file
with open("2_slit_dx={0}_dt={1}_k={2}_{3}.pkl".format(dx, dt, kx0, now.strftime('%Y-%m-%d_%H-%M-%S')), 'wb') as pickleFile:
    pickle.dump(coupe, pickleFile)
    pickleFile.close()

# Calculate the elapsed time for running the simulation
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
