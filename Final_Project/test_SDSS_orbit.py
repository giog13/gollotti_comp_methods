#Importing packages (not uploading this to Github until I understand how Claude.ai debugged my code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Galactic, Galactocentric
import astropy.units as u
import astropy.constants as const

#Plummer potential model
def plummer_potential(r, b = 4.12, M = 1.29e12):
    """
        This function represents the Plummer Potential, which is a potential energy equation that is similar to Kepler's potential except
        that it's also dependent on the half-mass radius of a given galaxy. The Plummer Potential is used as a model for simple 
        Keplerian orbits within a spheroid. For this project, I am specifically modeling the orbits of stars within the Milky Way. Since
        the star's mass is negligibly small compared to the whole mass of the Milky Way, we don't need to include it. This is a simple
        model, because I am assuming the Milky Way's rotation is static and the stars won't interact/collide with other stars.

        Parameters:
            r (float) = Distance of a star from the Milky Way's center (kpc)
            b (float) = The Milky Way's half-mass radius (default for Milky Way = 4.12 kpc - based on Lian et. al 2024)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            pot (float) = Potential energy of a star orbiting around the Milky Way (m^2/s^2 - energy per mass)
    
    """
    MW_mass = (M * u.solMass).to(u.kg)
    r = (r * u.kpc).to(u.m)
    b = (b * u.kpc).to(u.m)
    
    pot = - (const.G * MW_mass) / np.sqrt(np.square(r) + np.square(b))

    return pot

#Acceleration of the star (derivative of the circular velocity with the plummer potential)
def acceleration_plummer(pos, b = 4.12, M = 1.29e12):
    """
        This function calculates the gravitational acceleration using the Plummer potential. This derivative will be plugged into
        the circular velocity equation, since the circular velocity is dependent on the star's distance from the Milky Way's center
        and the absolute value of the Plummer potential's acceleration in the radial direction.
        
        Plummer acceleration: a = -GM * r / (r^2 + b^2)^(3/2)

        Parameters:
            pos (np.array) = Position vector [x, y, z] (kpc)
            b (float) = The Milky Way's half-mass radius (default for Milky Way = 4.12 kpc - based on Lian et. al 2024)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            acc (np.array) = Acceleration vector [a_x, a_y, a_z] (m/s^2)
    """
    # Convert to SI units
    MW_mass = (M * u.solMass).to(u.kg).value
    G_val = const.G.to(u.m**3 / (u.kg * u.s**2)).value #Suggested by Claude.ai
    
    pos_m = (pos * u.kpc).to(u.m).value  # Convert position to meters
    b_m = (b * u.kpc).to(u.m).value
    
    # Calculate distance from center
    r = np.sqrt(np.sum(np.square(pos_m)))
    
    # Plummer acceleration
    denominator = (np.square(r) + np.square(b_m))**(3/2)
    acc = -(G_val * MW_mass * pos_m) / denominator
    
    return acc

#Circular velocity method (simple model)
def circular_velocity(r, b = 4.12, M = 1.29e12):
    """
        TThis function returns the circular velocity of a star orbiting the Milky Way, assuming that the Milky Way's rotation is static and
        the star doesn't interact/collide with other stars. This is a SIMPLE model; stars don't actually orbit around the galaxy in a
        perfect circle, since it has velocity components in more than just the radial direction. This equation will test whether my
        leapfrog method works before I model more elliptical orbits.

        Parameters:
            r (float) = Star's distance from the Milky Way's center (kpc)
            b (float) = The Milky Way's half-mass radius (default for Milky Way = 4.12 kpc - based on Lian et. al 2024)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            v_circ (Quantity) = Star's circular velocity measurement (m/s)
    """
    #Converting into SI units for easy calculations
    MW_mass = (M * u.solMass).to(u.kg).value
    G_val = const.G.to(u.m**3 / (u.kg * u.s**2)).value
    r_m = (r * u.kpc).to(u.m).value
    b_m = (b * u.kpc).to(u.m).value
    
    # For circular orbit in Plummer potential:
    # v_circ = sqrt(GM * r^2 / (r^2 + b^2)^(3/2))
    v_circ = np.sqrt((G_val * MW_mass * np.square(r_m) / ((np.square(r_m) + np.square(b_m))**(3/2))))
    
    return v_circ * (u.m/u.s)

#Leapfrog method
def leapfrog_method(pos, vel, dt):
    """
         This function performs one integration step for the Leapfrog Method. I'm using the Leapfrog Method, since it conserves energy
        pretty well over long timescales. It's the preferred integrator for orbital simulations. 

        Parameters:
            pos (np.array) = Current position [x, y, z] (kpc)
            vel (np.array) = Current velocity [v_x, v_y, v_z] (m/s)
            dt (float) = Time step (s)

        Returns:
            new_pos (np.array) = New position [x, y, z] (kpc)
            new_vel (np.array) = New velocity [vx, vy, vz] (m/s)
    """
    # Step 1: Half-step velocity update
    acc = acceleration_plummer(pos)
    vel_half = vel + (0.5 * dt * acc)
    
    # Step 2: Full-step position update (convert velocity to kpc/s for position update)
    vel_half_kpc = (vel_half * (u.m/u.s)).to(u.kpc/u.s).value
    new_pos = pos + (dt * vel_half_kpc)
    
    # Step 3: Calculate acceleration at new position
    new_acc = acceleration_plummer(new_pos)
    
    # Step 4: Complete velocity update
    new_vel = vel_half + (0.5 * dt * new_acc)
    
    return new_pos, new_vel

#Simulating the orbit
def simulate_orbit(r_0, v_0, dt, t_max, b = 4.12, M = 1.29e12):
    """
        This function simulates the orbit of a star around the Milky Way through integrating with the leapfrog method. The number of time
        step (n_steps) is dependent on the given time step (dt) and maximum simulation time (t_max). Every time I iterate the leapfrog
        method, the new times, positions, and velocities will be added to a np.array (times, positions, velocities), which are all
        unitless. This function will return the time, position, and velocity arrays, so they can be easily plotted or animated.

        Parameters:
            r_0 (float) = Initial radial distance away from the center of the Milky Way (kpc)
            v_0 (astropy Quantity) = Initial velocity component (m/s)
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)
            b (float) = The Milky Way's half-mass radius (default for Milky Way = 4.12 kpc - based on Lian et. al 2024)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)
            
        Returns:
            times (np.array) = Array of time steps (Myr)
            positions (np.array) = Array of positions with coordinates [x, y, z] (kpc)
            velocities (np.array) = Array of velocities [v_x, v_y, v_z] (kpc/Myr)
    """
    # Calculate the total number of steps
    n_steps = int(t_max / dt)
    
    # Initialize arrays
    times = np.zeros(n_steps + 1)
    positions = np.zeros((n_steps + 1, 3))  # [x, y, z] in kpc
    velocities = np.zeros((n_steps + 1, 3))  # [v_x, v_y, v_z] in kpc/Myr
    
    # Set initial conditions
    positions[0] = np.array([r_0, 0.0, 0.0])
    
    # Convert initial velocity to m/s if it has units (suggested by Claude.ai)
    if hasattr(v_0, 'unit'):
        v_0_ms = v_0.to(u.m/u.s).value
    else:
        v_0_ms = v_0
    
    # For circular orbit: velocity should be tangential (in y-direction, since position is in x-direction)
    velocities[0] = np.array([0.0, (v_0_ms * (u.m/u.s)).to(u.kpc/u.Myr).value, 0.0])
    
    # Convert dt from Myr to seconds for leapfrog
    dt_sec = (dt * u.Myr).to(u.s).value
    
    # Integration loop
    for i in range(n_steps):
        times[i + 1] = times[i] + dt
        
        # Get velocity in m/s for leapfrog
        vel_ms = (velocities[i] * (u.kpc/u.Myr)).to(u.m/u.s).value
        
        # Perform leapfrog step
        new_pos, new_vel = leapfrog_method(positions[i], vel_ms, dt_sec)
        
        # Store new values
        positions[i + 1] = new_pos
        velocities[i + 1] = (new_vel * (u.m/u.s)).to(u.kpc/u.Myr).value
    
    return times, positions, velocities

#Plotting the orbital model of a star
def plot_simulation(r_0, v_0, dt, t_max):

    """
        This function simulates the orbit of a star around the Milky Way through integrating with the leapfrog method. The number of time
        step (n_steps) is dependent on the given time step (dt) and maximum simulation time (t_max). Every time I iterate the leapfrog
        method, the new times, positions, and velocities will be added to a np.array (times, positions, velocities), which are all
        unitless. This function will return the time, position, and velocity arrays, so they can be easily plotted or animated.

        Parameters:
            r_0 (float) = Initial radial distance away from the center of the Milky Way (kpc)
            v_0 (float) = Initial velocity component (m/s) - will need to convert to kpc/Myr
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)
            b (float) = The Milky Way's half-mass radius (default for Milky Way = 4.12 kpc - based on Lian et. al 2024)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Variables:
            n_steps (int) = Number of time steps for the simulation (unitless)
            times (np.array) = Array of time steps (Myr)
            positions (np.array) = Array of positions with coordinates [x, y, z] (kpc)
                                   Currently I only care about the radial component, so x will be the only non-zero value
            velocities (np.array) = Array of velocities [v_x, v_y, v_z] (kpc/Myr)
                                    Currently I only care about the radial velocity component, so v_x will be the only non-zero value
            
        Returns:
            times (np.array) = Array of time steps (Myr)
            positions (np.array) = Array of positions with coordinates [x, y, z] (kpc)
            velocities (np.array) = Array of velocities [v_x, v_y, v_z] (kpc/Myr)
            
    """
    
    times, positions, velocities = simulate_orbit(r_0, v_0, dt, t_max)
    
    fig = plt.figure(figsize = (8, 6))
    
    # Orbital trajectory in the xy-plane
    plt.title('Orbital trajectory in xy-plane', fontsize = 15)
    plt.xlabel('x (kpc)', fontsize = 15)
    plt.ylabel('y (kpc)', fontsize = 15)
    
    plt.plot(positions[:, 0], positions[:, 1], color = 'blue', label = 'Stellar Orbit', linewidth=1.5)
    plt.scatter(0, 0, color = 'orangered', s = 100, label = 'MW Center')
    plt.scatter(positions[0, 0], positions[0, 1], color = 'green', s = 80, label = 'Start')
    
    plt.legend(loc = 'upper right', fontsize = 12)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print some diagnostics
    r_values = np.sqrt(np.sum(np.square(positions), axis=1))
    print(f"Initial distance: {r_values[0]:.3f} kpc")
    print(f"Min distance: {r_values.min():.3f} kpc")
    print(f"Max distance: {r_values.max():.3f} kpc")

if __name__ == '__main__':
    # Stellar parameters
    r_0 = 8.122422#kpc
    
    # For a CIRCULAR orbit, we need tangential velocity equal to the circular velocity
    # Calculate circular velocity at this radius
    v_circ = circular_velocity(r_0)
    print(f"Circular velocity at {r_0:.2f} kpc: {v_circ.to(u.kpc/u.Myr):.2f}")
    
    # Use circular velocity for a nice circular orbit
    v_0 = v_circ
    
    plot_simulation(r_0, v_0, dt = 0.1, t_max = 500)