#Importing packages (not uploading this to Github until I understand how Claude.ai debugged my code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Galactic, Galactocentric
import astropy.units as u
import astropy.constants as const

#Plummer potential model (works for GCs and dwarf galaxies, NOT spirals)
def plummer_potential(r, b = 15.0, M = 1.29e12):  #Claude.ai suggests to use 15 kpc instead
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
def acceleration_plummer(pos, b = 15.0, M = 1.29e12):
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
def circular_velocity(r, b = 15.0, M = 1.29e12):
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
def simulate_orbit(r_0, v_rad, pm, parallax, dt, t_max):
    """
        This function simulates the orbit of a star around the Milky Way through integrating with the leapfrog method. The number of time
        step (n_steps) is dependent on the given time step (dt) and maximum simulation time (t_max). Every time I iterate the leapfrog
        method, the new times, positions, and velocities will be added to a np.array (times, positions, velocities), which are all
        unitless. This function will return the time, position, and velocity arrays, so they can be easily plotted or animated.

        Parameters:
            r_0 (float) = Initial radial distance away from the center of the Milky Way (kpc)
            v_rad (float) = Velocity in the radial direction (km/s - convert to kpc/Myr)
            pm (float) = Velocity in the tangential direction, which is represented by proper motion (mas/yr - convert to kpc/Myr)
            parallax (float) = Stellar parallax when observed from Earth (microarcsec)
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)
            
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

    #Converting proper motion units to kpc/Myr
    proper_motion = pm * (u.mas/u.yr)
    p = (parallax * u.mas) #Giving units of milliarcsec to parallax
    distance = p.to(u.kpc, equivalencies = u.parallax())
    v_tan = (proper_motion * distance).to(u.kpc/u.Myr, equivalencies = u.dimensionless_angles())
    print(f"Tangential velocity: {v_tan:.4e}")

    #Converting radial velocity back to kpc/Myr
    v_r = (v_rad * (u.km/u.s)).to(u.kpc/u.Myr)

    #Adding velocities to an array
    velocities[0] = np.array([v_r.value, v_tan.value, 0.0]) #[x = radial velocity, y = tangential velocity (proper motion)]

    #Print statements for debugging
    print(f"Initial position: {positions[0]} kpc")
    print(f"Initial velocity: {velocities[0]} kpc/Myr")
    print(f"Radial velocity: {v_r:.4f} kpc/Myr")
    print(f"Proper motion: {v_tan:.4f} kpc/Myr\n")
    
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

def generate_random_rgb_color():
    """Generates a random RGB color tuple."""
    return (random.random(), random.random(), random.random())

def plot_simulation(r0, v_rad, v_tan, p, dt = 0.1, t_max = 500):

    """
        This function simulates the orbit of a star around the Milky Way through integrating with the leapfrog method. The number of time
        step (n_steps) is dependent on the given time step (dt) and maximum simulation time (t_max). Every time I iterate the leapfrog
        method, the new times, positions, and velocities will be added to a np.array (times, positions, velocities), which are all
        unitless. This function will return the time, position, and velocity arrays, so they can be easily plotted or animated.

        Parameters:
            r0 (list) = List of initial radial distances for all the stars in the list (kpc)
            v_rad (list) = List of  radial velocity components (km/s) - will be converted to kpc/Myr
            v_tan (list) = List of  tangential velocity components, representing the proper motion (mas/yr) - will be converted to kpc/Myr
            parallax (list) = List of parallaxes for each star (milliarcsec)
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)

        Variables:
            n_steps (int) = Number of time steps for the simulation (unitless)
            times (np.array) = Array of time steps (Myr)
            positions (np.array) = Array of positions with coordinates [x, y, z] (kpc)
                                   Currently I only care about the radial component, so x will be the only non-zero value
            velocities (np.array) = Array of velocities [v_x, v_y, v_z] (kpc/Myr)
                                    Currently I only care about the radial velocity component, so v_x will be the only non-zero value
            
        Returns:
            Plot with all the circular stellar orbits around the center of the Milky Way
            
    """

    fig = plt.figure(figsize = (8, 6))

    # Orbital trajectory in the xy-plane
    plt.title('Orbital trajectory in xy-plane', fontsize = 15)
    plt.xlabel('x (kpc)', fontsize = 15)
    plt.ylabel('y (kpc)', fontsize = 15)

    plt.scatter(0, 0, color = 'orangered', s = 100, label = 'MW Center')
    
    for i in range(len(r0)):
    
        times, positions, velocities = simulate_orbit(r0[i], v_rad[i], v_tan[i], p[i], dt, t_max)

        rand_color = generate_random_rgb_color()
    
        plt.plot(positions[:, 0], positions[:, 1], color = rand_color, linewidth=1.5) #Orbital model
        #plt.plot(positions[0, 0], positions[0, 1], marker = '*', color = 'gold', size = 30) #Start position

        # Print some diagnostics
        r_values = np.sqrt(np.sum(np.square(positions), axis=1))
        print(f"Info about Star {i + 1}\n")
        print(f"Initial distance: {r_values[0]:.3f} kpc")
        print(f"Initial radial velocity: {v_rad[0]} kpc/Myr")
        print(f"Initial tangential velocity: {v_tan[0]} kpc/Myr")
        print(f"Min distance: {r_values.min():.3f} kpc")
        print(f"Max distance: {r_values.max():.3f} kpc\n")
    
    plt.legend(loc = 'upper right', fontsize = 12)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

#Converting to Galactocentric coordinates
def galactic_coords(ra, dec, dist, pm_ra, pm_dec, rv): #Suggested by Claude.ai
    """
        This function converts observation coordinates to Galactocentric coordinates, so I can simulate stellar orbits in 3 dimensions.

        Parameters:
            ra (float) = RA (degrees)
            dec (float) = Declination (degrees)
            dist (float) = Distance from Earth (kpc)
            pm_ra (float) = Proper motion in RA direction (mas/yr)
            pm_dec (float) = Proper motion in Dec direction (mas/yr)
            rv (float) = Radial velocity (km/s)

        Returns:
            position (np.array): Galactocentric position [X, Y, Z] (kpc)
            velocity (np.array): Galactocentric velocity [vX, vY, vZ] (km/s)
    
    """

    #Create SkyCoords for a star based on observational data (as seen from Earth)
    star = SkyCoord(
            ra = ra * u.deg,
            dec = dec * u.deg,
            distance = dist * u.kpc,
            pm_ra_cosdec = pm_ra * (u.mas/u.yr), #Proper motion * cos(dec)
            pm_dec = pm_dec * (u.mas/u.yr),
            radial_velocity = rv * (u.km/u.s),
            frame = 'icrs'
            )

    #Transforms to the frame of the Milky Way's center
    gal_centric = star.transform_to(Galactocentric())

    #Extracting positional coordinates
    position = np.array([
            gal_centric.x.to(u.kpc).value,
            gal_centric.y.to(u.kpc).value,
            gal_centric.z.to(u.kpc).value
            ])

    #Extract velocities
    velocity = np.array([
            gal_centric.v_x.to(u.km/u.s).value,
            gal_centric.v_y.to(u.km/u.s).value,
            gal_centric.v_z.to(u.km/u.s).value
            ])

    return position, velocity

def simulate_galactic_orbit(r_0, v_0, dt, t_max):

    """
        This function simulates a stellar orbit in Galactocentric coordinates instead of observational coordinates.

        Parameters:
            r_0 (np.array) = Array of initial Galactocentric positions [X, Y, Z] (kpc)
            v_0 (np.array) = Array of initial Galactocentric velocities [vX, vY, vZ] (km/s)
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)

        Returns:
            times (np.array): Array of time steps (Myr)
            positions (np.array): Array of positions [X, Y, Z] (kpc)
            velocities (np.array): Array of velocities [vX, vY, vZ] (kpc/Myr)
    
    """

    # Calculate the total number of steps
    n_steps = int(t_max / dt)
    
    # Initialize arrays
    times = np.zeros(n_steps + 1)
    positions = np.zeros((n_steps + 1, 3))  # [x, y, z] in kpc
    velocities = np.zeros((n_steps + 1, 3))  # [v_x, v_y, v_z] in kpc/Myr
    
    # Set initial conditions
    positions[0] = r_0

    #Converting radial velocity back to kpc/Myr
    velocities[0] = ((v_0 * (u.km/u.s)).to(u.kpc/u.Myr)).value

    #Print initial conditions
    dist_0 = np.sqrt(np.sum(np.square(r_0)))
    v_tot = np.sqrt(np.sum(np.square(v_0)))

    print(f"Initial Galactocentric position: [{r_0[0]:.3f}, {r_0[1]:.3f}, {r_0[2]:.3f}] kpc")
    print(f"Initial distance from Galactic center: {dist_0:.3f} kpc")
    print(f"Initial velocity: [{v_0[0]:.2f}, {v_0[1]:.2f}, {v_0[2]:.2f}] km/s")
    print(f"Total velocity magnitude: {v_tot:.2f} km/s")

    #Calculating circular velocity
    v_circ = (circular_velocity(dist_0).to(u.km/u.s)).value
    print(f"Circular velocity: {v_circ:.2f} km/s\n")
    
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

def plot_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv, dt = 0.1, t_max = 1000):

    """
        This function converts observational coordinates to Galactocentric and also plots the orbits in both 2D and 3D.

        Parameters:
            ra (float) = RA (degrees)
            dec (float) = Declination (degrees)
            dist (float) = Distance from Earth (kpc)
            pm_ra (float) = Proper motion in RA direction (mas/yr)
            pm_dec (float) = Proper motion in Dec direction (mas/yr)
            rv (float) = Radial velocity (km/s)
            dt (float) = Time step (Myr)
            t_max (float) = Total simulation time (Myr)
        
    """

    print("="*60) #Print functions suggested by Claude.ai for debugging purposes
    print("CONVERTING TO GALACTOCENTRIC COORDINATES")
    print("="*60)
    print(f"Input observational data:")
    print(f"  RA: {ra:.6f} deg")
    print(f"  Dec: {dec:.6f} deg")
    print(f"  Distance from Earth: {dist:.6f} kpc")
    print(f"  PM_RA*cos(dec): {pm_ra:.6f} mas/yr")
    print(f"  PM_Dec: {pm_dec:.6f} mas/yr")
    print(f"  Radial velocity: {rv:.6f} km/s\n")

    #Convert to Galactocentric coordinates
    r_0, v_0 = galactic_coords(ra, dec, dist, pm_ra, pm_dec, rv)

    #Simulate the orbit
    print("="*60)
    print("SIMULATING ORBIT")
    print("="*60)

    times, positions, velocities = simulate_galactic_orbit(r_0, v_0, dt, t_max)

    #Determine orbital statistics
    r_vals = np.sqrt(np.sum(np.square(positions), axis = 1))

    print("="*60)
    print("ORBITAL STATISTICS")
    print("="*60)
    print(f"Initial distance: {r_vals[0]:.3f} kpc")
    print(f"Min distance (periapsis): {r_vals.min():.3f} kpc")
    print(f"Max distance (apoapsis): {r_vals.max():.3f} kpc")
    print(f"Final distance: {r_vals[-1]:.3f} kpc")
    print(f"Eccentricity (approx): {(r_vals.max() - r_vals.min())/(r_vals.max() + r_vals.min()):.3f}\n")

    #Creating 3D plots
    fig = plt.figure(figsize = (14, 6))

    rand_color = generate_random_rgb_color()

    # 3D orbit
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color = rand_color, linewidth = 1.5, alpha = 0.7, label = 'Stellar Orbit', zorder = 3)
    ax1.scatter(0, 0, 0, color = 'orangered', s = 100, label='MW Center', zorder = 1)
    #ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                #color='green', s=80, label='Start')
    ax1.set_xlabel('X (kpc)', fontsize=15)
    ax1.set_ylabel('Y (kpc)', fontsize=15)
    ax1.set_zlabel('Z (kpc)', fontsize=15)
    ax1.set_title('3D Galactocentric Orbit', fontsize = 15)
    ax1.legend(loc = 'upper right', fontsize = 15)

    # xy-plane projection
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 1], color = rand_color, linewidth = 1.5, alpha = 0.7, label = 'Stellar Orbit')
    ax2.scatter(x = 0, y = 0, color = 'orangered', s = 100, label = 'MW Center')
    #ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=80, label='Start')
    ax2.set_xlabel('X (kpc)', fontsize = 15)
    ax2.set_ylabel('Y (kpc)', fontsize = 15)
    ax2.set_title('Orbit in XY-Plane', fontsize = 15)
    ax2.axis('equal')
    ax2.legend(loc = 'upper right', fontsize = 15)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__': #Next task, create animation and load in Gaia/SDSS files
    # Stellar parameters
    ra = [39.364098]
    dec = [5.292768]
    dist = [0.660446]
    pm_ra = [-8.348716]
    pm_dec = [-5.955241]
    rv = [-21.060333]
    
   plot_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv)