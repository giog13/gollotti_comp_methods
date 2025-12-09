#Importing packages (not uploading this to Github until I understand how Claude.ai debugged my code)
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Galactic, Galactocentric
from astropy.coordinates import Angle
import astropy.units as u
import astropy.constants as const

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#Reading crossmatched Gaia/SDSS file and isolating the Nitrogen-rich stars
def reading_file(): #Need to write commentary

    filename = 'ASPCAP_Gaia_00311_Crossmatch.csv'

    df = pd.read_csv(filename)

    df_cross_gaia = df[['sdss_id', 'gaia_dr3_source_id', 'healpix', 'catalogid31', 'ra_1', 'dec_1', 'teff', 'logg', 'bp_rp', 'fe_h', 'n_h',
                    'o_h','parallax', 'l_2', 'b_2', 'pm', 'pmra_2', 'pmdec', 'dr2_radial_velocity']]

    #Deleting rows with np.nan values
    df_cross_gaia = df_cross_gaia.dropna()
    df_cross_gaia = df_cross_gaia.reset_index(drop=True) #Resetting index values
    
    #Deleting rows where the Gaia DR3 ID == -1
    gaia_id = np.array(df_cross_gaia['gaia_dr3_source_id'])
    bad_gaia = gaia_id == -1
    df_cross_gaia = df_cross_gaia[~bad_gaia]
    
    df_cross_gaia = df_cross_gaia.reset_index(drop=True)

    #Isolating stars with reliable temperatures
    teff = np.array(df_cross_gaia['teff'])
    teff_mask = teff > 4500
    
    df_cross_gaia = df_cross_gaia[teff_mask]
    df_cross_gaia = df_cross_gaia.reset_index(drop=True)

    #Calculating nitrogen abundances
    n_abundances = []
    
    for i in range(len(df_cross_gaia['n_h'])):
        n_fe_val = df_cross_gaia['n_h'][i] - df_cross_gaia['fe_h'][i]
        n_abundances.append(n_fe_val)
    
    n_fe_vals = np.array(n_abundances)

    #Isolating nitrogen-rich stars
    n_abund_mask = n_fe_vals >= 0.5
    
    df_cross_gaia = df_cross_gaia[n_abund_mask]
    df_cross_gaia = df_cross_gaia.reset_index(drop=True)

    return df_cross_gaia
    

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

#Converting to Galactocentric coordinates
def galactic_coords(ra, dec, dist, pm_ra, pm_dec, rv): #Suggested by Claude.ai - Need to write more detailed commentary
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

def simulate_galactic_orbit(r_0, v_0, dt, t_max): #Need to write more detailed commentary

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

## Making an animation ##

#Need to write commentary
def animate_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv, dt = 0.1, t_max = 1000, disk_radius = 15.0, disk_height = 0.3,
                          speed = 10, trail_length = 500, save_animation = False, gif_name = 'Nrich_stellar_orbits.gif'):

    # Store all star data
    all_positions = []
    all_velocities = []
    all_colors = []

    for i in range(len(ra)):
        print(f"\nProcessing Star {i + 1}...")
        print("="*60)
        
        # Convert to Galactocentric coordinates
        r_0, v_0 = galactic_coords(ra[i], dec[i], dist[i], pm_ra[i], pm_dec[i], rv[i])
        
        # Simulate the orbit
        times, positions, velocities = simulate_galactic_orbit(r_0, v_0, dt, t_max)
        
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_colors.append((random.random(), random.random(), random.random()))

    #Plotting all the orbits on one plot
    fig = plt.figure(figsize = (14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    #Plotting the Milky Way's disk
    # 2D disk (circle) for xy-plane plot
    theta = np.linspace(0, 2 * np.pi, 100)
    disk_x = disk_radius * np.cos(theta)
    disk_y = disk_radius * np.sin(theta)
    ax2.plot(disk_x, disk_y, color = 'lightgray', linestyle = '--', linewidth = 2, 
            alpha = 0.5, label = 'MW Disk Edge', zorder = 1)
    ax2.fill(disk_x, disk_y, color = 'lightgray', alpha = 0.2, zorder = 0)
    
    # 3D disk for xyz-plane plot
    theta_cyl = np.linspace(0, 2 * np.pi, 50)
    z_cyl = np.linspace(-disk_height, disk_height, 2)
    Theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_cyl = disk_radius * np.cos(Theta_cyl)
    Y_cyl = disk_radius * np.sin(Theta_cyl)
        
    # Plotting 3D cylinder surface
    ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, color = 'lightgray', alpha = 0.2, zorder = 0)
        
    # Plot cylinder edges
    circle_x = disk_radius * np.cos(theta)
    circle_y = disk_radius * np.sin(theta)
    ax1.plot(circle_x, circle_y, disk_height, color = 'gray', linestyle = '--', 
            linewidth = 1.5, alpha = 0.5, zorder = 1)
    ax1.plot(circle_x, circle_y, -disk_height, color = 'gray', linestyle = '--', 
            linewidth = 1.5, alpha = 0.5, zorder = 1)

    #Plotting the Milky Way center
    ax1.scatter(0, 0, 0, color = 'black', s = 100, zorder = 2, label='MW Center')
    ax2.scatter(x = 0, y = 0, color = 'black', s = 100, zorder = 2, label = 'MW Center')

    # Plot full static orbits first
    for i in range(len(ra)):
        # Plot complete orbit path (static, faded)
        ax1.plot(all_positions[i][:, 0], all_positions[i][:, 1], all_positions[i][:, 2],
                color = all_colors[i], linewidth = 1, alpha = 0.3, linestyle = '--',
                zorder = 1, label = f'Star {i+1} Full Orbit')
        ax2.plot(all_positions[i][:, 0], all_positions[i][:, 1],
                color = all_colors[i], linewidth = 1, alpha = 0.3, linestyle = '--',
                zorder = 1, label = f'Star {i+1} Full Orbit')

    # Initialize plot elements for each star
    points_3d = []
    trails_3d = []
    
    points_2d = []
    trails_2d = []

    for i in range(len(ra)):
        
        # 3D elements
        point_3d, = ax1.plot([], [], [], marker = '*', color = all_colors[i], 
                            markersize = 12, zorder = 4, markeredgecolor = 'black', 
                            markeredgewidth = 0.5, label = f'Star {i+1}')
        trail_3d, = ax1.plot([], [], [], linestyle = '-', color = all_colors[i], 
                            linewidth = 2, alpha = 0.6, zorder = 3)
        points_3d.append(point_3d)
        trails_3d.append(trail_3d)
        
        # 2D elements
        point_2d, = ax2.plot([], [], marker = '*', color = all_colors[i], 
                            markersize = 12, zorder = 4, markeredgecolor = 'black',
                            markeredgewidth = 0.5, label = f'Star {i+1}')
        trail_2d, = ax2.plot([], [], linestyle = '-', color = all_colors[i], 
                            linewidth = 2, alpha = 0.6, zorder = 3)
        points_2d.append(point_2d)
        trails_2d.append(trail_2d)

        # Time text (suggested by Claude.ai)
        time_text_3d = ax1.text2D(0.02, 0.95, '', transform = ax1.transAxes, 
                              fontsize = 12, fontweight = 'bold',
                              bbox=dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.8))
        time_text_2d = ax2.text(0.02, 0.98, '', transform = ax2.transAxes,
                           fontsize = 12, fontweight = 'bold', verticalalignment = 'top',
                           bbox=dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.8))

        
        #Formatting axes
        # 3D orbit
        ax1.set_xlabel('X (kpc)', fontsize=15)
        ax1.set_ylabel('Y (kpc)', fontsize=15)
        ax1.set_zlabel('Z (kpc)', fontsize=15)
        ax1.set_title('3D Galactocentric Orbit', fontsize = 15)
        ax1.legend(loc = 'upper right', fontsize = 9)

        max_range = disk_radius * 1.1
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range/2, max_range/2])
    
        # xy-plane projection
        ax2.set_xlabel('X (kpc)', fontsize = 15)
        ax2.set_ylabel('Y (kpc)', fontsize = 15)
        ax2.set_title('Orbit in XY-Plane', fontsize = 15)
        ax2.axis('equal')
        ax2.legend(loc = 'upper right', fontsize = 9)

        ax2.set_xlim([-max_range, max_range])
        ax2.set_ylim([-max_range, max_range])


    ## Claude.ai suggested to include the init() and animate() methods inside this function
    def init(): #Need to write commentary
        """ Initialize animation """
        
        for i in range(len(ra)):
                
            #3D arrays
            points_3d[i].set_data(np.array([]), np.array([]))
            points_3d[i].set_3d_properties(np.array([]))
            trails_3d[i].set_data(np.array([]), np.array([]))
            trails_3d[i].set_3d_properties(np.array([]))
    
            #2D arrays
            points_2d[i].set_data(np.array([]), np.array([]))
            trails_2d[i].set_data(np.array([]), np.array([]))
                
        time_text_3d.set_text('')
        time_text_2d.set_text('')
    
        return points_3d + trails_3d + points_2d + trails_2d +[time_text_3d, time_text_2d]

    def animate(frame): #Created with Claude.ai - need to write commentary
        """Update animation for each frame"""
    
        idx = frame * speed
        if idx >= len(all_positions[0]):
            idx = len(all_positions[0]) - 1
            
        # Calculate time
        current_time = idx * dt
            
        for i in range(len(ra)):
            # Get current position
            pos = np.array(all_positions[i][idx])
                
            # Get trail positions
            trail_start = max(0, idx - trail_length)
            trail_pos = all_positions[i][trail_start:idx+1]
                
            # Update 3D plot
            points_3d[i].set_data(np.array([pos[0]]), np.array([pos[1]]))
            points_3d[i].set_3d_properties(np.array([pos[2]]))
                
            if len(trail_pos) > 1:
                trail_array = np.array(trail_pos) if isinstance(trail_pos, list) else trail_pos
                trails_3d[i].set_data(trail_array[:, 0], trail_array[:, 1])
                trails_3d[i].set_3d_properties(trail_array[:, 2])
                
            # Update 2D plot
            points_2d[i].set_data(np.array([pos[0]]), np.array([pos[1]]))
                
            if len(trail_pos) > 1:
                trail_array = np.array(trail_pos) if isinstance(trail_pos, list) else trail_pos
                trails_2d[i].set_data(trail_array[:, 0], trail_array[:, 1])
            
        # Update time display
        time_text_3d.set_text(f'Time: {current_time:.1f} Myr')
        time_text_2d.set_text(f'Time: {current_time:.1f} Myr')
            
        return points_3d + trails_3d + points_2d + trails_2d + [time_text_3d, time_text_2d]
    

    #Calculate number of frames
    n_frames = len(all_positions[0]) // speed #Total number of frames must be an integer value
    
    # Create animation
    print(f"\nCreating animation with {n_frames} frames...")
    
    anim = FuncAnimation(fig, animate, init_func = init, frames = n_frames,
                        interval = 50, blit = True, repeat = True)    

    # Save animation if requested
    if save_animation == True: #This method isn't working right now
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer = 'pillow', fps = 10, dpi = 100)
        print("Animation saved!")

    plt.tight_layout()
    plt.show()

    return anim

if __name__ == '__main__': #Next task, load in Gaia/SDSS files

    #Write argparse arguments

    #Obtaining dataframe of the Gaia/SDSS file
    df_cross_gaia = reading_file()

    #Write a random number generator, so it picks random indices based on the number of stars the user wants to simulate
    
    # Stellar parameters

    #Initializing arrays
    ra = []
    dec = []
    dist = []
    pm_ra = []
    pm_dec = []
    rv = []
    random_star = [] #Array of random indices generated by random number generator

    #Appending  values to the arrays based on the number of stars the user wants to plot

    #for i in range(arg.stellar_number):
        #random_index = random.randint(0, len(df_cross_gaia))
        #random_star.append(random_index)

        #Make sure random index wasn't already picked for a previous star  (need to write a while statement)    

        #Appending values to arrays
        #ra.append(df_cross_gaia['ra_1'][random_index])
        #dec.append(df_cross_gaia['dec_1'][random_index])
        #dist.append(df_cross_gaia['dist_from_Earth(kpc)'][random_index])
        #pm_ra.append(df_cross_gaia['pmra_2'][random_index])
        #pm_dec.append(df_cross_gaia['pmdec'][random_index])
        #rv.append(df_cross_gaia['dr2_radial_velocity'][random_index])
    
    animate_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv)