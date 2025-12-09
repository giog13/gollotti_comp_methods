#Importing packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

from astropy.coordinates import SkyCoord, Galactic, Galactocentric
from astropy.coordinates import Angle
import astropy.units as u
import astropy.constants as const

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#Calculating stellar distances from Earth based on the given parallax
def distance_from_Earth(parallax_list):

    """
        This function returns a list of stellar distances from Earth in kpc, where the distances are the inverse of a specific star's
        parallax. Since Gaia measures parallax with units of milliarcseconds, the distance will initially have units in parsecs. In order to
        obtain units in kiloparsecs, we must divide the distance by 1000.

        Parameters:
            parallax_list (list) = List of parallax values (milliarcsec)

        Returns:
            dist (list) = List of stellar distances from Earth (kiloparsec)
    """

    dist = []
    
    for p in parallax_list:
        d = (1 / (p * 1e-3)) / 1000

        dist.append(d) #Adds distance measurements to a list

    return dist

#Reading crossmatched Gaia/SDSS file and isolating the Nitrogen-rich stars
def reading_file(): #Need to write commentary

    """
        This function reads the crossmatched Gaia/SDSS file from my Github repository and, after some manipulation, returns the file as a
        usable DataFrame (df_cross_gaia). For context, I cross-matched the Gaia DR3 Nearby Stars catalogue with the SDSS ASPCAP file from the
        Milky Way Mapper, since the ASPCAP file includes T_eff, metallicity, and stellar abundance parameters. I only cared about a few columns
        such as the Gaia DR3 source ID (gaia_dr3_source_id), effective temperature (teff), Nitrogen/Hydrogen ratio (n_h), Iron/Hydrogen ratio 
        (fe_h), radial velocities (dr2_radial_velocity), proper motion (pm, pm_ra2, pm_dec), and parallax. I also deleted all the rows with
        np.nan values, or source IDs that equate to -1. 

        All the abundances from the ASPCAP file are calculated with the APOGEE pipeline, which makes reliable estimates for stars with
        temperatures > 4500 K. As a result, I isolated stars with reliable temperatures, and then looked specifically at the stellar abundances.
        Since I want to simulate the motion of stars that originated from globular clusters, I'm specifically looking for stars with a high 
        nitrogen abundance (N/Fe). Therefore, I isolated the table further by only analyzing stars with N/Fe ratios >= 0.5. I also wanted the
        stars to have reliable distances, so I added a column that represents stellar distances from Earth (calculated by distance_from_Earth()).
        I then removed all rows with the distances less than 0. Only then do I return the DataFrame for use.
    
        Returns:
            df_cross_gaia (DataFrame) = Table of crossmatched Gaia/SDSS values
        
    """

    #Crossmatched Gaia/SDSS file
    filename = 'ASPCAP_Gaia_00311_Crossmatch.csv'

    df = pd.read_csv(filename)

    #Columns I want to keep
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
    n_abund_mask = n_fe_vals >= 0.5 #0.5 is the accepted value for when a star is considered "Nitrogen-rich"
    
    df_cross_gaia = df_cross_gaia[n_abund_mask]
    df_cross_gaia = df_cross_gaia.reset_index(drop=True)

    #Calculating stellar distances from Earth and adding them to the DataFrame
    dist_vals = distance_from_Earth(df_cross_gaia['parallax'])
    df_cross_gaia['dist_from_Earth(kpc)'] = dist_vals #Adds distance values to the dataframe

    #Mask out distance values less than 0
    good_dist = np.array(dist_vals) >= 0
    df_cross_gaia = df_cross_gaia[good_dist]
    df_cross_gaia = df_cross_gaia.reset_index(drop=True)

    return df_cross_gaia

    
def plummer_potential(r, b = 15.0, M = 1.29e12):  #Claude.ai suggests to use 15 kpc instead
    
    """
        This function represents the Plummer Potential. It's a potential energy equation that is similar to Kepler's potential, except
        it's also dependent on the half-mass radius of a given galaxy. The Plummer Potential is used as a model for simple 
        Keplerian orbits within a spheroid. For this project, I am specifically modeling the orbits of stars within the Milky Way. Since
        the star's mass is negligibly small compared to the whole mass of the Milky Way, we don't need to include it. This is a simple
        model; I am assuming the Milky Way's rotation is static, and the stars won't transfer energy each other.

        For context, the Plummer Potential is an accepted simple gravitational model for globular clusters (GC) and dwarf galaxies.
        Unfortunately, it is NOT the simple model for spiral galaxies. Typically the Plummer constant would represent the half-mass radius of
        a GC or dwarf galaxy, but for spirals such as the Milky Way, the half-mass radius doesn't take the dark matter halo into account.
        Initially I implemented b = 4.12 kpc, since that was the half-mass radius, since 4.12 kpc was the recorded half-mass radius by Lian et.
        al 2024. Unfortunately the stars were slingshot around the center of the Milky Way in a crazy rosette pattern. Claude.ai suggested to
        use b = 15.0 kpc as the Milky Way's Plummer constant instead.

        Parameters:
            r (float) = Distance of a star from the Milky Way's center (kpc)
            b (float) = Plummer constant as suggested by Claude.ai (default = 15.0 kpc)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            pot (float) = Potential energy of a star orbiting around the Milky Way (m^2/s^2 - energy per mass)
    
    """

    #Converting all the units to something that will cancel out in the other functions
    MW_mass = (M * u.solMass).to(u.kg)
    r = (r * u.kpc).to(u.m)
    b = (b * u.kpc).to(u.m)
    
    pot = - (const.G * MW_mass) / np.sqrt(np.square(r) + np.square(b))

    return pot

#Acceleration of the star (derivative of the circular velocity with the plummer potential)
def acceleration_plummer(pos, b = 15.0, M = 1.29e12):
    """
        This function calculates the gravitational acceleration of the Plummer potential, which is the derivate of the potential equation. This 
        derivative will be plugged into the circular velocity equation, since the circular velocity is dependent on the star's distance from the 
        Milky Way's center and the absolute value of the Plummer potential's acceleration in the radial (x) direction.
        
        Plummer acceleration: a = -GM * r / (r^2 + b^2)^(3/2)

        Parameters:
            pos (np.array) = Position vector [x, y, z] (kpc)
            b (float) = Plummer constant as suggested by Claude.ai (default = 15.0 kpc)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            acc (np.array) = Acceleration vector [a_x, a_y, a_z] (m/s^2)
    """
    # Convert to SI units
    MW_mass = (M * u.solMass).to(u.kg).value
    G_val = const.G.to((u.m ** 3) / (u.kg * np.square(u.s))).value #SI unit conversion suggested by Claude.ai
    
    pos_m = (pos * u.kpc).to(u.m).value  # Convert position to meters
    b_m = (b * u.kpc).to(u.m).value # Convert Plummer constant to meters
    
    # Calculate distance from Milky Way's center
    r = np.sqrt(np.sum(np.square(pos_m)))
    
    # Calculating the Plummer acceleration in the radial direction
    denominator = (np.square(r) + np.square(b_m))**(3/2)
    acc = -(G_val * MW_mass * pos_m) / denominator
    
    return acc

#Circular velocity method (simple model)
def circular_velocity(r, b = 15.0, M = 1.29e12):
    """
        TThis function returns the circular velocity of a star orbiting the Milky Way, while assuming the Milky Way's rotation is static and
        the stars don't transfer energy to each other. This is a SIMPLE model; stars don't actually orbit around the galaxy in a
        perfect circle, since they have velocity components in more than just the radial direction. This equation will test whether my
        leapfrog method works before I model elliptical stellar orbits.

        Parameters:
            r (float) = Star's distance from the Milky Way's center (kpc)
            b (float) = Plummer constant as suggested by Claude.ai (default = 15.0 kpc)
            M (float) = Mass of the Milky Way (most recent measurement = 1.29e12 solar masses - based on Grand et. al 2019)

        Returns:
            v_circ (Quantity) = Star's circular velocity measurement (m/s)
    """
    
    #Converting into SI units for easy calculations
    MW_mass = (M * u.solMass).to(u.kg)
    G_val = const.G.to((u.m ** 3) / (u.kg * np.square(u.s)))
    r_m = (r * u.kpc).to(u.m)
    b_m = (b * u.kpc).to(u.m)
    
    # For circular orbit in Plummer potential: v_circ = sqrt(GM * r^2 / (r^2 + b^2)^(3/2))
    v_circ = np.sqrt((G_val * MW_mass * np.square(r_m) / ((np.square(r_m) + np.square(b_m))**(3/2))))
    
    return v_circ # m/s

#Leapfrog method
def leapfrog_method(pos, vel, dt):
    """
        This function performs one integration step with the Leapfrog Method. I'm using the Leapfrog Method, since it conserves energy
        pretty well over long timescales. It's also the preferred integrator for orbital simulations. 

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
    vel_half_kpc = (vel_half * (u.m / u.s)).to(u.kpc / u.s).value
    new_pos = pos + (dt * vel_half_kpc)
    
    # Step 3: Calculate acceleration at new position
    new_acc = acceleration_plummer(new_pos)
    
    # Step 4: Complete velocity update
    new_vel = vel_half + (0.5 * dt * new_acc)
    
    return new_pos, new_vel

#Converting to Galactocentric coordinates
def galactic_coords(ra, dec, dist, pm_ra, pm_dec, rv):
    """
        This function converts Cartesian coordinates (default coordinate system for observations) to Galactocentric (cylindrical) coordinates, 
        so I can simulate stellar orbits in 3 dimensions within the Milky Way's thin disk. First, I need to create Sky Coordinates for a given
        star, which requires RA (ra), Dec (dec), distance, proper motion * cos(dec) (pm_ra), pm_dec, and radial velocity (rv) measurements.
        I then transform the Sky Coordinates into Galactocentric coordinates with Astropy's Galactocentric() method, where I can decompose
        the positions and velocities back to Cartesian coordinates in relation to the Milky Way's coordinates [X, Y, Z]. 
        
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
            pm_ra_cosdec = pm_ra * (u.mas / u.yr), #Proper motion * cos(dec)
            pm_dec = pm_dec * (u.mas / u.yr),
            radial_velocity = rv * (u.km / u.s),
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
            gal_centric.v_x.to(u.km / u.s).value,
            gal_centric.v_y.to(u.km / u.s).value,
            gal_centric.v_z.to(u.km / u.s).value
            ])

    return position, velocity

def simulate_galactic_orbit(r_0, v_0, dt, t_max): #Need to write more detailed commentary

    """
        This function returns the time, position, and velocity arrays for a star orbiting around the center of the Milky Way, and the
        position/velocity arrrays are in Galactocentric coordinates. The number of Leapfrog integrations is based on the ratio between
        the total simulation time (t_max) and integration time step (dt). In order to keep track of the times, positions, and velocities,
        I need to initialize arrays with zeros, where the length of each array equates to the number of integration steps. The time,
        position, and velocity arrays are updated for every integration step, where the positions and velocities equate to the outputs from
        the galactic_coords() method. 

        Parameters:
            r_0 (np.array) = Array of initial Galactocentric positions [X, Y, Z] (kpc)
            v_0 (np.array) = Array of initial Galactocentric velocities [vX, vY, vZ] (km/s)
            dt (float) = Integration time step (Myr)
            t_max (float) = Total simulation time (Myr)

        Returns:
            times (np.array): Array of time steps (Myr)
            positions (np.array): Array of positions [X, Y, Z] (kpc)
            velocities (np.array): Array of velocities [vX, vY, vZ] (kpc/Myr)
    
    """

    # Calculate the total number of steps
    n_steps = int(t_max / dt)
    
    # Initialize arrays
    times = np.zeros(n_steps + 1) #1D array
    positions = np.zeros((n_steps + 1, 3))  # [x, y, z] in kpc (3D array)
    velocities = np.zeros((n_steps + 1, 3))  # [v_x, v_y, v_z] in kpc/Myr (3D array)
    
    # Set initial conditions
    positions[0] = r_0 #Positions from galactic_coords() method

    #Converting radial velocity back to kpc/Myr
    velocities[0] = ((v_0 * (u.km/u.s)).to(u.kpc/u.Myr)).value #Velocities from galactic_coords() method

    #Print initial conditions
    dist_0 = np.sqrt(np.sum(np.square(r_0))) #r = √(x^2 + y^2 + z^2)
    v_tot = np.sqrt(np.sum(np.square(v_0))) #v = √(v_x^2 + v_y^2 + v_z^2)

    print(f"Initial Galactocentric position: [{r_0[0]:.3f}, {r_0[1]:.3f}, {r_0[2]:.3f}] kpc")
    print(f"Initial distance from Galactic center: {dist_0:.3f} kpc")
    print(f"Initial velocity: [{v_0[0]:.2f}, {v_0[1]:.2f}, {v_0[2]:.2f}] km/s")
    print(f"Total velocity magnitude: {v_tot:.2f} km/s")

    #Calculating circular velocity
    v_circ = (circular_velocity(dist_0).to(u.km / u.s)).value
    print(f"Circular velocity: {v_circ:.2f} km/s\n")
    
    # Convert dt from Myr to seconds for leapfrog
    dt_sec = (dt * u.Myr).to(u.s).value
    
    # Integration loop
    for i in range(n_steps):
        times[i + 1] = times[i] + dt
        
        # Get velocity in m/s for leapfrog
        vel_ms = (velocities[i] * (u.kpc / u.Myr)).to(u.m / u.s).value
        
        # Perform leapfrog step
        new_pos, new_vel = leapfrog_method(positions[i], vel_ms, dt_sec)
        
        # Store new values
        positions[i + 1] = new_pos
        velocities[i + 1] = (new_vel * (u.m / u.s)).to(u.kpc / u.Myr).value
    
    return times, positions, velocities

## Transforming the plots into an animation ##

def animate_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv, dt, t_max, speed, save_animation, disk_radius = 15.0, disk_height = 0.3,
                          trail_length = 500, gif_name = 'Nrich_stellar_orbits.gif'):

    """
        This function plots the positions of each star in both 2D (face-on to the Milky Way's thin disk) and 3D (edge-on) and stitches each
        frame into an animation. Before creating plots, I need to convert all the observed coordinates into Galactocentric coordinates, so I
        can obtain position and velocity arrays. These arrays are then added to two lists (all_positions and all_velocities) that track the
        positions and velocities of ALL the stars in the simulation. Each star's color is randomly selected, since I don't know how many stars
        the user will want to simulate. 

        After obtaining all the stellar positions and velocities, I can now begin making plots. I want to plot the stars, their trajectories,
        and the Milky Way's thin disk. I'm visualizing the thin disk, since most of the Milky Way's stars formed there. I also want to see
        whether the star's orbit leaves the thin disk, since some of the nitrogen-rich stars actually came from merger events with dwarf
        galaxies. I simplified the Milky Way disk to be plotted as a thin cylinder in the 3D plot and a uniform circle in the 2D plot. I also
        plotted a point at the Milky Way's center, representing the location of the supermassive black hole Sagitarrius A*. This point doesn't
        affect the stars at all though; it's just a visual marker. 

        After all the frames are generated, they are stitched together into an animated GIF. The number of GIF frames is dependent on the length
        of the position array and the given animation speed. 

        Parameters:
            ra (list) = List of RA values (degrees)
            dec (list) = List of declination values (degrees)
            dist (list) = List of stellar distances from Earth (kpc)
            pm_ra (list) = List of proper motions in RA direction (mas/yr)
            pm_dec (list) = List of proper motion in Dec direction (mas/yr)
            rv (list) = List of radial velocities measured by Gaia (km/s)
            dt (float) = Integration time step (default = 0.1 s)
            t_max (float) = Total simulation time (default = 1000 Myr)
            disk_radius (float) = Radius of the Milky Way's thin disk (default = 15.0 kpc - diameter of ~30 kpc)
            disk_height (float) = Height of the Milky Way's thin disk (default = 0.3 kpc)
            speed (float) = Total GIF run time (default = 10 s)
            trail_length (float) = Length of tail as the star orbits around the Milky Way (default = 500 positional values)
            save_animation (boolean) = Return True to save a GIF to your computer (default = False)
            gif_name (str) = Name of saved GIF (default = 'Nrich_stellar_orbits.gif'

        Returns:
            anim = Animation of the stellar orbits within in the Milky Way from a 2D (face-on) and 3D (edge-on) point of view
    
    """

    # Initialzing arrays, so I can track all stellar data
    all_positions = []
    all_velocities = []
    all_colors = []

    for i in range(len(ra)):
        print(f"\nProcessing Star {i + 1}...")
        print("="*60)
        
        # Convert from  observed (Cartesian) to Galactocentric coordinates
        r_0, v_0 = galactic_coords(ra[i], dec[i], dist[i], pm_ra[i], pm_dec[i], rv[i])
        
        # Obtain time, position, and velocity arrays for the animation
        times, positions, velocities = simulate_galactic_orbit(r_0, v_0, dt, t_max)

        #Add values to position and velocity arrays
        all_positions.append(positions)
        all_velocities.append(velocities)

        #generate random RGB colors for stars in the animation
        all_colors.append((random.random(), random.random(), random.random()))

    ## Plotting the stars and Milky Way's thin disk ##

    fig = plt.figure(figsize = (14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.view_init(elev=10, azim=10)

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
                zorder = 1)
        ax2.plot(all_positions[i][:, 0], all_positions[i][:, 1],
                color = all_colors[i], linewidth = 1, alpha = 0.3, linestyle = '--',
                zorder = 1)

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
        ax1.set_zlabel('Height (kpc)', fontsize=15)
        ax1.set_title('3D Stellar Orbits Within the Milky Way Thin Disk', fontsize = 15)
        ax1.legend(loc = 'upper right', fontsize = 9)

        max_range = disk_radius * 1.1
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range/2, max_range/2])
    
        # xy-plane projection
        ax2.set_xlabel('Milky Way Disk in X-direction (kpc)', fontsize = 15)
        ax2.set_ylabel('Milky Way Disk in Y-direction (kpc)', fontsize = 15)
        ax2.set_title('Face-On Stellar Orbits within the Milky Way', fontsize = 15)
        ax2.axis('equal')
        ax2.legend(loc = 'upper right', fontsize = 9)

        ax2.set_xlim([-max_range, max_range])
        ax2.set_ylim([-max_range, max_range])


    ## Claude.ai suggested to include the init() and animate() methods inside this function
    def init():
        """ Initialize animation (suggested by Claude.ai) """
        
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

    def animate(frame):
        """ Update animation for each frame (suggested by Claude.ai) """
    
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
    if save_animation == True:
        print(f"Saving animation to {gif_name}...")
        anim.save(gif_name, writer = 'pillow', fps = 20, dpi = 100)
        print("Animation saved!")

    plt.tight_layout()
    plt.show()

    return anim

if __name__ == '__main__':

    #Write argparse arguments
    parser = argparse.ArgumentParser(description = 'This .py file simulates and animates the stellar orbits of nitrogen-rich stars within the thin disk of the Milky Way. Nitrogen-rich stars are theorized to originate from globular clusters, which are clusters of stars formed very early in the lifetime of our galaxy. One way to study the evolution of the Milky Way is to track the motions of nitrogen-rich stars over time. In order to create an orbital model, we require the radial velocity and proper motion measurements of a star, which are determined by the Gaia telescope. With this file, you can simulate the orbits of stars that are currently within 8 kpc of the Milky Way center. In order to run the animation, you must specify the number of stars you want in the simulation (stellar_number). You can also change these default parameters: integration time step (--time_step), simulation run time (--simulation_time), Total GIF run time (--GIF_time), and whether or not you want to save the animation as a GIF (--save_animation). To run your simulation, please type this in your terminal: python test_SDSS_orbit.py stellar_number --time_step --simulation_time --GIF_time --save.')

    parser.add_argument('stellar_number', type = int, help = 'Number of stellar orbits you want to model (min = 1, max = 108)')
    parser.add_argument('--time_step', type = float, default = 0.1, help = 'Integration time step in seconds (default = 0.1 s)')
    parser.add_argument('--simulation_time', type = float, default = 1000, help = 'Total simulation time in Myr (default = 1000 Myr)')
    parser.add_argument('--GIF_time', type = int, default = 10, help = 'Total GIF run time (default = 10 s)')
    parser.add_argument('--save', type = bool, default = False, help = 'Type true if you want to save the animation as a GIF (default = False)')

    args = parser.parse_args()

    #Obtaining dataframe of the Gaia/SDSS file
    df_cross_gaia = reading_file()

    #Appending  values to the arrays based on the number of stars the user wants to plot
    #Initializing arrays for stellar parameters
    ra = []
    dec = []
    dist = []
    pm_ra = []
    pm_dec = []
    rv = []
    random_star = [] #Array of random indices generated by random number generator

    #Creates an array of random integers that don't repeat, and the integers represent indices in the Gaia/SDSS table
    indices_array = random.sample(range(0, len(df_cross_gaia['ra_1']) - 1), args.stellar_number) #Chooses random stars in the table

    for i in range(args.stellar_number):
        random_index = indices_array[i] #rand_index corresponds to the random index generates above

        #Appending values to arrays
        ra.append(df_cross_gaia['ra_1'][random_index])
        dec.append(df_cross_gaia['dec_1'][random_index])
        dist.append(df_cross_gaia['dist_from_Earth(kpc)'][random_index])
        pm_ra.append(df_cross_gaia['pmra_2'][random_index])
        pm_dec.append(df_cross_gaia['pmdec'][random_index])
        rv.append(df_cross_gaia['dr2_radial_velocity'][random_index])

    #Running animation
    animate_galactic_orbit(ra, dec, dist, pm_ra, pm_dec, rv, args.time_step, args.simulation_time, args.GIF_time, args.save)