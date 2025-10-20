#Importing packages
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from astropy.io import fits
import astropy.constants as const
import astropy.units as u

import argparse
import sys


## Well-defined slab ##

#Constants
cross_section_T = 6.652e-25 * (u.cm ** 2) #Thompson cross-section
mfp = 150.0 * (u.m) #mean free path, where n and the cross-section are well-defined
slab_width = (1.0 * (u.km)).to(u.m) #width of slab

#Simulation parameters
start_x = slab_width.value / 2 #Center of slab
start_y = slab_width.value / 2
start_angle = 0.0

max_scatters = 30 #Limit for number of frames in the simulation
time_delay = 500 #Time delay between frames in animation (milliseconds)

#Initalizing simulation data globally
x_positions = [start_x] #Array of all the photon's positions
y_positions = [start_y]

total_dist = 0.0 #Total distance traveled by photon (in meters)
sim_exit = False

## SIMULATING PHOTONS IN A WELL-DEFINED SLAB ##

def transformation_method(L = mfp.value):
    """
        This function returns the distance the photon travels after it scatters off an electron. The probability is
        an exponential function, which is why we're using the transformation method. The distance changes based
        on the random probability generated with the numpy random function.
        
        Parameter:
            L (float) = Fixed mean free path (150 meters)
            
        Returns:
            dist (float) = Distance photon travels (meters)
    """
    
    dist = -L * np.log(np.random.rand()) #Random() returns "random" number between 0-1 (represents probability)
    
    return dist


def thompson_angle():
    
    """
        This function calculates and returns the photon's angle, based on a random number between -π to π.
        
        Returns:
            theta (float) = Photon's angle with respect to the photon's current direction
    """
    
    theta = np.pi * random.uniform(-1, 1) #Picks random angle between -π to π 
    
    return theta


def simulate_one_scatter(x_positions, y_positions, total_dist): #Debugged with Claude.ai
    
    """
        This function simulates one scatter event, and it's called on once per animation frame. It returns
        the x/y positions (x_position and y_position), total distance traveled by the photon (total_dist),
        and whether or not the photon left the slab (exit).
        
        Parameters:
            x_positions (np.array) = Array of all the photon's x-coordinates (both starting and current)
            y_positions (np.array) = Array of all the photon's y-coordinates (both starting and current)
            total_dist (float) = Total distance traveled by the photon (meters)
        
        Variables:
            starting_x/y (float) = Point where photon scatters - last x and y-coordinates in the arrays
            distance (float) = Distance until next scatter, which is determined by the transformation method
            angle (float) = Angle of scatter with respect to the direction of motion, determined by the Thompson's Angle equation
            delta_x/y (float) = Change in c and y-coordinates after the scattering event
            current_x/y (float) = Current coordinates are dependent on the sum of the starting x/y-coords and the change in                                         coordinates (delta x/y)
            
        Returns:
           x_positions (np.array) = Array of all the photon's x-coordinates (both starting and current)
           y_positions (np.array) = Array of all the photon's y-coordinates (both starting and current)
           total_dist (float) = Total distance traveled by the photon (meters)
           time (float) = Total time it took for the photon to travel out of the slab (seconds)
           exit (boolean) = Return "True" when the photon exits the slab / "False" when photon is still inside the slab
        
    """
    
    #Initializing current position
    starting_x = x_positions[-1] 
    starting_y = y_positions[-1]

    #Distance to next scatter
    distance = transformation_method()

    #Position before the scatter
    angle = thompson_angle()

    delta_x = distance * np.cos(angle)
    delta_y = distance * np.sin(angle)

    current_x = starting_x + delta_x 
    current_y = starting_y + delta_y

    #Append new positions to arrays
    x_positions.append(current_x)
    y_positions.append(current_y)

    #Determining the total distance
    total_dist += distance #Total distance traveled
    time = (total_dist * u.m) / const.c #Total time elapsed  (t = total_dist / speed of light)

    exit = False
    
    #Stopping loop once photon leaves the slab
    if (current_x > slab_width.value) or (current_x < 0.0):
        exit = True

    if (current_y > slab_width.value) or (current_y < 0.0):
        exit = True

    #Checking whether the photon traveled back to the center of the slab
    if (current_x == start_x) and (current_y == start_y): #Restarts the whole method

        x_positions = [start_x] #Array of all the photon's positions
        y_positions = [start_y]

        current_x = start_x #Current position changes after every scattering event (starts at the center)
        current_y = start_y

        total_dist = 0
    
    return x_positions, y_positions, total_dist, time, exit

def init(): #Taken from Gemini AI
    """Initialization function for the animation."""
    line.set_data([], [])
    scat_point.set_data([], [])
    current_point.set_data([], [])
    
    return line, scat_point, current_point


def update_animation(frame): #Code debugged by Claude.ai
    """ 
        This function is by FuncAnimation for each frame, which updates each new scattering event in the animation. 
        
        Parameters:
            frame (int) = Represents each frame of the animation (dependent on the max_scatter value)
            
        Returns:
            line (object) = Set of data representing the distances between each scattering event
            scat_point (object) = Set of data points corresponding to each scattering location (pink in the gif)
            current_point (object) = Set of data points corresponding to the photon's current position (green in the gif)
        
    """    
    global x_positions, y_positions, total_dist, sim_exit #Variables defined outside this function (initializers)
    
    if sim_exit == True: #Ends the whole function/animation
        return line, scat_point, current_point
    
    #Run one scatter event
    x_positions, y_positions, total_dist, time, sim_exit = simulate_one_scatter(x_positions, y_positions, total_dist)
    
    #Update line (photon path)
    line.set_data(x_positions, y_positions)
    
    #Updates points of scatter
    if len(x_positions) > 1: #If there is more than 1 coordinate
        scat_point.set_data(x_positions[:-1], y_positions[:-1]) #Scatter point = second-to-last coordinate
        
    #Updates current positions
    current_point.set_data(x_positions[-1], y_positions[-1]) #Current point = last udpated coordinate
    
    #Updates title with every frame
    ax.set_title(f'Photon Scattering In Slab | L: {mfp.value} m | Scatters: {len(x_positions) - 1}', fontsize = 15)
    
    #Update stats text box (total distance and time)
    stats_text.set_text(f'Total Distance: {total_dist:.3f} m\n'
                       f'Time Elapsed: {time.to(u.microsecond):.3f}')
    
    return line, scat_point, current_point, stats_text

    
## SIMULATING PHOTONS IN THE SUN ##

#Simulation parameters
cross_section_T = 6.652e-25 * (u.cm ** 2) #Thompson cross-section

solar_start_x = 0.0 #Center of the Sun (do I use Cartesian or Polar coordinates?)
solar_start_y = 0.0
solar_start_z = 0.0
start_angle = 0.0

solar_max_scatters = 100 #Limit for number of frames in the simulation
solar_time_delay = 200 #Time delay between frames in animation (milliseconds)

#Initalizing simulation data globally
solar_x_positions = [solar_start_x] #Array of all the photon's positions
solar_y_positions = [solar_start_y]
solar_z_positions = [solar_start_z]

solar_total_dist = 0.0 #Total distance traveled by photon (in meters)
radius = 0.0 #Photon's distance from the center of the Sun
solar_sim_exit = False

def solar_mean_free_path(radius, cross_section = cross_section_T):
    
    """
        This function calculates the mean free path based on the current radius. It returns the distance the 
        photon traveled between the scatter and current point.
        
        Parameters:
        
        Variables:
        
        Returns:
        
    """
    
    #Electron density of the Sun depends on the photon's distance from the center (r) --> Is r randomly generated?
    n_e = (2.5e26 * np.exp(-(radius) / (0.096 * u.R_sun).to(u.m).value)) * (u.cm ** -3) 
    
    #Calculating mean free path (dependent on electron density and Thompson's cross-section)
    L = 1.0 / (n_e * cross_section)
    
    #Scaling up for visualization (real mean free path is too small to see in an animation)
    vis_scale = 1e11 #Scale up by an order of 1 billion
    
    return (L.to(u.m)).value * vis_scale


def simulate_solar_scatter(x_positions, y_positions, z_positions, total_dist): #Debugged with Claude.ai
    
    """
        This function simulates one scatter event, and it's called on once per animation frame. It returns
        the x/y/z positions (x_positions, y_positions, z_positions), total distance traveled by the photon (total_dist),
        the distance of the photon from the center of the Sun (radius), the time elapsed for a photon to leave the
        Sun (time), and whether or not the photon left the Sun (exit).
        
        Parameters:
            x_positions (np.array) = Array of all the photon's x-coordinates (both starting and current)
            y_positions (np.array) = Array of all the photon's y-coordinates (both starting and current)
            z_positions (np.array) = Array of all the photon's z-coordinates (both starting and current)
            total_dist (float) = Total distance traveled by the photon (meters)
        
        Variables:
            starting_x/y/z (float) = Point where photon scatters - last x/y/z-coordinates in the arrays
            distance (float) = Distance until next scatter, which is determined by the solar_mean_free_path() method
            angle (float) = Angle of scatter with respect to the direction of motion, determined by the Thompson's Angle equation
            delta_x/y/z (float) = Change in x/y/z-coordinates after the scattering event
            current_x/y/z (float) = Current coordinates are dependent on the sum of the starting x/y/z-coords and the change in                                         coordinates (delta x/y)
            
        Returns:
           x_positions (np.array) = Array of all the photon's x-coordinates (both starting and current)
           y_positions (np.array) = Array of all the photon's y-coordinates (both starting and current)
           z_positions (np.array) = Array of all the photon's z-coordinates (both starting and current)
           total_dist (float) = Total distance traveled by the photon (meters)
           radius (float) = Photon's distance from the Sun's center (meters)
           time (float) = Total time it took for the photon to travel out of the slab (seconds)
           exit (boolean) = Return "True" when the photon exits the slab / "False" when photon is still inside the slab
        
    """
    
    #Initializing current position
    starting_x = x_positions[-1] 
    starting_y = y_positions[-1]
    starting_z = z_positions[-1]
    
    #Current radius
    current_radius = np.sqrt(np.square(starting_x) + np.square(starting_y) + np.square(starting_z))

    #Distance to next scatter
    distance = solar_mean_free_path(current_radius)

    #Angles relative to direction of motion
    phi = random.uniform(0.0, 2.0 * np.pi) #Azimuthal angles
    theta = random.uniform(-np.pi, np.pi) #Polar angle

    #Direction vector components
    delta_x = distance * np.sin(theta) * np.cos(phi)
    delta_y = distance * np.sin(theta) * np.sin(phi)
    delta_z = distance * np.cos(theta)

    #New position
    current_x = starting_x + delta_x 
    current_y = starting_y + delta_y
    current_z = starting_y + delta_z

    #Append new positions to arrays
    x_positions.append(current_x)
    y_positions.append(current_y)
    z_positions.append(current_z)

    #Determining the total distance
    total_dist += distance #Total distance traveled
    time = (total_dist * u.m) / const.c #Total time elapsed  (t = total_dist / speed of light)

    exit = False
    
    #Stopping loop once photon "leaves" the Sun
    if (current_radius >= (0.9 * const.R_sun).to(u.m).value): #Density drops quickly and photon longer scatters at this point
        exit = True

    #Checking whether the photon traveled back to the center of the slab
    if (current_x == solar_start_x) and (current_y == solar_start_y) and (current_z == solar_start_z): #Restarts the whole method

        x_positions = [solar_start_x] #Array of all the photon's positions
        y_positions = [solar_start_y]
        z_positions = [solar_start_z]

        current_x = solar_start_x #Current position changes after every scattering event (starts at the center)
        current_y = solar_start_y
        current_z = solar_start_z

        total_dist = 0.0
        radius = 0.0
    
    return x_positions, y_positions, z_positions, total_dist, current_radius, time, exit

def solar_init(): #Updated with Claude.ai
    """Initialization function for the animation."""
    solar_line.set_data(np.array([]), np.array([]))
    solar_line.set_3d_properties(np.array([]))
    
    solar_scat_point.set_data(np.array([]), np.array([]))
    solar_scat_point.set_3d_properties(np.array([]))
    
    solar_current_point.set_data(np.array([]), np.array([]))
    solar_current_point.set_3d_properties(np.array([]))
    
    return solar_line, solar_scat_point, solar_current_point, stats_text

def update_solar_animation(frame): #Code debugged by Claude.ai
    """ 
        This function is by FuncAnimation for each frame, which updates each new scattering event in the solar
        animation. 
        
        Parameters:
            frame (int) = Represents each frame of the animation (dependent on the max_scatter value)
            
        Returns:
            solar_line (object) = Set of data representing the distances between each scattering event
            solar_scat_point (object) = Set of data points corresponding to each scattering location (pink in the gif)
            solar_current_point (object) = Set of data points corresponding to the photon's current position (green in the gif)
        
    """   
    #Variables defined outside this function (initializers)
    global solar_x_positions, solar_y_positions, solar_z_positions
    global solar_total_dist, solar_sim_exit, solar_time
    
    if solar_sim_exit == True: #Ends the whole function/animation
        return solar_line, solar_scat_point, solar_current_point, stats_text
    
    #Run one scatter event
    solar_x_positions, solar_y_positions, solar_z_positions, solar_total_dist, radius, solar_time, solar_sim_exit = simulate_solar_scatter(
        solar_x_positions, solar_y_positions, solar_z_positions, solar_total_dist
    )
    
    #Update line (photon path)
    solar_line.set_data(np.array(solar_x_positions), np.array(solar_y_positions))
    solar_line.set_3d_properties(np.array(solar_z_positions))
    
    #Updates points of scatter
    if len(solar_x_positions) > 1: #If there is more than 1 coordinate
        solar_scat_point.set_data(np.array(solar_x_positions[:-1]), np.array(solar_y_positions[:-1])) #Scatter point = second-to-last coordinate
        solar_scat_point.set_3d_properties(np.array(solar_z_positions[:-1]))
        
    #Updates current positions
    solar_current_point.set_data(np.array(solar_x_positions[-1]), np.array(solar_y_positions[-1])) #Current point = last udpated coordinate
    solar_current_point.set_3d_properties(np.array(solar_z_positions[-1]))
    
    #Updates title with every frame
    ax.set_title(f'Solar Photon Scattering | L = {solar_mean_free_path(radius):.2f} m | Scatters: {len(solar_x_positions) - 1}', fontsize = 15)
    
    #Update stats text box (total distance and time)
    stats_text.set_text(f'Total Distance: {solar_total_dist:.3e} m\n'
                        f'Radius: {(radius * u.m).to(u.R_sun):.3e}\n'
                       f'Time Elapsed: {solar_time.to(u.microsecond):.3e}')
    
    return solar_line, solar_scat_point, solar_current_point, stats_text
    

## CALLING EITHER THE SLAB OR SUN FUNCTIONS FROM THE COMMAND LINE ##
    
if __name__ == "__main__":

    ##Configure animation##

    #Creating parser object
    parser = argparse.ArgumentParser(description = 'This .py file allows you to simulate random photon scattering within two different objects: a well-defined SLAB or the SUN. Please type either SLAB or SUN in the command terminal. For example if you want to simulate the Sun, you can type this in your terminal: python monte_carlo_gollotti.py SUN. The module will then return a playable gif to your computer.') #Will allow users to change visualization scale and time delay at some point
    
    #Adding simulation type as a positional parser variable / visualization scale and time delay as keyword variables
    parser.add_argument('simulation_type', type = str, help = 'Type of simulation you want to run: SLAB or SUN')
    
    parser.add_argument('--max_scatters', type = int, default = 100, help = 'The max number of frames in a simulation. Right now, it is set to 100. If you want to increase/decrease the number of frames/scatters, you can change this keyword argument. For example, if you want to increase the max scatters to 300, you can type this: python monte_carlo_gollotti.py SUN --max_scatters 300')
    
    #Figure out how to implement visualization_scale argument into code
    parser.add_argument('--visualization_scale', type = float, default = 1e11, help = 'If you are simulating the Sun, the distance between photon scatters is very small; so small that you cannot see the change with your eyes. In reality, photons take thousands to millions of years to leave the Sun. In order to see the photons scattering on a quick timescale, you can change the visualization scale. Right now, it is set to visualization_scale = 1e11. if you want to change the timescale to something different (for example: 1e12, type this in your terminal: python monte_carlo_gollotti.py SUN --visualization_scale 1e12')
    
    parser.add_argument('--time_delay', type = int, default = 200, help = 'The time delay determines the time between each consecutive frame in the gif. Right now, the time delay is set to 200 milliseconds. If you want to slow down/speed up the time between each frame in the gif, you can change it in the terminal. For example, if you want the time delay to be 300 milliseconds instead, you can type something like this: python monte_carlo_gollotti.py SUN --time_delay 300')
    
    #Makes new variables into arguments for parser
    args = parser.parse_args()
    
    if (args.simulation_type == 'SLAB') or (args.simulation_type == 'slab'):
        
        #Animation set-up
        fig, ax = plt.subplots(figsize=(12, 10))
        line, = ax.plot([], [], color = 'black', linewidth=1.5, label='Photon Path')
        scat_point, = ax.plot([], [], 'o', color = 'deeppink', markersize=6, label='Scattering Event')
        current_point, = ax.plot([], [], 'o', color = 'mediumspringgreen', markersize=8, label='Current Position')

        #Plot configuration
        ax.set_xlim(-slab_width.value/2, slab_width.value * 1.5)
        ax.set_ylim(-slab_width.value/2, slab_width.value * 1.5)

        ax.set_aspect('equal', adjustable='box')

        ax.set_title(f'Photon Scattering In Slab | L: {mfp.value} m', fontsize = 15)
        ax.set_xlabel('x (m)', fontsize = 15)
        ax.set_ylabel('y (m)', fontsize = 15)

        ax.axvline(0, color='blue', linestyle='--', linewidth=2, label='Slab Boundary')
        ax.axvline(slab_width.value, color='blue', linestyle='--', linewidth=2)
        ax.axvspan(0, slab_width.value, color='blue', alpha=0.1, label='Slab Region')


        # Add text box for stats (assisted by Claude.ai) 
        stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=12, verticalalignment='bottom',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(loc = 'lower right', fontsize = 12)

        # Create the animation
        anim = FuncAnimation(
            fig,
            update_animation,
            frames = range(args.max_scatters),
            init_func = init,
            blit = True,
            interval = args.time_delay,
            repeat = False
        )

        plt.show()

        
    if (args.simulation_type == 'SUN') or (args.simulation_type == 'sun'):
        
        #Plotting sphere
        solar_fig = plt.figure(figsize = (10, 8))
        ax = solar_fig.add_subplot(111, projection = '3d')

        plt.rcParams['animation.embed_limit'] = 50  # MB

        #Creating spherical coordinates
        theta = np.linspace(0.0, 2.0 * np.pi, 100)
        phi = np.linspace(0.0, np.pi, 100)
        x = np.outer(np.cos(theta), np.sin(phi)) * (const.R_sun).to(u.m)
        y = np.outer(np.sin(theta), np.sin(phi))  * (const.R_sun).to(u.m)
        z = np.outer(np.ones(np.size(theta)), np.cos(phi))  * (const.R_sun).to(u.m)

        #Plotting sphere
        ax.plot_surface(x, y, z, color = 'orangered', alpha = 0.6, edgecolor = 'none')

        #Adding wireframe
        ax.plot_wireframe(x, y, z, color = 'yellow', alpha = 0.2, linewidth = 0.5)

        #Plotting the scatter points
        solar_line, = ax.plot3D([], [], [], color = 'black', linewidth=1.5, label='Photon Path')
        solar_scat_point, = ax.plot3D([], [], [], 'o', color = 'deeppink', markersize=6, label='Scattering Event')
        solar_current_point, = ax.plot3D([], [], [], 'o', color = 'mediumspringgreen', markersize=8, label='Current Position')

        #Stats text box (taking from Claude.ai)
        stats_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        #Setting labels and title
        ax.set_xlabel('x (m)', fontsize = 15)
        ax.set_ylabel('y (m)', fontsize = 15)
        ax.set_zlabel('z (m)', fontsize = 15)
        ax.set_title('Sun', fontsize = 20)

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        ax.legend(loc = 'upper right', fontsize = 12)

        #Axis limits
        lim = (const.R_sun.to(u.m)).value * 1.1
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])

        # Create the animation
        anim = FuncAnimation(
            solar_fig,
            update_solar_animation,
            frames = range(args.max_scatters),
            init_func = solar_init,
            blit = False, #Redraws entire figure each frame (slower but more reliable for 3D plots)
            interval = args.time_delay,
            repeat = False
        )

        plt.show() #I wonder if there's a way for the user to actively change the orientation of the animation 
