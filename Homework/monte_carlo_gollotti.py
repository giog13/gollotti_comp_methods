#Importing packages
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
time_delay = 200 #Time delay between frames in animation (milliseconds)

#Initalizing simulation data globally
x_positions = [start_x] #Array of all the photon's positions
y_positions = [start_y]

total_dist = 0.0 #Total distance traveled by photon (in meters)
sim_exit = False


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

if __name__ == "__main__":

    ##Configure animation##

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
        frames = range(max_scatters),
        init_func = init,
        blit = True,
        interval = time_delay,
        repeat = False
    )

    # Save and display as HTML
    plt.show()

