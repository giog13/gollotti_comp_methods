#Importing packages
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

#4th-order Runga-Kunta method
def runga_kutta_4_order(f, state, h): #Debugged with Claude.ai
    """ Single step of RK4 method """ #State = dx_dt, dy_dt, dx_2_dt, dy_2_dt
    k_1 = h * f(state)
    k_2 = h * f(state + k_1/2)
    k_3 = h * f(state + k_2/2)
    k_4 = h * f(state + k_3)
    
    return state + (h/6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4) #4th order Runga-Kutta algorithm

#Derivatives for the equations of motion
def equation_motion(state, G = 1.0, M = 10.0, L = 2.0):
    
    """
        This function returns an array with solutions for each of the first-order ODEs: dx_dt (velocity in the 
        x-direction), dy_dt (velocity in the y-direction), dx_2_dt (acceleration in the x-direction), and dy_2_dt
        (acceleration in the y-direction). Each of the ODEs are dependent on r, which is the distance the ball
        bearing is away from the rod in spherical coordinates. If r is less than 1e-10, this function returns
        acceleration values of 0, so we avoid division by zero.
        
        Parameters:
            state (np.array) = Array with values representing the x/y coordinates, and x/y velocities
            G (float) = Gravitational constant (G = 1)
            M (float) = Mass of the rod (M = 10.0)
            L (float) = Length of the rod (L = 2.0)
        
        Returns (in the form of a np.array):
            dx_dt (float) = Velocity in the x-direction
            dy_dt (float) = Velocity in the y-direction
            dx_2_dt (float) = Acceleration in the x-direction
            dy_2_dt (float) = Acceleration in the y-direction
        
    """
    
    x, y, v_x, v_y = state
    
    r = np.sqrt(np.square(x) + np.square(y)) #Distance between ball bearing and rod at a given time
    
    #First-order differential (represent velocity)
    dx_dt = v_x
    dy_dt = v_y

    #Second-order differential (represent acceleration)
    dx_2_dt = (-G * M) * (x / (np.square(r)) * np.sqrt(np.square(r) + ((1/4) * np.square(L))))
    dy_2_dt = (-G * M) * (y / (np.square(r)) * np.sqrt(np.square(r) + ((1/4) * np.square(L))))
    
    if r < 1e-10: #Avoids the possibility of dividing by zero
        return np.array([dx_dt, dy_dt, 0.0, 0.0])
    
    return np.array([dx_dt, dy_dt, dx_2_dt, dy_2_dt])


#Solving the equations of motion with the Runga-Kutta method
def solving_R4(x_0, y_0, vx_0, vy_0, t_start, t_end, h): #Debugged with Claude.ai
    
    #Initializing time array
    t_range = np.arange(t_start, t_end + h, h) #Range of t values that will be evaluated (t_start = 0.0, t_end = 10)
    n_steps = len(t_range)
    
    #Initializing position/velocity arrays
    x_array = np.zeros(n_steps)
    y_array = np.zeros(n_steps)
    vx_array = np.zeros(n_steps)
    vy_array = np.zeros(n_steps)
    
    #Defining initial conditions
    x_array[0] = x_0 #x_0 = 1.0
    y_array[0] = y_0 #y_0 = 0.0
    vx_array[0] = vx_0 #vx_0 = 0.0
    vy_array[0] = vy_0 #vy_0 = 1.0
    
    #Runga-Kutta integration
    for i in range(len(t_range) - 1):
        state = np.array([x_array[i], y_array[i], vx_array[i], vy_array[i]]) #(x, y, vx, vy)
        
        new_state = runga_kutta_4_order(equation_motion, state, h) #Updates state coordinates based on the Runga-Kutta method
        
        x_array[i + 1] = new_state[0] #New x_val
        y_array[i + 1] = new_state[1] #New y_val
        vx_array[i + 1] = new_state[2] #New velocity in x-direction
        vy_array[i + 1] = new_state[3] #New velocity in y-direction
        
    return t_range, x_array, y_array, vx_array, vy_array

def plot_rod_animation(t_start, t_end, h, x_0 = 1.0, y_0 = 0.0, vx_0 = 0.0, vy_0 = 1.0, L = 2.0):
    
    t, x, y, vx, vy = solving_R4(x_0, y_0, vx_0, vy_0, t_start, t_end, h)
    
    #Creating animation figure (help from Claude.ai)
    
    fig = go.Figure()
    
    #Axis limits
    x_range = [min(x.min(), -1) - 1, max(x.max(), 1) + 1]
    y_range = [min(y.min(), -1) - (L/2), max(y.max(), 1) + 1]
    
    # Downsample for animation (every nth point) - taken from Claude.ai
    skip = 5
    t_anim = t[::skip]
    x_anim = x[::skip]
    y_anim = y[::skip]
    
    #Adding the rod (horizontal line at origin)
    fig.add_trace(go.Scatter(
        x = [0, 0],
        y = [y_range[0], y_range[1]],
        mode = 'lines',
        line = dict(color = 'gray', width = 8),
        name = 'Rod',
        showlegend = True
    ))
    
    #Adding trajectory path
    fig.add_trace(go.Scatter(
        x = x,
        y = y,
        mode = 'lines',
        line = dict(color = 'mediumspringgreen', width = 2),
        name = 'Full Trajectory',
        showlegend = True
    ))
    
    # Add trail (will show past positions)
    fig.add_trace(go.Scatter(
        x = [x_anim[0]],
        y = [y_anim[0]],
        mode = 'lines',
        line = dict(color='deeppink', width=2),
        name='Trail',
        showlegend=True
    ))
    
    # Add ball bearing (moving point)
    fig.add_trace(go.Scatter(
        x = [x_anim[0]],
        y = [y_anim[0]],
        mode = 'markers',
        marker = dict(size = 15, color = 'black'),
        name = 'Ball Bearing',
        showlegend = True
    ))
    
     # Create frames for animation
    frames = []
    trail_length = 50  # Number of points to show in trail
    
    for i in range(len(x_anim)):
        # Trail: show last trail_length points
        trail_start = max(0, i - trail_length)
        trail_x = x_anim[trail_start:i+1]
        trail_y = y_anim[trail_start:i+1]
        
        frame = go.Frame(
            data=[
                # Rod (unchanged)
                go.Scatter(x=[0, 0], y=[y_range[0], y_range[1]]),
                # Full trajectory (unchanged)
                go.Scatter(x=x, y=y),
                # Trail
                go.Scatter(x=trail_x, y=trail_y),
                # Ball bearing
                go.Scatter(x=[x_anim[i]], y=[y_anim[i]])
            ],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title='Ball Bearing Orbiting Around Rod',
        xaxis=dict(range = x_range, title = 'x', scaleanchor = 'y', scaleratio = 1),
        yaxis=dict(range = y_range, title = 'y'),
        width=800,
        height=800,
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'steps': [
                {
                    'args': [[f'frame{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'method': 'animate',
                    'label': f'{t_anim[i]:.1f}'
                }
                for i in range(len(x_anim))
            ]
        }]
    )
    
    fig.show()

if __name__ == '__main__': #make argparse arguments
    
    #Creating parser object
    parser = argparse.ArgumentParser(description = 'This .py file returns the plotly animation for a ball bearing orbiting around a rod of length L = 2.0 m if they were floating in space as "space garbage." Unlike the Earth orbiting the Sun, the orbit of the ball bearing around the rod will not be circular. Instead, it will precess around the rod, since the rod has a much weaker gravitational force on the ball bearing than the Sun on the Earth. In order to get an animation, you need to type this in your terminal: python ODE_gollotti.py . You are welcome to change the start time (start_time), end time (end_time), and time step (time_step).If you want to make a change, use this format in your terminal: python ODE_gollotti.py --start_time 0.0 --end_time 100.0 --time_step 0.05.')
    
    parser.add_argument('--start_time', type = float, default = 0.0, help = 'Start time (seconds) - default = 0.0s. In order to make the change, type this in your terminal: python ODE_gollotti.py --start_time [number goes here]')
    parser.add_argument('--end_time', type = float, default = 100.0, help = 'End time (seconds) - default = 100.0 s. In order to make the change, type this in your terminal: python ODE_gollotti.py --end_time [number goes here]')
    parser.add_argument('--time_step', type = float, default = 0.05, help = 'Time step for Runga-Kutta method (seconds) - default = 0.05. In order to make the change, type this in your terminal: python ODE_gollotti.py --time_step [number goes here].')
    
    #Makes new variables into arguments for parser
    args = parser.parse_args()
    
    plot_rod_animation(t_start = args.start_time, t_end = args.end_time, h = args.time_step)