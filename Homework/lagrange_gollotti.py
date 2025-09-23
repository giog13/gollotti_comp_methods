#Importing packages
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
import argparse


#Lagrange functions
def lagrange(r):
    """
        This function determines the roots of the first Lagrangian equation, where L1 is equivalent to this equation's only 
        real root. For reference, roots are points where the function crosses the x-axis (y = 0). It returns the unitless 
        y-values of the Lagrangian.
        
        Parameters:
            r (float) = Distance away from Earth (meters)
        
        Constants:
            G (float) = Gravitational constant
            M (float) = Mass of Earth (kg)
            m (float) = Mass of the Moon (kg)
            R (float) = The Moon's distance from Earth (meters)
            omega (float) = Angular velocity (1/s)
        
        Returns:
            L (float) = Lagrangian point (meters) - only the unitless value is returned
    
    """
    
    G = const.G
    M = const.M_earth
    m = 7.348e22 * u.kg
    R = 3.844e8 * u.m
    omega = 2.662e-6 * (1/u.s)
    
    L = ((G * M) / np.square(r * u.m)) - ((G * m) / np.square(R - (r * u.m))) - (np.square(omega) * (r * u.m))
    
    return L.value

def deriv_lagrange(r):
    
    """
        This function represents the derivative of the first Lagrangian equation. The derivative is required for using
        the Newton's method properly. It returns the unitless y-values of the Lagrangian.
        
        Parameters:
            r (float) = Distance away from Earth (meters)
        
        Constants:
            G (float) = Gravitational constant
            M (float) = Mass of Earth (kg)
            m (float) = Mass of the Moon (kg)
            R (float) = The Moon's distance from Earth (meters)
            omega (float) = Angular velocity (1/s)
        
        Returns:
            L (float) = Lagrangian point (meters) - only the unitless value is returned
    
    """
    
    G = const.G
    M = const.M_earth
    m = 7.348e22 * u.kg
    R = 3.844e8 * u.m
    omega = 2.662e-6 * (1/u.s)
    
    L_prime = ((-2 * G * M)/((r * u.m) ** 3)) - ((2 * G * m) / ((R - (r * u.m)) ** 3)) - np.square(omega)
    
    return L_prime.value


#Newton's method
def newtons_method(x_1, f = lagrange, f_prime = deriv_lagrange, tol = 1e-10, max_iter = 100000): #Add commentary
    
    """
        This function represents the Newton's method, which determines L1's location with just one input guess value. 
        
        Parameters:
            x_1 (float) = Input guess from the user (positional)
            f (function) = Input equation (keyword) - in this instance, we are using the first Lagrangian equation.
            f_prime (function) = Derivative of input equation (keyword)
            tol (float) = Tolerance of how small a number can be before the function fails (keyword)
            max_iter (int) = Maximum number of iterations required to determine convergence (keyword)
            
        Values:
            num_iter (int) = Keeps track of how many times the while loop was iterated. The function ends (returns a None
                             type value) if num_iter exceeds the maximum number of iterations.
            x_2 (float) = Change in x and becomes the new guess value after every iteration of the while loop.
        
        Returns:
            x_2 (float) = Lagrangian point along x-axis (only if (x_2 - x_1) < tolerance value --> close enough to 0)
        
    """
    
    num_iter = 0
    
    x_2 = x_1 - (f(x_1) / f_prime(x_1)) #x_2 isn't dependent on the second guess. It's dependent on P(x_1) and its derivative.
    
    while abs(x_2 - x_1) > tol: #Make sure the derivative isn't 0, or the function will fail
        
        num_iter += 1
        x_1 = x_2 #x_1 keeps changing until a root is found
        x_2 = x_1 - (f(x_1) / f_prime(x_1))
        
        if num_iter > max_iter: #Will be an infinite loop if we don't define the max iterations
            print(f"Did not converge after {max_iter} iterations.")
            return None #Won't return any values
        
    return x_2


#Secant method
def secant_method(x_1, x_2, f = lagrange, tol = 1e-10, max_iter = 100000): 
    
    """
        This function represents the Secant method, which determines L1's location with two input guess values. L1's actual
        value will be located somewhere within the region between the first and second guess. Unlike the Newton's method, this
        function doesn't require us to know the Lagrangian equation's derivative. Instead, it takes the numerical derivative
        between the two guesses and returns the Lagrangian's location if the secant line between two guesses is nearly (if not
        completely) vertical.
        
        Parameters:
            x_1 (float) = First guess from the user (positional)
            x_2 (float) = Second guess from the user (positional)
            f (function) = Input equation (keyword) - in this instance, we are using the first Lagrangian equation.
            tol (float) = Tolerance of how small a number can be before the function fails (keyword)
            max_iter (int) = Maximum number of iterations required to determine convergence (keyword)
        
        Values:
            num_iter (int) = Keeps track of how many times the while loop was iterated. The function ends (returns a None
                             type value) if num_iter exceeds the maximum number of iterations.
            x_3 (float) = Change in x and becomes the current guess value after every iteration of the while loop. Similarly, 
                          x_2 then becomes the previous guess value.
        
        Returns:
            x_3 (float) = Lagrangian point along x-axis (only if (x_3 - x_2) < tolerance value --> close enough to 0)
    
    """
    
    num_iter = 0
    
    x_3 = x_2  - (f(x_2) * x_2  - ((x_2 - x_1) / (f(x_2) - f(x_1)))) #Calculates the numerical derivative between 2 guesses
    
    while abs(x_2 - x_3) > tol: #Make sure the derivative isn't 0, or the function will fail
        num_iter += 1
        
        x_1 = x_2 #x_2 becomes the first guess
        x_2 = x_3 #x_3 becomes the second guess
        
        x_3 = x_2  - (f(x_2) * ((x_2 - x_1) / (f(x_2) - f(x_1))))
        
        if num_iter > max_iter: #Will be an infinite loop if we don't define the max iterations
            print(f"Did not converge after {max_iter} iterations.")
            return None #Won't return any values
        
    return x_3


#Plotting method
    
def plotting_L1_points(L1_point, guess_1, guess_2 = None):
    
    """"
        This function plots the Lagrangian equation, the user's guesses, and L1's location after the implementation of the
        either the NEWTON or SECANT method. The plot will pop up in the terminal after the user inputs 1-2 guesses.
        
        Parameters:
            L1_point (float) = Location of L1, found either with the NEWTON or SECANT method (positional)
            guess_1 (float) = User's first guess (positional)
            guess_2 (float) = User's second guess (only relevant for the SECANT method) (keyword)
        
        Returns:
            Plot of the Lagrangian, where the guesses and the actual L1 are also plotted
    
    """
    
    r_range = np.linspace(1e8, 2e9, 1000)
    
    #Finding possible Lagrange values
    L_vals = []
    
    for r in r_range:
        L_vals.append(lagrange(r)) #Required for plotting the Lagrangian equation
        
    #Plotting function
    
    fig = plt.figure(figsize = (12, 8))

    plt.xlabel('r (m)', fontsize = 15)
    plt.ylabel('Distance (m)', fontsize = 15)
    
    plt.xlim(2.0e8, 5.0e8)
    plt.ylim(-0.05, 0.05)
    
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    
    plt.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 2, zorder = 1, label = 'y = 0')
    
    plt.plot(r_range, L_vals, linewidth = 3, color = 'blue', zorder = 1, label = r'$L_1$ Equation')
    plt.scatter(L1_point, 0, color = 'orangered', s = 70, zorder = 2, label = f" Actual Location = ({L1_point:.3e}, 0)")
    
    if guess_2 != None: #Represents the SECANT method
        plt.title(r'$L_1$ Equation Using the SECANT Method', fontsize = 20)
        
        plt.scatter([guess_1, guess_2], np.zeros(2), color = 'mediumspringgreen', zorder = 2, s = 70, label = f'Guesses = ({guess_1:.3e}, 0),\n ({guess_2:.3e}, 0)')
        
    if guess_2 == None: #Represents the NEWTON method
        plt.title(r'$L_1$ Equation Using the NEWTON Method', fontsize = 20)
        
        plt.scatter(guess_1, 0, color = 'mediumspringgreen', s = 70, zorder = 2, label = f'Guess = ({guess_1:.3e}, 0)')
        
    plt.legend(loc = 'upper right', fontsize = 15)
    plt.grid()
    plt.show()
    
    
#Method for gathering user input
def user_input():
    
    #Creates parser object
    parser = argparse.ArgumentParser(description = 'Use either the Newtons or Secant method to find the Lagrangian point (L1) between the Earth and the Moon. Using the given plot of the L1 equation with x-limits between 200 million and 500 million meters, guess the location of the L1 point. If you want to use the Newtons method, type NEWTON; and for secant, type SECANT. If you typed NEWTON, please input 1 guess <meters>. If you typed SECANT, please 2 guesses <meters>, where the true value is located somewhere between the two guessed points.'\
                                    
 'For NEWTON method, use this format in your terminal: python lagrange_gollotti.py NEWTON first_guess. ' \
     'Ex: If you are guessing 330 million meters as your guess, type: python lagrange_gollotti.py NEWTON 3.3e8. ' \
                                    
  'For SECANT method, use this format in your terminal: python lagrange_gollotti.py SECANT first_guess --second_guess. '\
      'Ex: If you are guessing 310 million and 330 million meters as your guesses, type: python lagrange_gollotti.py SECANT 3.1e8 --second_guess 3.3e8.')
    
    #Adding variables for the parser
    parser.add_argument('method_type', type = str, help = 'Method for solving nonlinear equations: NEWTON or SECANT <string>') #Positional argument 
    parser.add_argument('first_guess', type = float, help = r'First guess for L1 location (between 2e8 to 5e8) <meters>') #Positional argument 
    parser.add_argument('--second_guess', type = float, default = None, help = r'For SECANT method ONLY: Second guess for L1 method (between 2e8 to 5e8) <meters>') #Keyword argument
    
    #Makes new variables into arguments for parser
    args = parser.parse_args()
    
    if args.method_type == 'NEWTON':
        
        newton_answer = newtons_method(args.first_guess)
        
        print(f"Langrange Point: {newton_answer:.3e} meters")
        
        plotting_L1_points(newton_answer, args.first_guess)
        
    if args.method_type == 'SECANT':
        
        if args.second_guess == None:
            raise ValueError("Must include a second guess (--second_guess [float value]).")
        
        secant_answer = secant_method(args.first_guess, args.second_guess)
        
        print(f"Lagrange Point: {secant_answer:.3e} meters")
        
        plotting_L1_points(secant_answer, args.first_guess, guess_2 = args.second_guess)
        

#Running from command line
if __name__ == "__main__":
    
    user_input()
    
    input("Press Enter to close the plot and exit...")