import numpy as np
import argparse

def drop_time(h, g):
    
    """"
    This function calculates the number of seconds it takes for an object to fall for a specific height. Earth's gravitational
    acceleration is the default 'g' value. 
    
    Parameters:
        h (float) = Height of object above ground (meters)
        
        g (float) = Gravitational acceleration (m/s^2)
                    Default value = 9.81 (Earth)
    
    Returns:
        t (float) = Time it takes for object to fall from specific height (seconds)
    
    """
    
    t = ((2.0 * h) / g) ** (1/2)
    return t


if __name__ == "__main__": #Runs from command line
    
    #Creates parser object
    parser = argparse.ArgumentParser(description = 'Input object height <meters>')

    #Adding height and gravitational acceleration as variables for the parser
    parser.add_argument('h', type = float, help = 'Height of object above ground') #Positional argument
    parser.add_argument('--g', type = float, default = 9.81, help = 'Gravitational acceleration') #Keyword argument

    #Makes new variables into arguments for parser
    args = parser.parse_args()
    
    #Prints the time it takes for an object to fall from a specific height
    print(f"It takes {drop_time(args.h, args.g):.2f} seconds for an object to fall from {args.h} meters above ground.")