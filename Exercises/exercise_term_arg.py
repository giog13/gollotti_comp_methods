import numpy as np
import sys

def drop_time(h, g = 9.8): #h = positional, g = keyword
    t = ((2.0 * h) / g) ** (1/2) #Avoid using a package unless absolutely necessary
    return t

if __name__ == "__main__": #Runs from command line
    print(drop_time(float(sys.argv[1]))) #index 0 = .py filename, cast arg as float (initially a string)