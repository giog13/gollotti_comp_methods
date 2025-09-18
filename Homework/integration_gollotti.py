#Importing packages
import numpy as np
import matplotlib.pyplot as plt
import argparse


#Function undergoing integration
def exp_func(t):
    return np.e ** (-(t ** 2))


#Trapezoid rule
def trap_rule(N, a, b, step_size = 0.1, function = exp_func):
    
    """
    This function integrates a given function using the trapezoidal method. This function will integrate over
    bounds of 0 (lower) and 3 (upper), but the user can change the bounds if desired. The given function will
    be iterated over increments of 0.1, which is considered the "step size." Similarly to the bounds, the user
    can change the step size if desired. 
    
    The integral will be iterated N times, where the upper bounds changes from a (0) to b(3) via step_size (0.1)
    increments (ex: ingerate from 0 to 0.1, 0 to 0.2, etc.). With each iteration of the 'for loop,' delta_x will
    change depending on the upper bound, and from there, we can calculate the area of a trapezoidal segment.
    Every time the upper bound changes, the trapezoid's areas will be summed together. After calculating the area,
    it will be added to the trapezoidal method's integration equation, and the answer for the integral will be
    added to an array (sum_trap). x_range and sum_trap will then be returned, so the values can be plotted.
    
    
    Parameters:
        N (int) = Number of times an integral will be calculated (positional argument)
        a (int or float) = Lower bound for the integral (keyword argument)
        b (int or float) = Upper bound for the integral (keyword argument)
        step_size (float) = Incremental change of bounds (keyword argument)
        function (function name) = Function being integrated (keyword argument)
    
    Values:
        area_trap (float) = Area corresponding to each trapezoidal segment
        delta_x (float) = Height of each trapezoidal segment 
        
    
    Returns:
        sum_trap (list) = List of sums corresponding to each iteration of the integration method
        x_range (np.array) = Array of increments spanning the upper and lower bound, where the increments are
                             dependent on the step size
        integral_trap (float) = Final integration of the inputted function
    
    """
    
    x_range = np.arange(a, b, step_size)
    
    sum_trap = [] #List of each trapezoid slice's sum
    
    for i in x_range: #Range of summation (range stops at N)
        
        area_trap = 0 #Area of individual trapezoid slice
        
        delta_x = (i - a) / N #Change in x for each iteration of the integration
        
        for j in range(N + 1): #Loops N times
            area_trap += function(a + (j * delta_x)) #Sum for each iteration
            
        lower_end_point = 0.5 * function(a) #Starting value
        upper_end_point = 0.5 * function(i) #Ending value
        
        #Integration equation for trapezoidal method
        integral_trap = delta_x * (lower_end_point + area_trap + upper_end_point)
        
        sum_trap.append(integral_trap) #Appends each trapezoid's sums together
        
    #print(f"Trapezoid rule with step size of {step_size} = {integral_trap:.3f}")
        
    return x_range, sum_trap, integral_trap #Multiplies height by the sum


#Simpson's Rule
def simpsons_rule(N, a = 0, b = 3, step_size = 0.1, function = exp_func):
    
    """
    This function integrates a given function using the Simpson's rule. This function will integrate over
    bounds of 0 (lower) and 3 (upper), but the user can change the bounds if desired. The given function will
    be iterated over increments of 0.1, which is considered the "step size." Similarly to the bounds, the user
    can change the step size if desired. 
    
    The integral will be iterated N times, where the upper bounds changes from a (0) to b(3) via step_size (0.1)
    increments (ex: ingerate from 0 to 0.1, 0 to 0.2, etc.). With each iteration of the 'for loop,' delta_x will
    change depending on the upper bound, and from there, we can calculate the area under the curve that spans a
    specific bound range. Every time the upper bound changes, the areas will be summed together. After calculating
    the area, it will be added to the Simpson method's integration equation, and the answer for the integral will
    be added to an array (sum_simp). x_range and sum_simp will then be returned, so the values can be plotted.
    
    Parameters:
        N (int) = Number of times an integral will be calculated (positional argument)
        a (int or float) = Lower bound for the integral (keyword argument)
        b (int or float) = Upper bound for the integral (keyword argument)
        step_size (float) = Incremental change of bounds (keyword argument)
        function (function name) = Function being integrated (keyword argument)
    
    Values:
        area_simp (float) = Area under the curve corresponding to different bound ranges
    
    Returns:
        x_range (np.array) = Array of increments spanning the upper and lower bound, where the increments are
                             dependent on the step size
        sum_simp (list) = List of sums corresponding to each iteration of the integration method
    
    """
    
    x_range = np.arange(a, b, step_size)
    
    sum_simp = [] #List of each trapezoid slice's sum
        
    for i in x_range:
        
        area_simp = 0 #Area under the curve of a specific integration bound
        
        delta_x = (i - a) / N #Change in x for each iteration of the integration
        
        lower_end_point = function(a) #Stays constant
        upper_end_point = function(i) #Upper bound incrementally increases by the step_size value every iteration
        
        #Odd integer loop
        for k in range(1, int(N/2) + 1):
            area_simp += 4 * function(a + (((2 * k) - 1) * delta_x))
    
        #Even integer loop
        for k in range(1, int((N/2))):
            area_simp += 2 * function(a + (2 * k * delta_x))
                
        integral_simp = (delta_x/3) * (lower_end_point + area_simp + upper_end_point) #Simpson's equation
        
        sum_simp.append(integral_simp) #Appending each set of bounds' integrations to a list    
        
    #print(f"Simpson's rule with step size of {step_size} = {integral_simp:.3f}")
        
    return x_range, sum_simp, integral_simp


#Plotting method
def plotting_integral(x_range, sum_list, method_type):
    
    fig = plt.figure(figsize = (12, 8))

    plt.xlabel('t', fontsize = 15)
    plt.ylabel('Sum', fontsize = 15)

    if sum_list == sum_simp:
        plt.title('Integrating With Simpsons Method', fontsize = 20)
        plt.plot(x_range, sum_list, color = 'orangered', label = 'Simpsons')
        
    if method_type == 'trapezoidal':
        plt.title('Integrating With Trapezoidal Method', fontsize = 20)
        plt.plot(x_range, sum_list, color = 'blue', label = 'Trapezoidal')
        
    if method_type == 'both':
        plt.title('Integrating With Trapezoid and Simpsons Method', fontsize = 20) # Need to fix this section
        plt.plot(x_range, sum_list[0], color = 'orangered', linewidth = 2, label = 'Simpsons')
        plt.plot(x_range, sum_list[1], color = 'blue', linewidth = 2, linestyle = '--', label = 'Trapezoidal')

    plt.grid()
    plt.legend(loc = 'lower right', fontsize = 15)
    plt.show()


#If '__name__' == "__main__"
if __name__ == "__main__": #Runs from command line #How do we ask for differences in inputs?
    
    #Creates  parser object (specifics about integration)
    parser = argparse.ArgumentParser(description = r'What kind of integration method would you like to use on e^-(t^2)? Here are your choices: trapezoidal, simpsons, or both. How many times do you want to integrate $e^-(t^2)$? Please type an integer value anywhere between 100 - 100000. Optional: Feel free to change the lower/upper bounds and step size if desired. Here are the default values: lower bound = 0, upper bound = 3, step size = 0.1. If you want to change the default values, you will need to type in this order: python integration_gollotti.py num_iterations int_method --a(lower bound) --b(upper bound) --step_size. For example, if you want to use the trapezoid method, num_iterations = 1000, lower_bound = 0, upper_bound = 5, step_size = 0.5, you will need to type this in your terminal: python integration_gollotti.py 1000 trapezoid --upper_bound 5 --step_size 0.5.')

    #Adding height and gravitational acceleration as variables for the parser
    parser.add_argument('num_iterations', type = int, help = 'Number of times an integral will be calculated') #Positional
    parser.add_argument('int_method', type = str, help = 'Type of integration method: trapezoidal, simpsons, or both')#Positional
    parser.add_argument('--lower_bound', type = float, default = 0, help = 'Lower bound for the integral') #Keyword
    parser.add_argument('--upper_bound', type = float, default = 3, help = 'Upper bound for the integral') #Keyword
    parser.add_argument('--step_size', type = float, default = 0.1, help = 'Incremental change of bounds (must be smaller than the difference between the lower and upper bounds)') #Keyword
    

    #Makes new variables into arguments for parser
    args = parser.parse_args()
    
    #Action goes here
    if args.int_method == 'trapezoidal':
        x_range, sum_trap, answer_trap = trap_rule(args.num_iterations, args.lower_bound, args.upper_bound, args.step_size)
        
        print(f"Integral from {args.lower_bound} to {args.upper_bound} with step size of {args.step_size}: {answer_trap:.3f}")
        plotting_integral(x_range, sum_trap, args.int_method)
        
    if args.int_method == 'simpsons':
        x_range, sum_simp, answer_simp = simpsons_rule(args.num_iterations, args.lower_bound, args.upper_bound, args.step_size)
        
        print(f"Integral from {args.lower_bound} to {args.upper_bound} with step size of {args.step_size}: {answer_simp:.3f}")
        plotting_integral(x_range, sum_simp, args.int_method)
        
    if args.int_method == 'both': #x_range is the same for both methods
        x_range, sum_trap, answer_trap = trap_rule(args.num_iterations, args.lower_bound, args.upper_bound, args.step_size)
        print(f"Integral from {args.lower_bound} to {args.upper_bound} with step size of {args.step_size}: {answer_trap:.3f}\n")
        
        x_range, sum_simp, answer_simp = simpsons_rule(args.num_iterations, args.lower_bound, args.upper_bound, args.step_size)
        print(f"Integral from {args.lower_bound} to {args.upper_bound} with step size of {args.step_size}: {answer_simp:.3f}\n")
        
        print(f"Difference between both methods (Trapezoidal - Simpsons): {(answer_trap - answer_simp):.3f}")
        
        plotting_integral(x_range, [sum_trap, sum_simp], args.int_method)
        
        
        
        