# This algorithm is used to find the minima of a continuous multivariate nonlinear function using hyperplanes that
# will be used in a iterative way to minimse the function in steps that are adapated as the 
# alogrithm progresses.
# The algorithm takes as an input the the non linear function (func), two points (x1,x2) 
# and the number of iterations (niter).
# The algorithm returns the global minimum of the function (xmin) and the value of the function at the minimum (fmin).
# in the specified interval, we allow as well for the user to specify the step size (step) 
# and the seed for the random generator (seed).
# This program was runned using an Ubuntu distribution of Linux and Python 3.10.6

import matplotlib
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad, numpy as anp
matplotlib.use('TkAgg')

def check_sol(func, K, bounds, seed=23):
    """Check if there is a solution to the equation func(x_1,...,x_n) = K.

    Arguments:
    - func: a function object that accepts a 1D numpy array and returns a real number.
    - K: the real number to which the function should be equal.
    - bounds: a list of tuples specifying the lower and upper bounds for each dimension.
    - seed: a seed for the random number generator.

    Returns:
    - True and a solution point (as a 1D numpy array) if a solution is found, False otherwise.
    """
    # Define a new function that is the absolute difference between the original function and K
    def diff_func(x):
        return np.abs(func(x) - K)

    # Use differential evolution to find the minimum of this new function
    result = differential_evolution(diff_func, bounds, seed=seed)

    # If the function's minimum is zero, return True and the solution point
    if result.fun < 1e-6:  # 1e-6 to account for numerical errors
        return [True, result.x]
    else:  # If the function's minimum is not zero, return False
        return [False, None]

def bounder(a,b):
    """
    Generates a list of lists of the form [[a1,b1],[a2,b2],...,[an,bn]] where a1 <= b1, a2 <= b2, ..., an <= bn.
    """
    I = []
    for i in range(len(a)):
        I.append([min(a[i],b[i]),max(a[i],b[i])])
    return I

def comparsion(func,a,b):
    if func(a) <= func(b):
        return [a,func(a)]
    else:
        return [b,func(b)]

def bars(func,a,b,niter,step=2, DIFFERENTIAL=False):
    """
    Finds a global minimum in a given interval of a continuous multivariate nonlinear function using hyperplanes
    """
    i = 0
    x_n = []
    x_n.append(comparsion(func,a,b))
    bounds = bounder(a,b)  # Use the function to generate the bounds
    while i <= niter:
        solution = check_sol(func, x_n[-1][-1]- step**(1-i), bounds)
        if solution[0]:
            print(f"Numerical solution found, the hyperplane is intersecting the function with step {step**(1-i)}" + f" - {i}")
            x_n.append([check_sol(func, x_n[-1][-1]- step**(1-i), bounds)[1],func(check_sol(func, x_n[-1][-1]- step**(1-i), bounds)[1])])
        else:
            print(f"Numerical solution not found, the hyperplane is not intersecting the function with step {step**(1-i)}" + f" - {i}")
            i += 1
    
    print("Solutions: \n", x_n)

    # Check the gradient at each point

    if DIFFERENTIAL == True:
        def func(x):
            return -20 * anp.exp(-0.2 * anp.sqrt(0.5 * (x[0]**2 + x[1]**2))) - anp.exp(0.5* (anp.cos(2 * anp.pi * x[0] ) + anp.cos(2 * anp.pi * x[1]))) + anp.exp(1) + 20 
            #put your function here, has to be the same as the one used above
        
        func_grad = grad(func)
        gradients = []
        for solution in x_n:
            point = solution[0]
            point = np.array(point, dtype=float)
            gradient_at_point = func_grad(point)
            gradients.append(gradient_at_point)

        print("Gradient vectors: \n", gradients)
        print("Norms of the gradient vectors: \n", [np.linalg.norm(gradient) for gradient in gradients])



    # Visualize the solution path

    # If the problem is 2D, visualize using a contour plot
    if len(bounds) == 1:
    # Create an array of x values based on the bounds
        x = np.linspace(bounds[0][0], bounds[0][1], 400)
        # Calculate the corresponding y values for each x
        y = func([x])

        # Make a line plot
        plt.figure(figsize=(10,6))
        plt.plot(x, y, 'b-')

        # Mark all the points used by the algorithm
        for sol in x_n:
            plt.plot(sol[0], sol[1], 'r.', markersize=10)

        plt.show()

    # If the problem is 3D, visualize using a 3D plot
    if len(bounds) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)

        # Calculate the corresponding Z values for each pair of x and y
        Z = func([X,Y])

        # Make a surface plot
        ax.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.3)

        # Mark the solution path
        sol_x = [pt[0][0] for pt in x_n]
        sol_y = [pt[0][1] for pt in x_n]
        sol_z = [pt[1] for pt in x_n]
        ax.plot(sol_x, sol_y, sol_z, 'r.-')

        plt.show()

    # If it is 4D function, visualize using a 3D plot
    if len(bounds) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        z = np.linspace(bounds[2][0], bounds[2][1], 100)
        X, Y, Z = np.meshgrid(x, y, z)

        # Mark the solution path
        sol_x = [pt[0][0] for pt in x_n]
        sol_y = [pt[0][1] for pt in x_n]
        sol_z = [pt[1] for pt in x_n]
        ax.plot(sol_x, sol_y, sol_z, 'r.-')

        plt.show()


    return x_n[-1][0], x_n[-1][1]


def func(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0] ) + np.cos(2 * np.pi * x[1]))) + np.exp(1) + 20 
    #check the function used in the algorithm

#use my algorithm
bars(func,[-5,-5],[5,5], 2, 10, DIFFERENTIAL=True)


#use scipy algorithm
from scipy.optimize import minimize
x0 = np.array([-5, -5]) #initial guess
res = minimize(func, x0)
print(res.x,func(res.x))