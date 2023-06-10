# This program is will be used to test the xmethod function
# This function will be used to minimize continuous real nonlinear functions with real single variable
# with a given the inputs : function, initial interval point, end interval point, max-iteration

# Import the necessary modules
import math
import numpy as np
from scipy.optimize import fsolve

# Define a function that finds the intersection point between a line and a function
def intersection_with_min_y(line, func, x_range):
    # If the line is vertical (i.e., of the form "x = k")
    if line.startswith("x = "):
        # Parse the x-coordinate from the line
        _, k = line.split("=")
        k = float(k.strip())

        # Calculate the y-coordinate using the function
        y = func(k)

        # Return the intersection point
        return [k, y]

    # Otherwise, the line is of the form "y = mx + c"
    else:
        # Parse the slope and y-intercept from the line
        _, line_coeffs = line.split("=")
        slope, y_intercept = map(float, line_coeffs.split("x + "))

        # Define a function for the line
        line_func = lambda x: slope * x + y_intercept

        # Define a function that represents the difference between the given function and the line
        diff_func = lambda x: func(x) - line_func(x)

        # Find the roots of the difference function over the given range
        x_values = np.linspace(*x_range, 500)  # Create 500 points over the x_range
        roots = set()  # Use a set to avoid duplicates
        for val in x_values:
            root, = fsolve(diff_func, val)
            if min(x_range) <= root <= max(x_range):  # Check if the root is within the range
                roots.add(root)

        # If there are no roots, return False
        if not roots:
            return False

        # Evaluate the y-coordinate for each root and find the one with the minimum y-value
        min_root = min(roots, key=line_func)
        min_y_value = line_func(min_root)

        # Return the root and the corresponding y-value
        return [min_root, min_y_value]

# Define a function that creates an orthogonal line to a given line that passes through a given point
def create_orthogonal_line(point1, point2, point3):
    # Unpack the points
    a, b = point1
    c, d = point2
    e, f = point3

    # Calculate the slope for the line
    if c != a:  # To avoid division by zero
        slope = (d - b) / (c - a)
    else:  # If c == a, the line is vertical
        return None

    # If the slope is zero, the orthogonal line is vertical
    if slope == 0:
        # "x = k" form, where k is the x-coordinate of the point through which the line passes
        return f"x = {e}"

    # Calculate the slope and y-intercept for the orthogonal line
    orthogonal_slope = -1 / slope
    orthogonal_y_intercept = f - (orthogonal_slope * e)

    # Create the orthogonal line expression
    orthogonal_line_expression = f"y = {orthogonal_slope}x + {orthogonal_y_intercept}"

    return orthogonal_line_expression

# Define the xmethod function
def xmethod(func, x1, x2, max_iter):
    i = 0
    x_n = []
    while i <= max_iter:
        if func(x1) >= func(x2):
            A = [x1, func(x1)]
            B = [x2, func(x2)]
            C = [(x1+x2)/2, func((x1+x2)/2)]
            m_o = create_orthogonal_line(A, B, C)
            intersection = intersection_with_min_y(m_o, func, (x1, x2))
            x_n.append(intersection)
            if intersection:  # if there is an intersection point
                x1 = intersection[0]  # update x1 with the x coordinate of the intersection point
            i += 1
        else:
            A = [x1, func(x1)]
            B = [x2, func(x2)]
            C = [(x1+x2)/2, func((x1+x2)/2)]
            m_o = create_orthogonal_line(A, B, C)
            intersection = intersection_with_min_y(m_o, func, (x1, x2))
            x_n.append(intersection)
            if intersection:  # if there is an intersection point
                x2 = intersection[0]  # update x2 with the x coordinate of the intersection point
            i += 1

    return x_n


# Test the function

# Define the function
def f(x):
    return math.sin(2*x)


def func(x):
    return np.sin(2*x)


print(xmethod(f, -5, 5, 10))