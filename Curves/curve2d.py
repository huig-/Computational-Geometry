# Ignacio Funke Prieto <igofunke@gmail.com> 
# Pablo Gordillo Alguacil <tutugordillo@gmail.com>
# Ignacio Gago Padreny <igago@ucm.es>

## License GPL

"""
    This module plots a 2d-curve given its curvature, the interval in which the	    curvature is defined and the initial conditions in a point of the interval,	    not necessarily an endpoint.

    Ex: k(s)=1; s0=0; x0,y0,dx0,dy0=0,0,1,0; interval_right,interval_left=-pi,pi
    Ex: k(s)=s*(s-3)*(s+5); s0=6; x0,y0,dx0,dy0=0,0,1,0; interval_rigth,interval_left=-10,10
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#Curvature 
def k(s):
    return s*(s-3)*(s+5)

#Right hand side of the differential equation that defines the curve
def rhs_eqs(Y,s):
    x,y,dx,dy = Y
    return [dx,dy,-k(s)*dy,k(s)*dx]

def main():
    #A point s0 in [interval_left,interval_right] and the values x0,y0,dx0,dy0 in that point
    s0 = 10
    x0,y0 = 0,0
    dx0,dy0 = 1,0
    init_cond = [x0,y0,dx0,dy0]
    #[interval_left,interval_right] in which the curvature is defined 
    interval_left = -10;
    interval_right = 10;
    #Step for defining the grid	
    step = 0.01;
    #Function linspace takes the number of points you wish to divide the interval, we compute it using the step previously defined
    interval_2 = np.linspace(s0,interval_right, (interval_right-s0)/step)
    interval_1 = np.linspace(s0,interval_left,(s0-interval_left)/step);
    #We use odeint for resolving the differential equation system in each interval
    solution_2 = odeint(rhs_eqs,init_cond,interval_2)
    solution_1 = odeint(rhs_eqs,init_cond,interval_1)
    #Concatenate the solution of each interval and take out the element repeated
    curve_x = np.hstack((solution_1[::-1,0],solution_2[1:,0]))
    curve_y = np.hstack((solution_1[::-1,1],solution_2[1:,1]))
    #Plot the curve
    plt.plot(curve_x,curve_y)
    plt.show()

if __name__ == "__main__":
    main()
