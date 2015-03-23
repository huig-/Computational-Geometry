# Ignacio Funke Prieto <igofunke@gmail.com>
# Pablo Gordillo Alguacil <tutugordillo@gmail.com>
# Ignacio Gago Padreny <igago@ucm.es>

## License GPL

"""
  Given two curves and their respective intervals in which they are defined computes numerically the probability that their signatures are the same in the interval or a subset of it. If the probability is high enough then the curves are equivallent in some subset.

"""
import sympy as sp
import numpy as np
from scipy.spatial.distance import cdist

def near_enough(x, epsilon):
    return x <= epsilon

#Takes as input the curves, their intervals, N as the number of points to evaluate numerically the signature and epsilon as the precission
def compare_curves(curve_1, curve_2, range_1, range_2, N, epsilon):

    #Defining the elements of the curves to compare
    x0, y0 = curve_1[0], curve_1[1]
    x1, y1 = curve_2[0], curve_2[1]

    #Compute first and second derivatives from both curves
    dx0, dy0 = x0.diff(t), y0.diff(t)
    d2x0, d2y0 = dx0.diff(t), dy0.diff(t)

    dx1, dy1 = x1.diff(t), y1.diff(t)
    d2x1, d2y1 = dx1.diff(t), dy1.diff(t)

    #Compute curvature k0, k1 from curve_1 and curve_2 
    norm_dx0 = sp.sqrt(dx0**2 + dy0**2)
    k0 = (dx0 * d2y0 - dy0 * d2x0) / norm_dx0**3
    k0 = k0.simplify()

    norm_dx1 = sp.sqrt(dx1**2 + dy1**2)
    k1 = (dx1 * d2y1 - dy1 * d2x1) / norm_dx1**3
    k1 = k1.simplify()

    #Compute dk0/ds and dk1/ds for signature
    dk0 = (k0.diff(t) / norm_dx0).simplify()
    dk1 = (k1.diff(t) / norm_dx1).simplify()

    #Uncomment the following two lines to plot signature
    #sp.plotting.plot_parametric(k0,dk0,(t, range_1[0], range_1[1]))
    #sp.plotting.plot_parametric(k1,dk1,(t, range_2[0], range_2[1]))

    #Sympy function that permits working numerically
    num_k0 = np.vectorize(sp.lambdify(t, k0, [{'ImutableMatrix' : np.array}, 'numpy']))
    num_k1 = np.vectorize(sp.lambdify(t, k1, [{'ImutableMatrix' : np.array}, 'numpy']))
    num_dk0 = np.vectorize(sp.lambdify(t, dk0, [{'ImutableMatrix' : np.array}, 'numpy']))
    num_dk1 = np.vectorize(sp.lambdify(t, dk1, [{'ImutableMatrix' : np.array}, 'numpy']))

    #Numerical values of the components of the signature of each curve in its interval
    t0 = np.linspace(range_1[0],range_1[1],N)
    values_k0 = num_k0(t0)
    values_dk0 = num_dk0(t0)

    t1 = np.linspace(range_2[0],range_2[1],N)
    values_k1 = num_k1(t1)
    values_dk1 = num_dk1(t1)

    #Compare both signatures
    m_k0 = zip(values_k0, values_dk0)
    m_k1 = zip(values_k1, values_dk1)
    distances = cdist(m_k0,m_k1)

    #Only consider the minimun value for each column in matrix distances (a point may only be compared with another point)
    posible_points = np.nanmin(distances,axis=0)
    count = sum(1 for p in posible_points if near_enough(p,epsilon))

    #Print the percentage 
    print count * 1.0 / len(posible_points)

##
if __name__ == "__main__" :
    
    #Variables
    t = sp.symbols('t')

    curve_1 = [t-1, t]
    curve_2 = [2*t-5, 3-t]
    range_1 = [0,1]
    range_2 = [-1, 0]
    N = 1000
    epsilon = 0.001
    compare_curves(curve_1, curve_2, range_1, range_2, N, epsilon)
