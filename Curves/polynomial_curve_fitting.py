from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



def polynomial_curve_fitting(points, knots, method, L=0, libraries=False,
                             num_points=100, degree=None):    
    '''
       Fits planar curve to points at given knots. 

       Arguments:
           points -- coordinates of points to adjust (x_i, y_i) given by a numpy array of shape (N, 2)
           knots -- strictly increasing sequence at which the curve will fit the points, tau_i
               It is given by a np.array of shape N, unless knots='chebyshev', in this case
                   N Chebyshev's nodes between 0 and 1 will be used instead of tau.
           method -- one of the following: 
               'newton' computes the interpolating polynomial curve using Newton's method. 
               'least_squares' computes the best adjusting curve in the least square sense,
                   i.e., min_a ||Ca - b||**2 + L/2 ||a||**2
           L -- regularization parameter
           libraries -- If False, only numpy linear algebra operations are allowed. 
               If True, any module can be used. In this case, a very short and fast code is expected
           num_points -- number of points to plot between tau[0] and tau[-1]
           degree -- degree of the polynomial. Needed only if method='least_squares'.
                     If degree=None, the function will return the interpolating polynomial.

       Returns:
           numpy array of shape (num_points, 2) given by the evaluation of the polynomial
           at the evenly spaced num_points between tau[0] and tau[-1]
    '''
    if knots == 'chebyshev':
	knots = chebyshev_knots(0, 1, points.shape[0])
    if method == 'newton':
        return newton_polynomial(points,knots,num_points,libraries)
    elif method == 'least_squares':
	return least_squares_fitting(points, knots, degree, num_points, L, libraries)    
    

def polynomial_curve_fitting1d(points, knots, method, L=0, libraries=False,
                             num_points=100):    
    pass


def newton_polynomial(x, tau, num_points=100, libraries=False):    
    '''
    Computes de Newton's polynomial interpolating values x at knots tau
    x: numpy array of size n; points to interpolate
    tau: numpy array of size n; knots tau[0] < tau[1] < ... < tau[n-1]
    num_points: number of points at which the polynomial will be
                evaluated

    libraries: False means only linear algebra can be used
               True means every module can be used.

    returns:
       numpy array of size num_points given by the polynomial 
       evaluated at np.linspace(tau[0], tau[1], num_points)

    Maximum cost allowed: 5,43 s at lab III computers
            degree = n - 1 = 9
            num_points = 100
    '''
    
    if libraries == False:
        n = x.shape[0]
        m = x.ndim

        ## Compute divided differences ##
        if m > 1 : tau = tau[:,np.newaxis]
        aux = x
        divided_differences = np.array([aux[0]])
        for k in range(1,n):
            aux = np.divide(aux[1:] - aux[0:-1],(tau[k:] - tau[0:-k])*1.0)
            divided_differences = np.append(divided_differences,[aux[0]],axis=0)

        ## Compute polynomial ##
        t = np.linspace(tau[0],tau[-1],num_points)
        product = np.multiply
        # A few tweaks to work with dimensions greater than 1.
        # It allows using the same code for m=1 but with matrices.
        if m>1:
            t = t[:,np.newaxis]
            divided_differences = divided_differences[:,np.newaxis,:]
            product = np.dot
            
        # Compute the polynomial using Horner's fast evalution method.
        polynomial = divided_differences[-1]
        # The first iteration is slightly different when working with matrices.
        if n>1: 
            polynomial = divided_differences[-2] + product((t - tau[-2]),polynomial)
        for k in range(n-3,-1,-1):
            polynomial= divided_differences[k]+(t-tau[k])*polynomial

        return polynomial

    else:
        t = np.linspace(tau[0],tau[-1],num_points)
        coef = np.polyfit(tau,x,x.shape[0]-1)
        if x.ndim > 1:
            polynomial = np.empty((t.shape[0],x.shape[1]))
            for k in range(0,x.shape[1]):
                polynomial[:,k] = np.polyval(coef[:,k],t) 
        else:
            polynomial = np.polyval(coef,t)

        return polynomial

def eval_poly(t, coefs, tau=None):    
    pass
        
def least_squares_fitting(points, knots, degree, num_points, L=0, libraries=True):    

    if degree == None:
	degree = points.shape[0]-1
    if libraries:
	coeffs = np.polyfit(knots, points, degree)
	t = np.linspace(knots[0], knots[-1], num_points)
	polynomial = np.empty((t.shape[0], points.shape[1]))
	for k in xrange(points.shape[1]):
	    polynomial[:,k] = np.polyval(coeffs[:,k],t)
	return polynomial
    else:
	n = knots.shape[0]

	C = np.vander(knots, degree+1)

	if n == (degree+1):
	    coeffs = np.linalg.solve(np.add(C, L*0.5*np.identity(degree+1)), points)
	else:
	    coeffs = np.linalg.solve(np.add(np.dot(np.transpose(C), C), L*0.5*np.identity(degree+1)), np.dot(np.transpose(C), points))

	t = np.linspace(knots[0], knots[-1], num_points)
	polynomial = np.empty((t.shape[0], points.shape[1]))
	for k in xrange(points.shape[1]):
	    polynomial[:,k] = np.polyval(coeffs[:,k],t)
	return polynomial

def chebyshev_knots(a, b, n):
    tau = np.empty(n)
    for i in xrange(1,n+1):
	tau[i-1] = (a+b - (a-b)*np.cos((2*i-1)*np.pi/(2.*n)))*0.5
    return tau
