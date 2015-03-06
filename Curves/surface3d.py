# Ignacio Funke Prieto <igofunke@gmail.com> 
# Pablo Gordillo Alguacil <tutugordillo@gmail.com>
# Ignacio Gago Padreny <igago@ucm.es>

## License GPL

"""
  This module plots a surface given its parametrization and a geodesic curve in the surface. It is based on the differential equation obtained from the definition involving minimal energy. Alternatively plots a geodesic curve given a First Fundamental Form.

"""

import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab

#Given the first fundamental form, (u0,v0), (u0', v0'), the interval and t0 in that interval computes the geodesic curve with those properties 
def compute_geodesic(E,F,G,u0,v0,du0,dv0,interval,t0) :
    #I es the matrix of the First Fundamental Form
    I = sp.Matrix([[E,F],[F,G]])
    Iu = I.diff(u)
    Iv = I.diff(v)
    DU = sp.Matrix([du,dv])
    #rhs is the right hand side of the differential equation which defines the geodesic curve, based on the minimal energy definition of a geodesic curve
    rhs_sym_aux = (0.5 * sp.Matrix([DU.T * Iu * DU, DU.T * Iv * DU]).T - 
	       DU.T * (Iu * du + Iv * dv)) * I ** -1
    rhs_sym = [du, dv, rhs_sym_aux[0], rhs_sym_aux[1]]
    rhs_num = sp.lambdify((u,v,du,dv,t), rhs_sym, [{'ImutableMatrix' : np.array}, 'numpy'])
    rhs_eq = lambda Y,t : rhs_num(Y[0],Y[1],Y[2],Y[3],t)
    points = np.linspace(interval[0], interval[1], 1000)
    geodesic = odeint(rhs_eq, (u0,v0,du0,dv0), points)
    return geodesic

#Computes E,F,G given a geodesic curve
def compute_EFG(X) :
    x, y, z = X
    Xu = [x.diff(u), y.diff(u), z.diff(u)]
    Xv = [x.diff(v), y.diff(v), z.diff(v)]
    #E = <Xu, Xu>
    E = np.inner(Xu, Xu).simplify()
    #F = <Xu, Xv>
    F = np.inner(Xu, Xv).simplify()
    #G = <Xv, Xv>
    G = np.inner(Xv, Xv).simplify()
    return [E,F,G]

#Plots the surface and the geodesic curve together
def plot_surface_geodesic(X, interval_u, interval_v, geodesic, N) :
    u_grid = np.linspace(interval_u[0], interval_u[1], N)
    v_grid = np.linspace(interval_v[0], interval_v[1], N)
    u_mesh, v_mesh = np.meshgrid(u_grid, v_grid)
    X_num = sp.lambdify((u,v), X, [{'ImutableMatrix' : np.array}, 'numpy'])

    x, y, z = X_num(u_mesh, v_mesh)
    geodesic_u_mod = np.mod(geodesic[:,0], interval_u[1])
    geodesic_v_mod = np.mod(geodesic[:,1], interval_v[1])
    x_curve, y_curve, z_curve = X_num(geodesic_u_mod, geodesic_v_mod)

    mlab.plot3d(x_curve, y_curve, z_curve, representation='points')
    mlab.mesh(x,y,z)
    mlab.show()

if __name__ == "__main__" :

    #Common definitions for both (a) and (b)
    u, v, du, dv, t =  sp.symbols('u v du dv t')
    u0, v0 = 1, 1
    du0, dv0 = 1, 2
    interval_t = [1, 10]
    t0 = 2

    ## (a) Given a surface plot it together with a geodesic curve in that surface
    N = 1000
    interval_u = [0, 2*np.pi]
    interval_v = [0, 2*np.pi]
    X = [(sp.cos(u) + 2) * sp.cos(v),(sp.cos(u) + 2) * sp.sin(v), sp.sin(u)]
    E,F,G = compute_EFG(X)
    geodesic = compute_geodesic(E,F,G,u0,v0,du0,dv0,interval_t,t0)
  
    plot_surface_geodesic(X, interval_u, interval_v, geodesic, N)
    geodesic_u_mod = np.mod(geodesic[:,0], interval_u[1])
    geodesic_v_mod = np.mod(geodesic[:,1], interval_v[1])
    plt.plot(geodesic_u_mod, geodesic_v_mod, ',')
    plt.show()

    ## (b) Given E, F, G the components of the First Fundamental Form plots the geodesic curve
    #E = 1.0 / v**2 
    #F = 0
    #G = E 
    #geodesic = compute_geodesic(E,F,G,u0,v0,du0,dv0,interval_t,t0)
    #geodesic_u, geodesic_v = geodesic[:,0], geodesic[:,1]
    #plt.plot(geodesic_u, geodesic_v, ',')
    #plt.show()
