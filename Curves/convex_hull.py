from __future__ import division
import math
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

class ConvexHull:

    #AUXILIARY FUNCTIONS

    def _compute_angle(self, o, x, y):
	ox = (x[0]-o[0], x[1]-o[1])
	xy = (y[0]-x[0], y[1]-x[1])
	norm_xy = np.linalg.norm(xy)
	if norm_xy == 0:
	    return 6.5
	arcos = np.dot(ox, xy)/np.linalg.norm(ox)/norm_xy
	if arcos > 1: arcos = 1.0
	elif arcos < -1: arcos = -1.0
	return math.acos(arcos)

    """
    Three points are a counter-clock-wise turn if ccw > 0, clockwise if ccw < 0 and collinear if ccw = 0. ccw is a determinant that gives twice the signed area of the triangle formed by p1, p2 and p3.
    """
    def _ccw(self, p1, p2, p3):
	return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])

    #MAIN FUNCTIONS

    def __call__(self, points, method='andrew'):
	if method == 'gift wrapping':
	    return self.gift_wrapping(points)
	elif method == 'graham scan':
	    return self.graham_scan(points)
	elif method == 'andrew':
	    return self.andrew_hull(points)
	#elif method == 'quickhull':
	#    return None

    def gift_wrapping(self, L):
	points = np.array(L)
	i0 = np.argmin(points[:,1]); i = i0
	v = (points[i][0]-1, points[i][1])
	hull = [points[i]]
	while(True):
	    k = np.argmin([self._compute_angle(v, points[i], p) for p in np.concatenate((points[0:i], points[i+1:]))])
	    if k >= i:
		k = k + 1
	    hull.append(points[k])
	    if i0 == k:
		break
	    i = k
	    v = hull[-2]
	return np.array(hull)

    def graham_scan(self, points):
	P = np.array(sorted(points, key=itemgetter(0), reverse=True))
	P = np.array(sorted(P, key=itemgetter(1)))
	angles = [self._compute_angle((P[0][0]-1,P[0][1]), P[0], p) for p in P[1:]]
	sorted_angles = [i[0] + 1 for i in sorted(enumerate(angles), key=lambda x:x[1])] 
	S = [P[0], P[sorted_angles[0]]]
	i = 1
	while i < len(points) - 1: 
	    if len(S) == 1:
		S.append(P[sorted_angles[i]])
		i = i + 1
	    elif (self._ccw(P[sorted_angles[i]], S[-2], S[-1]) > 0):
		S.append(P[sorted_angles[i]])
		i = i + 1
	    else:
		S.pop()
	S.append(S[0]) #for plotting purposes only
	return np.array(S)

    def andrew_hull(self, points):
	P = sorted(points)
	L_upper = [P[0], P[1]]
	L_lower = [P[-1], P[-2]]
	n = len(points)

	for i in xrange(2, n):
	    L_upper.append(P[i])
	    L_lower.append(P[n-i-1])
	    while len(L_upper) > 2 and self._ccw(L_upper[-3], L_upper[-2], L_upper[-1]) >= 0:
		L_upper.pop(-2)
	    while len(L_lower) > 2 and self._ccw(L_lower[-3], L_lower[-2], L_lower[-1]) >= 0:
		L_lower.pop(-2)

	#L_lower.pop(-1); 
	L_lower.pop(0)
	L_upper.extend(L_lower)
	return L_upper

    def plot(self, points, hull):
	points_np = np.array(points)
	hull_np = np.array(hull)
	
	x_lim = max(np.absolute(points_np[:,0])) + 2
	y_lim = max(np.absolute(points_np[:,1])) + 2
	
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect=1)
	ax.set_xlim(-x_lim, x_lim)
	ax.set_ylim(-y_lim, y_lim)

	plt.plot(points_np[:,0], points_np[:,1], color='blue', marker='o', linestyle='None')
	plt.plot(hull_np[:,0], hull_np[:,1])
	fig.canvas.draw()
	plt.show()
