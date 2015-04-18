
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

class IntersectionBezier:

    def __call__(self, P, Q, epsilon):
        self.P = P
        self.Q = Q
        self.epsilon = epsilon

        cuts = self._intersect(P,Q,epsilon)
        return self._unique(cuts)

    # Removes the repeated elements in cuts along axis 0
    def _unique(self,cuts):
        ncols = cuts.shape[1]
        dtype = cuts.dtype.descr * ncols
        struct = cuts.view(dtype)

        uniq = np.unique(struct)
        uniq = uniq.view(cuts.dtype).reshape(-1, ncols)
        return uniq

    # Computes the intersection of the Bezier Curves given by their control points P and Q.
    def _intersect(self,P,Q,epsilon):
        if self._intersect_P_Q(P, Q) == False:
            return np.array([]).reshape(0,2)

        n = P.shape[0]
        m = Q.shape[0]
        if 1/8.0*n*(n-1)*np.amax(np.absolute(self._diferences(P,2))) > epsilon:
            P1, P2 = self._compute_composite_polygon(P)
            intersection_1 = self._intersect(P1, Q, epsilon)
            intersection_2 = self._intersect(P2, Q, epsilon)
            return np.vstack((intersection_1, intersection_2))
        elif 1/8.0*m*(m-1)*np.amax(np.absolute(self._diferences(Q,2))) > epsilon:
            Q1, Q2 = self._compute_composite_polygon(Q)
            intersection_1 = self._intersect(P, Q1, epsilon)
            intersection_2 = self._intersect(P, Q2, epsilon)
            return np.vstack((intersection_1, intersection_2))
        else:
            return self._intersect_segments(P[0,:], P[-1,:], Q[0,:], Q[-1,:])

	# Returns the minimum area rectangle containing the control points P
    def _box(self, P):
        return np.amin(P, axis=0), np.amax(P, axis=0)

    # Returns whether the two rectangles box_P and box_Q intersect or not
    def _intersect_boxes(self, box_P, box_Q):
        [p_xmin, p_ymin], [p_xmax, p_ymax] = box_P
        [q_xmin, q_ymin], [q_xmax, q_ymax] = box_Q
        dx = min(p_xmax, q_xmax) - max(p_xmin, q_xmin)
        dy = min(p_ymax, q_ymax) - max(p_ymin, q_ymin)
        if (dx>=0) and (dy>=0):
            return True
        return False

	# Tests if the boxes associated with P and Q intersect
    def _intersect_P_Q(self, P, Q):
        return self._intersect_boxes(self._box(P),self. _box(Q))

    # Computes the differences of order n to the control points P
    def _diferences(self, P, n):
        if n <= 0:
            return P
        
        d = P[1:]-P[0:-1]
        return self._diferences(d,n-1)

    # Computes the composite polygon of the control polygon P
    def _compute_composite_polygon(self, P):
        if P.size == 0:
            return np.empty([0,2]), np.empty([0,2])

        next_P = 0.5*P[0:-1,:] + 0.5*P[1:,:]
        head_P1 = P[0,:]
        head_P2 = P[-1,:]
        P1,P2 = self._compute_composite_polygon(next_P)
        return np.insert(P1, 0, [head_P1], axis=0), np.insert(P2, 0, [head_P2], axis=0)

    # Plots the two Bezier Curves set in __call__ and paints their intersections
    def plot(self, k=3):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)

        self._plotting_color = 'r'
        self._plot_bezier(self.P, self.epsilon, k)

        self._plotting_color = 'k'
        self._plot_bezier(self.Q, self.epsilon, k)

        I = self._intersect(self.P, self.Q, self.epsilon)
        self._plot_intersections(I)

        self.fig.canvas.draw()
        plt.show()

    # Plots a Bezier curve given its control polygon
    def _plot_bezier(self, P, epsilon, k):
        n = P.shape[0]

        if k == 0:
            self._plot_polygon(P)
        elif 1/8.0*n*(n-1)*np.amax(np.absolute(self._diferences(P,2))) < epsilon:
            self._plot_segment(P[0,:],P[-1,:])    
        else:
            P1, P2 = self._compute_composite_polygon(P)
            self._plot_bezier(P1,k-1,epsilon)
            self._plot_bezier(P2,k-1,epsilon)

    # Plots a control polygon
    def _plot_polygon(self,P):
        polygon = Polygon(P,closed=False,fill=False,color=self._plotting_color)
        self.ax.add_patch(polygon)

    # Plots a fixed segment
    def _plot_segment(self,A,B):
        segment = Polygon(np.array([A,B]),closed=False,fill=False,color=self._plotting_color)
        self.ax.add_patch(segment)

    # Plots the intersections previously computed
    def _plot_intersections(self, I):
        for i in range(I.shape[0]):
            self.ax.add_patch(Circle(I[i,:], 0.5))

    # Computes the intersection of two segments.
    def _intersect_segments(self, p0, q0, p1, q1):
        v_p0q0 = q0 - p0
        v_p0p1 = p1 - p0
        v_p0q1 = q1 - p0
        v_p1q1 = q1 - p1
        v_p1q0 = q0 - p1
        if (np.linalg.det(np.array([v_p0q0, v_p0p1])) * np.linalg.det(np.array([v_p0q0, v_p0q1])) >= 0) or (np.linalg.det(np.array([v_p1q1, -v_p0p1]))*np.linalg.det(np.array([v_p1q1,v_p1q0])) >=0):
            return np.array([]).reshape(0,2)

        p0_proy = [p0[0], p0[1], 1]
        q0_proy = [q0[0], q0[1], 1]
        p1_proy = [p1[0], p1[1], 1]
        q1_proy = [q1[0], q1[1], 1]

        r0 = np.cross(p0_proy, q0_proy)
        r1 = np.cross(p1_proy, q1_proy)

        x, y, z = np.cross(r0,r1)
        return np.array([[x*1.0/z, y*1.0/z]])

if __name__ == '__main__':

    intersection_bezier = IntersectionBezier() 
    #P = np.array([[-10,-10], [0,20], [10,-10]])
    #Q = np.array([[10,10], [0, -20], [-10,10]])
    P = np.random.randint(-20, 20, (3+1, 2))
    Q = np.random.randint(-20, 20, (3+1,2))
    cuts = intersection_bezier(P, Q, epsilon=0.01)
    print(cuts)

    intersection_bezier.plot()
