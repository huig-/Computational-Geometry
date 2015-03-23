import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from bezier_curve import CurvadeBezier
import numpy as np
#from mis_bonitas_geodesicas import Geodesica
#Line2D for curves

class DrawPoints:
    def __init__(self, fig, ax, bezier_curve):
	self.ax = ax
	self.fig = fig

	#A list of points
	self.circles = []

	self.bezier_curve = bezier_curve

	self.control_polygon = None

	self.exists_touched_circle = False

	#Conects an event with a procedure
	#It is a process running in background, by assigning it to a variable it can be killed
	self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
	#Always the mouse is moved
	self.cid_move = fig.canvas.mpl_connect('motion_notify_event', self.on_move)
	#On button release
	self.cid_release_button = fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
	for circle in self.circles:
	    #Returns two arguments, first one boolean
	    contains, attr = circle.contains(event)    
	    if contains:
		self.touched_circle = circle
		self.exists_touched_circle = True
		self.pressed_event = event
		self.touched_x0, self.touched_y0 = circle.center
		return

	self.bezier_curve.add_point_to_Polygon((event.xdata, event.ydata))
	c = Circle((event.xdata, event.ydata), 0.5)
	self.circles.append(c)
	self.ax.add_patch(c)

	if self.control_polygon == None:
	    self.control_polygon = Polygon(self.bezier_curve.polygon, closed=False, fill=False, ec='m', ls='dotted')
	    self.ax.add_patch(self.control_polygon)
	
	self.control_polygon.set_xy(self.bezier_curve.polygon)

	if (self.bezier_curve.N >= 1) :
	    #self.bezier_curve.plot_bezier()
	    self.bezier_curve.plot_casteljau()
	self.fig.canvas.draw()

    def on_move(self, event):
	if self.exists_touched_circle:
	    dx = event.xdata - self.pressed_event.xdata
	    dy = event.ydata - self.pressed_event.ydata

	    self.touched_circle.center = self.touched_x0 + dx, self.touched_y0 + dy
	    self.bezier_curve.set_polygon(np.array([c.center for c in self.circles]))
	    self.control_polygon.set_xy(self.bezier_curve.polygon)
	    if (self.bezier_curve.N > 1):
		#self.bezier_curve.plot_bezier()
		self.bezier_curve.plot_casteljau()
	    self.fig.canvas.draw()

    def on_release(self, event):
	self.exists_touched_circle = False
	    	

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    bezier_curve = CurvadeBezier(ax,fig)

    draw_points = DrawPoints(fig, ax, bezier_curve)
    plt.show()
