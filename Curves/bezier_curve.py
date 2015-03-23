import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

class CurvadeBezier :
    
    def __init__(self,ax,fig):
        self.polygon = None
        self.N = -1 
        self.num_points = 100
        self.t = np.linspace(0,1,self.num_points)
        
	#Dictionary to avoid recomputing values
        self.berstein = {}

        self.curve = None
        
	#For using the formulae involving berstein 
        self.curve_x = None
        self.curve_y = None

	#For using the formulae involving Casteljau algorithm
        self.curve_casteljau_x = None
        self.curve_casteljau_y = None

        self.ax = ax
        self.fig = fig

    def points(self):
        return self.N+1

    #Computes only the necessary berstein polynomials
    def _compute_berstein(self):
        for i in range(self.N+1):
            if (self.N,i) not in self.berstein:
                self.berstein[(self.N,i)] = self._coef_berstein(self.N,i)

    def _coef_berstein(self, n, i):
        return binom(n,i) * self.t ** i * (1-self.t) ** (n-i)

    def _compute_casteljau(self):
        self.casteljau_x = np.empty([self.N+1,self.num_points])
        self.casteljau_y = np.empty([self.N+1,self.num_points])

        for i in range(self.N+1-1):
            self.casteljau_x[i,:] = (1-self.t)*self.polygon[i,0] + self.t*self.polygon[i+1,0]
            self.casteljau_y[i,:] = (1-self.t)*self.polygon[i,1] + self.t*self.polygon[i+1,1]
        for k in range(2,self.N+1):
            for i in range(self.N+1 - k):
                self.casteljau_x[i,:] = (1-self.t)*self.casteljau_x[i,:] + self.t*self.casteljau_x[i+1,:]
                self.casteljau_y[i,:] = (1-self.t)*self.casteljau_y[i,:] + self.t*self.casteljau_y[i+1,:]

        self.curve_casteljau_x = self.casteljau_x[0,:]
        self.curve_casteljau_y = self.casteljau_y[0,:]

    #Computes the bezier curve using berstein polynomials
    def compute_curve(self):
        self.curve_x = sum(self.polygon[i,0]*self.berstein[(self.N,i)] for i in range(self.N+1))
        self.curve_y = sum(self.polygon[i,1]*self.berstein[(self.N,i)] for i in range(self.N+1))

    def add_point_to_Polygon(self, point):
        if (self.polygon == None):
            self.polygon = np.array([point])
        else:
            self.polygon = np.append(self.polygon,[point],axis=0)
        self.N = self.N + 1

    def set_polygon(self, polygon):
	self.polygon = polygon

    #Computes and plots the bezier curve based on berstein polynomials
    def plot_bezier(self) :
        self._compute_berstein()
        self.compute_curve()
        if self.curve == None:
            self.curve = Line2D(self.curve_x,self.curve_y)
            self.ax.add_line(self.curve)
        else:
            self.curve.set_xdata(self.curve_x)
            self.curve.set_ydata(self.curve_y)
            self.fig.canvas.draw()

    #Computes and plots the bezier curve based on Casteljau algorithm
    def plot_casteljau(self):
        self._compute_casteljau()
        if self.curve == None:
            self.curve = Line2D(self.curve_casteljau_x,self.curve_casteljau_y)
            self.ax.add_line(self.curve)
        else:
            self.curve.set_xdata(self.curve_casteljau_x)
            self.curve.set_ydata(self.curve_casteljau_y)
            self.fig.canvas.draw()
