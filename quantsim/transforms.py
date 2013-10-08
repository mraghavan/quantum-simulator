from __future__ import division, print_function
import numpy as np
import cmath as cm
from math import sin, cos

H = np.array([[1,1],[1,-1]]) / cm.sqrt(2)
NOT = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

def U(theta):
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
