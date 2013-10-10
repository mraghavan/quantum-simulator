from __future__ import division, print_function
import numpy as np
import cmath as cm
from math import sin, cos
from Transform import Transform

# SINGLE QUBIT GATES
H = Transform(
        np.array([[1,1],
                  [1,-1]]) / cm.sqrt(2),
        'H')
NOT = Transform(
        np.array([[0,1],
                  [1,0]]),
        'NOT')
Z = Transform(
        np.array([[1,0],
                  [0,-1]]),
        'Z')

I = Transform(
        np.array([[1,0],
                  [0,1]]),
        'I')

def U(theta):
    return Transform(
            np.array([[cos(theta),-sin(theta)],
                      [sin(theta),cos(theta)]]),
            'U(' + str(theta) + ')')

# TWO QUBIT GATES
CNOT = Transform(
        np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]]),
        'CNOT')
