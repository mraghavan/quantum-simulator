from __future__ import division, print_function
import numpy as np
import cmath as cm
import utils

class Transform:
    """Class to represent a unitary transformation"""
    def __init__(self, T, name=None):
        if T.shape[0] != T.shape[1]:
            raise ValueError("Tranformation must be square")
        self.N = T.shape[0]
        utils.check_transform(T, self.N)
        self.T = T
        self.name = name

    def __str__(self):
        s = ''
        if self.name:
            s = self.name + ':\n'
        return s + repr(self.T.shape) + '\n' + repr(self.T)
    def __repr__(self):
        s = ''
        if self.name:
            s = ',\n\'' + self.name + '\''
        return 'Transform(\n' + repr(self.T) + s + ')'
