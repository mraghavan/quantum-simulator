from __future__ import division, print_function
import cmath as cm
import numpy as np
import utils
from transforms import *
from random import random

class Qubit:
    def __init__(self, states=2):
        if states == 0:
            raise ValueError("Qubit must have >= 0 states")
        self.N = states
        self.state = [0] * self.N
        self.state[0] = 1
        self.state = np.array(self.state, dtype=complex)
        self.state = utils.normalize(self.state)

    def get_state(self):
        return self.state

    def get_num_states(self):
        return self.N

    def transform(self, T):
        utils.check_transform(T, self.N)
        self.state = T.dot(self.state)

    def measure(self):
        r = random()
        total = 0
        for i, s in enumerate(self.state):
            total += (s ** 2)
            if total > r:
                self.state = [0] * self.N
                self.state[i] = 1
                self.state = np.array(self.state, dtype=complex)
                return i

    def __repr__(self):
        return str(self.state)

    def __str__(self):
        st = ''
        for i, s in enumerate(self.state):
            st += str(s) + '|' + str(i) + '> + '
        return st[:-3]

def main():
    pass

if __name__ == '__main__':
    main()

q = Qubit()
T = np.array([[1,1],[1,-1]]) / cm.sqrt(2)
TT = T.conjugate().transpose().dot(T)
I = np.array([[1,0],[0,0]], dtype=complex)
