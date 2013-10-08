from __future__ import division, print_function
import cmath as cm
import numpy as np
import utils
from transforms import *
from random import random
from Qubit import Qubit

class QSystem:
    def __init__(self, num_qubits=2):
        if num_qubits == 0:
            raise ValueError("System must have >= 0 qubits")
        self.qubits = [0] * num_qubits
        for i in range(num_qubits):
            self.qubits[i] = Qubit(2)

    @property
    def num(self):
        return len(self.qubits)

    def transform(self, T, first, second=None):
        if first < 0 or first >= self.num:
            raise ValueError("Invalid qubit " + str(first))
        if second and (second < 0 or second >= self.num):
            raise ValueError("Invalid qubit " + str(second))
        if not second:              # single qubit gate
            self.qubits[first].transform(T)

    @property
    def state(self):
        base = self.qubits[0].state
        for q in self.qubits[1:]:
            base = utils.tensor(base, q.state)
        return base

    def __repr__(self):
        s = ''
        for q in self.qubits:
            s += repr(q)
            s += '\n'
        return s[:-1]

    def __str__(self):
        st = ''
        state = 0
        for s in self.state:
            st += str(s) + '|' + "{0:b}".format(state) + '> + '
            state += 1
        return st[:-3]

qs = QSystem()
