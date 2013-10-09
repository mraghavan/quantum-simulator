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
        self.N = num_qubits
        self._state = np.array([0] * (2 ** num_qubits), dtype=complex)
        self._state[0] = 1

    def single_gate(self, T, i):
        if i < 0 or i >= self.N:
            raise ValueError("Invalid qubit: " + str(i))
        transforms = [(I, 1)] * self.N
        transforms[i] = (T, 1)
        self.transform(transforms)

    def double_gate(self, T, i):
        if i < 0 or i >= self.N - 1:
            raise ValueError("Invalid qubit: " + str(i))
        transforms = [(I, 1)] * (self.N - 1)
        transforms[i] = (T, 2)
        self.transform(transforms)

    def transform(self, transforms):
        """Transforms is formatted like [(H, 1), (CNOT, 2), (I, 1)]"""
        sum = 0
        for transform, size in transforms:
            if size != 1 and size != 2:
                raise ValueError("Only one- or two-qubit gates allowed")
            sum += size
            utils.check_transform(transform, 2 ** size)
        if sum != self.N:
            raise ValueError("Wrong number of qubits")
        T = self.create_transform([t[0] for t in transforms])
        self.state = T.dot(self.state)

    @property
    def state(self):
        return self._state[:]

    def create_transform(self, transforms):
        """Transforms is formatted like [H, CNOT, I]"""
        S = transforms[0]
        for T in transforms[1:]:
            S = np.kron(S, T)
        return S

    def __repr__(self):
        return str(self)

    def __str__(self):
        st = ''
        state = 0
        for s in self.state:
            st += str(s) + '|' + "{0:{fill}{width}b}".format(state, fill='0', width=self.N) + '> + '
            state += 1
        return st[:-3]
