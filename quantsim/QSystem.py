from __future__ import division, print_function
import numpy as np
import utils
from transforms import *
from random import random
from Transform import Transform

class QSystem:
    def __init__(self, num_qubits=2, rand=None):
        if num_qubits == 0:
            raise ValueError("System must have >= 0 qubits")
        self.N = num_qubits
        self._state = np.array([0] * (2 ** num_qubits), dtype=complex)
        self._state[0] = 1
        self.r = rand if rand else random

    def single_gate(self, T, i):
        """Apply the transformation T to the ith qubit

        >>> qs = QSystem(2)
        >>> qs.single_gate(H, 0)
        >>> qs
        (0.707106781187+0j)|00> + 0j|01> + (0.707106781187+0j)|10> + 0j|11>
        >>> qs.single_gate(Z, 0)
        >>> qs
        (0.707106781187+0j)|00> + 0j|01> + (-0.707106781187+0j)|10> + 0j|11>
        >>> qs.single_gate(Z, 1)
        >>> qs
        (0.707106781187+0j)|00> + 0j|01> + (-0.707106781187+0j)|10> + 0j|11>
        """
        
        if i < 0 or i >= self.N:
            raise ValueError("Invalid qubit: " + str(i))
        transforms = [I] * self.N
        transforms[i] = T
        self.transform(transforms)

    def double_gate(self, T, i, j=None):
        """Apply the double-qubit transformation T to the ith and jth qubits,
        or ith and i+1th if j is not specified
        
        >>> qs = QSystem(3)
        >>> qs.single_gate(H, 0)
        >>> qs.double_gate(CNOT, 0)
        >>> qs
        (0.707106781187+0j)|000> + 0j|001> + 0j|010> + 0j|011> + 0j|100> + 0j|101> + (0.707106781187+0j)|110> + 0j|111>
        >>> qs = QSystem(3)
        >>> qs.single_gate(H, 0)
        >>> qs.double_gate(CNOT, 0, 2)
        >>> qs
        (0.707106781187+0j)|000> + 0j|001> + 0j|010> + 0j|011> + 0j|100> + (0.707106781187+0j)|101> + 0j|110> + 0j|111>
        """
        if i < 0 or i >= self.N - 1:
            raise ValueError("Invalid qubit: " + str(i))
        transforms = [I] * (self.N - 1)
        transforms[i] = T
        if j:
            S = utils.swap(self.N, i + 1, j)
            self._state = S.dot(self._state)
        self.transform(transforms)
        if j:
            S = utils.swap(self.N, i + 1, j)
            self._state = S.dot(self._state)

    def transform(self, transforms):
        """Transforms is formatted like [(H, 1), (CNOT, 2), (I, 1)]

        >>> qs = QSystem(2)
        >>> qs.transform([H, I])
        >>> qs
        (0.707106781187+0j)|00> + 0j|01> + (0.707106781187+0j)|10> + 0j|11>
        >>> qs.transform([CNOT])
        >>> qs
        (0.707106781187+0j)|00> + 0j|01> + 0j|10> + (0.707106781187+0j)|11>
        """
        total = 1
        for t in transforms:
            if t.N != 2 and t.N != 4:
                raise ValueError("Only one- or two-qubit gates allowed")
            total *= t.N
            utils.check_transform(t.T, t.N)
        if total != (2 ** self.N):
            raise ValueError("Wrong number of qubits. Expecting {0}, got {1}".format(self.N, total))
        T = utils.create_transform([t.T for t in transforms])
        self._state = T.dot(self._state)

    def measure(self, qubits):
        """Measure the specified qubits and update state accordingly.
        qubits is a list of numbers 0..N-1
        """
        qubits = list(set(qubits))
        for q in qubits:
            if q < 0 or q >= self.N:
                raise ValueError("Invalid qubit: {0}".format(q))
        measurements = []
        for q in qubits:
            measurements.append(self.measure_one(q))
        return measurements

    def measure_one(self, qubit):
        """Measure a single qubit and update the state accordingly.
        qubit is a number 0..N-1

        >>> qs = QSystem(2, lambda: 0)
        >>> qs.transform([H, H])
        >>> qs.measure_one(0)
        0
        >>> qs
        (0.707106781187+0j)|00> + (0.707106781187+0j)|01> + 0j|10> + 0j|11>
        >>> qs = QSystem(2, lambda: 1)
        >>> qs.transform([H, H])
        >>> qs.measure_one(0)
        1
        >>> qs
        0j|00> + 0j|01> + (0.707106781187+0j)|10> + (0.707106781187+0j)|11>
        >>> qs = QSystem(2)
        >>> qs.transform([H, I])
        >>> qs.transform([CNOT])
        >>> m1 = qs.measure_one(0)
        >>> m2 = qs.measure_one(1)
        >>> m1 == m2
        True
        """
        p0 = 0              # the probability that the qubit is in the 0 state
        shift = self.N - qubit - 1
        for i, s in enumerate(self._state):
            if (i >> shift) & 1 == 0:
                p0 += s * s.conjugate()
        if p0 > self.r():
            result = 0
        else:
            result = 1
        for i in range(len(self._state)):
            if (i >> shift) & 1 != result:
                self._state[i] = 0
        self._state = utils.normalize(self._state)
        return result

    @property
    def state(self):
        return self._state[:]

    def __repr__(self):
        return str(self)

    def __str__(self):
        st = ''
        state = 0
        for s in self.state:
            st += str(s) + '|' + "{0:{fill}{width}b}".format(state, fill='0', width=self.N) + '> + '
            state += 1
        return st[:-3]
