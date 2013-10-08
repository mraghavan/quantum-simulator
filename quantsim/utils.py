from __future__ import division, print_function
import numpy as np
import cmath as cm

def normalize(v):
    return v / magnitude(v)

def magnitude(v):
    return cm.sqrt(v.dot(v))

def check_transform(T, N):
    """Takes advantage of the fact that if T is unitary, then T x T.conjugate.transpose is the identity"""
    if type(T) != np.ndarray:
        raise ValueError("Transformation must be an array")
    if T.shape != (N, N):
        raise ValueError("Transformation must be %sx%s" % (N, N))
    I = identity(N)
    if not np.allclose(T.conjugate().transpose().dot(T), I):
        raise ValueError("Transformation must be unitary")

def identity(N):
    l = [0] * (N ** 2)
    for i in range(N):
        l[i * N + i] = 1
    I = np.array(l, dtype=complex)
    return I.reshape(N, N)

def tensor(u, v):
    l = np.array([0] * (len(u) * len(v)), dtype=complex)
    i = 0
    for e1 in u:
        for e2 in v:
            l[i] = e1 * e2
            i += 1
    return l
