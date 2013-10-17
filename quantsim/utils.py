from __future__ import division, print_function
import numpy as np
import cmath as cm

def normalize(v):
    """Returns an array normalized to have magnitude 1.

    >>> normalize(np.array([3, 4]))
    array([ 0.6+0.j,  0.8+0.j])
    """
    return v / magnitude(v)

def magnitude(v):
    """Finds the magnitude of an array
    
    >>> magnitude(np.array([1j, 1]))
    (1.4142135623730951+0j)
    """
    return cm.sqrt(v.dot(v.conjugate()))

def check_transform(T, N):
    """Raises an error if T is not a unitary matrix
    Takes advantage of the fact that if T is unitary, then T x T.conjugate.transpose is the identity
    
    >>> check_transform(np.array([[1, 0], [0, -1]]), 2)
    >>> check_transform(np.array([[1, 1], [0, -1]]), 2)
    Traceback (most recent call last):
        ...
    ValueError: Transformation must be unitary
    """
    if type(T) != np.ndarray:
        raise ValueError("Transformation must be an array")
    if T.shape != (N, N):
        raise ValueError("Transformation must be %sx%s" % (N, N))
    I = identity(N)
    if not np.allclose(T.conjugate().transpose().dot(T), I):
        raise ValueError("Transformation must be unitary")

def identity(N):
    """Returns an N x N identity matrix

    >>> identity(3)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    """
    l = [0] * (N ** 2)
    for i in range(N):
        l[i * N + i] = 1
    I = np.array(l)
    return I.reshape(N, N)

def tensor(u, v):
    l = np.array([0] * (len(u) * len(v)), dtype=complex)
    i = 0
    for e1 in u:
        for e2 in v:
            l[i] = e1 * e2
            i += 1
    return l

def create_transform(transforms):
    """Transforms is a list of unitary arrays"""
    S = transforms[0]
    for T in transforms[1:]:
        S = np.kron(S, T)
    return S

def swap_columns(T, i, j):
    T[:,[i,j]] = T[:,[j,i]]

def swap(N, i, j):
    """Creates a transformation that swaps the position of qubits i and j"""
    if i > j:
        return swap(N, j, i)
    # easier to work with 1..N instead of 0..N-1
    i += 1
    j += 1
    I = identity(2 ** N)
    # swap distance is difference between powers of 2
    diff = 2 ** (N - i) - 2 ** (N - j)
    for k in range(2 ** N):
        # swap columns if k[i] = 0 and k[j] = 1
        if ((1 << (N - i)) & k) == 0 and ((1 << (N - j)) & k) > 0:
            swap_columns(I, k, k + diff)
    return I
