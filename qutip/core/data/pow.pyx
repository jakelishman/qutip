#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython

from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR
from qutip.core.data.matmul cimport matmul_csr

__all__ = [
    'pow', 'pow_csr',
]


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef CSR pow_csr(CSR matrix, unsigned long long n):
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("matrix power only works with square matrices")
    if n == 0:
        return csr.identity(matrix.shape[0])
    if n == 1:
        return matrix.copy()
    # We do the matrix power in terms of powers of two, so we can do it
    # ceil(lg(n)) + bits(n) - 1 matrix mulitplications, where `bits` is the
    # number of set bits in the input.
    #
    # We don't have to do matrix.copy() or pow.copy() here, because we've
    # guaranteed that we won't be returning without at least one matrix
    # multiplcation, which will allocate a new matrix.
    cdef CSR pow = matrix
    cdef CSR out = pow if n & 1 else None
    n >>= 1
    while n:
        pow = matmul_csr(pow, pow)
        if n & 1:
            out = pow if out is None else matmul_csr(out, pow)
        n >>= 1
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

pow = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('n', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='pow',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
pow.__doc__ =\
    """
    Compute the integer matrix power of the square input matrix.  The power
    must be an integer >= 0.  `A ** 0` is defined to be the identity matrix of
    the same shape.

    Arguments
    ---------
    matrix : Data
        Input matrix to take the power of.

    n : non-negative integer
        The power to which to raise the matrix.
    """
pow.add_specialisations([
    (CSR, CSR, pow_csr),
], _defer=True)

del _inspect, _Dispatcher
