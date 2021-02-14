import numpy as np
import scipy.sparse.linalg

from . import csr
from .dense import Dense
from .csr import CSR
from .properties import isdiag_csr

__all__ = [
    'expm', 'expm_csr', 'expm_csr_dense',
]


def expm_csr(matrix: CSR) -> CSR:
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("can only exponentiate square matrix")
    if csr.nnz(matrix) == 0:
        return csr.identity(matrix.shape[0])
    if isdiag_csr(matrix):
        out = matrix.copy()
        sci = out.as_scipy()
        sci.data[:] = np.exp(sci.data)
        return out
    # The scipy solvers for the Pade approximant are more efficient with the
    # CSC format than the CSR one.
    csc = matrix.as_scipy().tocsc()
    return CSR(scipy.sparse.linalg.expm(csc).tocsr())


def expm_csr_dense(matrix: CSR) -> Dense:
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("can only exponentiate square matrix")
    return Dense(scipy.sparse.linalg.expm(matrix.to_array()))


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

expm = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='expm',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
expm.__doc__ = """Matrix exponential `e**A` for a matrix `A`."""
expm.add_specialisations([
    (CSR, CSR, expm_csr),
    (CSR, Dense, expm_csr_dense),
], _defer=True)

del _inspect, _Dispatcher
