# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module contains functions for generating Qobj representation of a variety
of commonly occuring quantum operators.
"""

__all__ = ['jmat', 'spin_Jx', 'spin_Jy', 'spin_Jz', 'spin_Jm', 'spin_Jp',
           'spin_J_set', 'sigmap', 'sigmam', 'sigmax', 'sigmay', 'sigmaz',
           'destroy', 'create', 'qeye', 'identity', 'position', 'momentum',
           'num', 'squeeze', 'squeezing', 'displace', 'commutator',
           'qutrit_ops', 'qdiags', 'phase', 'qzero', 'enr_destroy',
           'enr_identity', 'charge', 'tunneling',
           'expand_operator', 'controlled',
           'x_gate', 'cx_gate', 'rx_gate', 'cnot_gate',
           'y_gate', 'cy_gate', 'ry_gate',
           'z_gate', 'cz_gate', 'rz_gate', 'csign_gate',
           's_gate', 'cs_gate', 't_gate', 'ct_gate',
           'berkeley_gate', 'u3_gate', 'swap_gate', 'iswap_gate',
           'hadamard_gate', 'sqrtnot_gate', 'phase_gate', 'fredkin_gate',
           'toffoli_gate', 'sqrtswap_gate', 'sqrtiswap_gate', 'swapalpha_gate',
           'hadamard_transform', 'qft']

import numbers

import numpy as np
import scipy.sparse

from . import data as _data
from .qobj import Qobj
from .dimensions import flatten
from ..settings import settings as _settings


def _expand_operator_check(operator, new_indices, dimensions):
    """
    Internal function for validating the inputs to :obj:`expand_operator`.
    """
    if not isinstance(operator, Qobj):
        raise TypeError(
            "expected a Qobj for operator but got {}".format(operator)
        )
    if operator.dims[0] != operator.dims[1]:
        raise ValueError(
            "operator is not square; its dimensions are {}"
            .format(operator.dims)
        )
    if isinstance(new_indices, numbers.Integral):
        new_indices = [new_indices]
    new_indices = list(new_indices)
    dimensions = tuple(dimensions)
    for old_i, new_i in enumerate(new_indices):
        if not 0 <= new_i < len(dimensions):
            raise ValueError(
                "found an index {} outside the number of new dimensions {}"
                .format(new_i, len(dimensions))
            )
        if operator.dims[0][old_i] != dimensions[new_i]:
            raise ValueError(
                "dimension {} is not the same between input {} and output {}"
                .format(old_i, operator.dims[0][old_i], dimensions[new_i])
            )
    return operator, new_indices, dimensions


def expand_operator(operator, new_indices, dimensions):
    """
    Expand an operator, which potentially has a tensor structure, onto a larger
    square Hilbert space, by inserting identities into the "missing" spaces.
    The subspaces of the operator can be mapped to arbitrary locations in the
    new object.  For example, this function can be used to map a CY gate
    controlled by qubit 0 targetting qubit 1 into a 5-qubit space with qubit 4
    as the control and qubit 3 as the target by ::

        expand_operator(cy_gate(0, 1), (4, 3), [2]*5)

    Parameters
    ----------
    operator : :obj:`.Qobj`
        A square quantum operator.
    new_indices : Iterable[int]
        The indices of the subspaces of ``dimensions`` which the output
        operator should act on.  These are given in order relative to how they
        appear in the dimensions of ``operator``.
    dimensions : Iterable[int]
        The dimensions of the new Hilbert space.  All elements are square, so
        this object should be an iterable of their sizes.

    Returns
    -------
    :obj:`.Qobj`
        The operator expanded into a larger qubit space.

    Examples
    --------
    The default :obj:`.cx_gate` is controlled by qubit 0 and targets qubit 1.

    >>> cx_gate()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]

    We can use :obj:`.expand_operator` to simply reorder the control and the
    target.

    >>> expand_operator(cx_gate(), [1, 0], [2, 2])
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]]

    A more fitting example for the name of the function is to expand an
    operator into a larger Hilbert space.  Here we take the same controlled-X
    gate, and map it into a 3-qubit Hilbert space, with the control on the
    first qubit and the target on the last.

    >>> expand_operator(cx_gate(), [0, 2], [2, 2, 2])
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', \
isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 1. 0.]]
    """
    # Protected import to avoid circular dependencies.
    from .tensor import tensor
    operator, new_indices, dimensions =\
        _expand_operator_check(operator, new_indices, dimensions)
    new_indices_set = set(new_indices)
    id_dimensions = []
    permutation = [None] * len(dimensions)
    for old_i, new_i in enumerate(new_indices):
        permutation[new_i] = old_i
    ptr = len(new_indices)
    for new_i, dim in enumerate(dimensions):
        if new_i in new_indices_set:
            continue
        id_dimensions.append(dim)
        permutation[new_i] = ptr
        ptr += 1
    out = tensor(operator, qeye(id_dimensions)) if id_dimensions else operator
    return out.permute(permutation)


class _constant_operator_function:
    """
    A function which returns a constant operator.  See the docstring of the
    ``__call__`` method for more information if you can see this message.
    """

    def __init__(self, name: str, base: Qobj, docstring: str):
        self.__name__ = self.__qualname__ = name
        # We set both the instance and __call__ method docstring to make sure
        # that the user will be able to see it; help() gets the docstring from
        # a class descriptor, so misses instance docstrings, but it will catch
        # the one on the method call.
        self.__doc__ = self.__call__.__func__.__doc__ = docstring
        self._base = base
        self._cache = {dtype: base.to(dtype) for dtype in _data.to.dtypes}

    def __call__(self, *, dtype=None):
        if dtype is None:
            return self._base.copy()
        dtype = _data.to.parse(dtype)
        # The try/except is necessary in case a new dtype has been added to
        # _data.to since this object was initialised.
        try:
            return self._cache[dtype].copy()
        except KeyError:
            pass
        self._cache[dtype] = self._base.to(dtype)
        # Don't return the object we just created, in case the user mutates the
        # data backing---that would spoil our cache.
        return self._cache[dtype].copy()


def qdiags(diagonals, offsets, dims=None, shape=None, *, dtype=_data.CSR):
    """
    Constructs an operator from an array of diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Array of elements to place along the selected diagonals.

    offsets : sequence of ints
        Sequence for diagonals to be set:
            - k=0 main diagonal
            - k>0 kth upper diagonal
            - k<0 kth lower diagonal
    dims : list, optional
        Dimensions for operator

    shape : list, tuple, optional
        Shape of operator.  If omitted, a square operator large enough
        to contain the diagonals is generated.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Examples
    --------
    >>> qdiags(sqrt(range(1, 4)), 1) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isherm = False
    Qobj data =
    [[ 0.          1.          0.          0.        ]
     [ 0.          0.          1.41421356  0.        ]
     [ 0.          0.          0.          1.73205081]
     [ 0.          0.          0.          0.        ]]

    """
    data = _data.diag[dtype](diagonals, offsets, shape)
    return Qobj(data, dims=dims, type='oper', copy=False)


def jmat(j, which=None, *, dtype=_data.CSR):
    """Higher-order spin operators:

    Parameters
    ----------
    j : float
        Spin of operator

    which : str
        Which operator to return 'x','y','z','+','-'.
        If no args given, then output is ['x','y','z']

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    jmat : Qobj or tuple of Qobj
        ``qobj`` for requested spin operator(s).


    Examples
    --------
    >>> jmat(1) # doctest: +SKIP
    [ Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.          0.70710678  0.        ]
     [ 0.70710678  0.          0.70710678]
     [ 0.          0.70710678  0.        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j          0.-0.70710678j  0.+0.j        ]
     [ 0.+0.70710678j  0.+0.j          0.-0.70710678j]
     [ 0.+0.j          0.+0.70710678j  0.+0.j        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0. -1.]]]


    Notes
    -----
    If no 'args' input, then returns array of ['x','y','z'] operators.

    """
    if int(2 * j) != 2 * j or j < 0:
        raise ValueError('j must be a non-negative integer or half-integer')

    if not which:
        return (
            jmat(j, 'x', dtype=dtype),
            jmat(j, 'y', dtype=dtype),
            jmat(j, 'z', dtype=dtype)
        )

    dims = [[int(2*j + 1)]]*2
    if which == '+':
        return Qobj(_jplus(j, dtype=dtype), dims=dims, type='oper',
                    isherm=False, isunitary=False, copy=False)
    if which == '-':
        return Qobj(_jplus(j, dtype=dtype).adjoint(), dims=dims, type='oper',
                    isherm=False, isunitary=False, copy=False)
    if which == 'x':
        A = 0.5 * _jplus(j, dtype=dtype)
        return Qobj(A + A.adjoint(), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False)
    if which == 'y':
        A = -0.5j * _jplus(j, dtype=dtype)
        return Qobj(A + A.adjoint(), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False)
    if which == 'z':
        return Qobj(_jz(j, dtype=dtype), dims=dims, type='oper',
                    isherm=True, isunitary=False, copy=False)
    raise ValueError('invalid spin operator: ' + which)


def _jplus(j, *, dtype=_data.CSR):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = np.sqrt(j * (j + 1) - m * (m + 1))[1:]
    return _data.diag[dtype](data, 1)


def _jz(j, *, dtype=_data.CSR):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    N = int(2*j + 1)
    data = np.array([j-k for k in range(N)], dtype=complex)
    return _data.diag[dtype](data, 0)


#
# Spin j operators:
#
def spin_Jx(j, *, dtype=_data.CSR):
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x', dtype=dtype)


def spin_Jy(j, *, dtype=_data.CSR):
    """Spin-j y operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'y', dtype=dtype)


def spin_Jz(j, *, dtype=_data.CSR):
    """Spin-j z operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'z', dtype=dtype)


def spin_Jm(j, *, dtype=_data.CSR):
    """Spin-j annihilation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '-', dtype=dtype)


def spin_Jp(j, *, dtype=_data.CSR):
    """Spin-j creation operator

    Parameters
    ----------
    j : float
        Spin of operator

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '+', dtype=dtype)


def spin_J_set(j, *, dtype=_data.CSR):
    """Set of spin-j operators (x, y, z)

    Parameters
    ----------
    j : float
        Spin of operators

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    list : list of Qobj
        list of ``qobj`` representating of the spin operator.

    """
    return jmat(j, dtype=dtype)


#
# Pauli spin-1/2 operators.
#
sigmap = _constant_operator_function(
    'sigmap', jmat(0.5, '+'),
    """
    Creation operator for Pauli spins.

    Examples
    --------
    >>> sigmap()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', isherm=False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]
    """,
)
sigmam = _constant_operator_function(
    'sigmam', jmat(0.5, '-'),
    """
    Annihilation operator for Pauli spins.

    Examples
    --------
    >>> sigmam()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', isherm=False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]
    """,
)
sigmax = _constant_operator_function(
    'sigmax', 2*jmat(0.5, 'x'),
    r"""
    Pauli spin-:math:`\frac12` :math:`\sigma_x` operator.

    Examples
    --------
    >>> sigmax()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', isherm=True
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]
    """,
)
sigmay = _constant_operator_function(
    'sigmay', 2*jmat(0.5, 'y'),
    r"""
    Pauli spin-:math:`\frac12` :math:`\sigma_y` operator.

    Examples
    --------
    >>> sigmay()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', isherm=True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]
    """,
)
sigmaz = _constant_operator_function(
    'sigmaz', 2*jmat(0.5, 'z'),
    r"""
    Pauli spin-:math:`\frac12` :math:`\sigma_z` operator.

    Examples
    --------
    >>> sigmaz()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', isherm=True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]
    """,
)


def destroy(N, offset=0, *, dtype=_data.CSR):
    """
    Destruction (lowering) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for lowering operator.

    Examples
    --------
    >>> destroy(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=False
    Qobj data =
    [[ 0.00000000+0.j  1.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.41421356+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  1.73205081+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]]
    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, 1, dtype=dtype)


def create(N, offset=0, *, dtype=_data.CSR):
    """
    Creation (raising) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Examples
    --------
    >>> create(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=False
    Qobj data =
    [[ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 1.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  1.41421356+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.73205081+0.j  0.00000000+0.j]]
    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    return qdiags(data, -1, dtype=dtype)


def _implicit_tensor_dimensions(dimensions):
    """
    Total flattened size and operator dimensions for operator creation routines
    that automatically perform tensor products.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        First dimension of an operator which can create an implicit tensor
        product.  If the type is `int`, it is promoted first to `[dimensions]`.
        From there, it should be one of the two-elements `dims` parameter of a
        `qutip.Qobj` representing an `oper` or `super`, with possible tensor
        products.

    Returns
    -------
    size : int
        Dimension of backing matrix required to represent operator.
    dimensions : list
        Dimension list in the form required by ``Qobj`` creation.
    """
    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    flat = flatten(dimensions)
    if not all(isinstance(x, numbers.Integral) and x >= 0 for x in flat):
        raise ValueError("All dimensions must be integers >= 0")
    return np.prod(flat), [dimensions, dimensions]


def qzero(dimensions, *, dtype=_data.CSR):
    """
    Zero operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """
    size, dimensions = _implicit_tensor_dimensions(dimensions)
    # A sparse matrix with no data is equal to a zero matrix.
    type_ = 'super' if isinstance(dimensions[0][0], list) else 'oper'
    return Qobj(_data.zeros[dtype](size, size), dims=dimensions, type=type_,
                isherm=True, isunitary=False, copy=False)


def qeye(dimensions, *, dtype=_data.CSR):
    """
    Identity operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.

    Examples
    --------
    >>> qeye(3) # doctest: +SKIP
    Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, \
isherm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> qeye([2,2]) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, \
isherm = True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    """
    size, dimensions = _implicit_tensor_dimensions(dimensions)
    type_ = 'super' if isinstance(dimensions[0][0], list) else 'oper'
    return Qobj(_data.identity[dtype](size), dims=dimensions, type=type_,
                isherm=True, isunitary=True, copy=False)


# Name alias.
identity = qeye


def position(N, offset=0, *, dtype=_data.CSR):
    """
    Position operator x=1/sqrt(2)*(a+a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Position operator as Qobj.
    """
    a = destroy(N, offset=offset, dtype=dtype)
    return np.sqrt(0.5) * (a + a.dag())


def momentum(N, offset=0, *, dtype=_data.CSR):
    """
    Momentum operator p=-1j/sqrt(2)*(a-a.dag())

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Momentum operator as Qobj.
    """
    a = destroy(N, offset=offset, dtype=dtype)
    return -1j * np.sqrt(0.5) * (a - a.dag())


def num(N, offset=0, *, dtype=_data.CSR):
    """
    Quantum object for number operator.

    Parameters
    ----------
    N : int
        The dimension of the Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper: qobj
        Qobj for number operator.

    Examples
    --------
    >>> num(4) # doctest: +SKIP
    Quantum object: dims=[[4], [4]], shape=(4, 4), type='oper', isherm=True
    Qobj data =
    [[0 0 0 0]
     [0 1 0 0]
     [0 0 2 0]
     [0 0 0 3]]
    """
    data = np.arange(offset, offset + N, dtype=complex)
    return qdiags(data, 0, dtype=dtype)


def squeeze(N, z, offset=0, *, dtype=_data.CSR):
    """Single-mode squeezing operator.

    Parameters
    ----------
    N : int
        Dimension of hilbert space.

    z : float/complex
        Squeezing parameter.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.


    Examples
    --------
    >>> squeeze(4, 0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.98441565+0.j  0.00000000+0.j  0.17585742+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.95349007+0.j  0.00000000+0.j  0.30142443+0.j]
     [-0.17585742+0.j  0.00000000+0.j  0.98441565+0.j  0.00000000+0.j]
     [ 0.00000000+0.j -0.30142443+0.j  0.00000000+0.j  0.95349007+0.j]]

    """
    asq = destroy(N, offset=offset) ** 2
    op = 0.5*np.conj(z)*asq - 0.5*z*asq.dag()
    return op.expm(dtype=dtype)


def squeezing(a1, a2, z):
    """Generalized squeezing operator.

    .. math::

        S(z) = \\exp\\left(\\frac{1}{2}\\left(z^*a_1a_2
        - za_1^\\dagger a_2^\\dagger\\right)\\right)

    Parameters
    ----------
    a1 : :class:`qutip.qobj.Qobj`
        Operator 1.

    a2 : :class:`qutip.qobj.Qobj`
        Operator 2.

    z : float/complex
        Squeezing parameter.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.

    """
    b = 0.5 * (np.conj(z)*(a1 @ a2) - z*(a1.dag() @ a2.dag()))
    return b.expm()


def displace(N, alpha, offset=0, *, dtype=_data.Dense):
    """Single-mode displacement operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    alpha : float/complex
        Displacement amplitude.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Displacement operator.

    Examples
    ---------
    >>> displace(4,0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.96923323+0.j -0.24230859+0.j  0.04282883+0.j -0.00626025+0.j]
     [ 0.24230859+0.j  0.90866411+0.j -0.33183303+0.j  0.07418172+0.j]
     [ 0.04282883+0.j  0.33183303+0.j  0.84809499+0.j -0.41083747+0.j]
     [ 0.00626025+0.j  0.07418172+0.j  0.41083747+0.j  0.90866411+0.j]]

    """
    a = destroy(N, offset=offset)
    return (alpha * a.dag() - np.conj(alpha) * a).expm(dtype=dtype)


def commutator(A, B, kind="normal"):
    """
    Return the commutator of kind `kind` (normal, anti) of the
    two operators A and B.
    """
    if kind == 'normal':
        return A @ B - B @ A

    elif kind == 'anti':
        return A @ B + B @ A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


def qutrit_ops(*, dtype=_data.CSR):
    """
    Operators for a three level system (qutrit).

    Parameters
    ----------
    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    opers: array
        `array` of qutrit operators.

    """
    from .states import qutrit_basis

    out = np.empty((6,), dtype=object)
    one, two, three = qutrit_basis(dtype=dtype)
    out[0] = one * one.dag()
    out[1] = two * two.dag()
    out[2] = three * three.dag()
    out[3] = one * two.dag()
    out[4] = two * three.dag()
    out[5] = three * one.dag()
    return out


def phase(N, phi0=0, *, dtype=_data.Dense):
    """
    Single-mode Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    phi0 : float
        Reference phase.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    oper : qobj
        Phase operator with respect to reference phase.

    Notes
    -----
    The Pegg-Barnett phase operator is Hermitian on a truncated Hilbert space.

    """
    phim = phi0 + (2 * np.pi * np.arange(N)) / N  # discrete phase angles
    n = np.arange(N)[:, np.newaxis]
    states = np.array([np.sqrt(kk) / np.sqrt(N) * np.exp(1j * n * kk)
                       for kk in phim])
    ops = np.sum([np.outer(st, st.conj()) for st in states], axis=0)
    return Qobj(ops, dims=[[N], [N]], type='oper', copy=False).to(dtype)


def enr_destroy(dims, excitations, *, dtype=_data.CSR):
    """
    Generate annilation operators for modes in a excitation-number-restricted
    state space. For example, consider a system consisting of 4 modes, each
    with 5 states. The total hilbert space size is 5**4 = 625. If we are
    only interested in states that contain up to 2 excitations, we only need
    to include states such as

        (0, 0, 0, 0)
        (0, 0, 0, 1)
        (0, 0, 0, 2)
        (0, 0, 1, 0)
        (0, 0, 1, 1)
        (0, 0, 2, 0)
        ...

    This function creates annihilation operators for the 4 modes that act
    within this state space:

        a1, a2, a3, a4 = enr_destroy([5, 5, 5, 5], excitations=2)

    From this point onwards, the annihiltion operators a1, ..., a4 can be
    used to setup a Hamiltonian, collapse operators and expectation-value
    operators, etc., following the usual pattern.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    a_ops : list of qobj
        A list of annihilation operators for each mode in the composite
        quantum system described by dims.
    """
    from .states import enr_state_dictionaries

    nstates, _, idx2state = enr_state_dictionaries(dims, excitations)

    a_ops = [scipy.sparse.lil_matrix((nstates, nstates), dtype=np.complex128)
             for _ in dims]

    for n1, state1 in idx2state.items():
        for n2, state2 in idx2state.items():
            for idx, a in enumerate(a_ops):
                s1 = [s for idx2, s in enumerate(state1) if idx != idx2]
                s2 = [s for idx2, s in enumerate(state2) if idx != idx2]
                if (state1[idx] == state2[idx] - 1) and (s1 == s2):
                    a[n1, n2] = np.sqrt(state2[idx])

    return [Qobj(a, dims=[dims, dims]).to(dtype) for a in a_ops]


def enr_identity(dims, excitations, *, dtype=_data.CSR):
    """
    Generate the identity operator for the excitation-number restricted
    state space defined by the `dims` and `exciations` arguments. See the
    docstring for enr_fock for a more detailed description of these arguments.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    state : list of integers
        The state in the number basis representation.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        A Qobj instance that represent the identity operator in the
        exication-number-restricted state space defined by `dims` and
        `exciations`.
    """
    from .states import enr_state_dictionaries
    nstates, _, _ = enr_state_dictionaries(dims, excitations)
    return Qobj(_data.identity[dtype](nstates),
                dims=[dims, dims],
                type='oper',
                isherm=True,
                isunitary=True,
                copy=False)


def charge(Nmax, Nmin=None, frac=1, *, dtype=_data.CSR):
    """
    Generate the diagonal charge operator over charge states
    from Nmin to Nmax.

    Parameters
    ----------
    Nmax : int
        Maximum charge state to consider.

    Nmin : int (default = -Nmax)
        Lowest charge state to consider.

    frac : float (default = 1)
        Specify fractional charge if needed.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    C : Qobj
        Charge operator over [Nmin, Nmax].

    Notes
    -----
    .. versionadded:: 3.2

    """
    if Nmin is None:
        Nmin = -Nmax
    diag = frac * np.arange(Nmin, Nmax+1, dtype=float)
    out = qdiags(diag, 0, dtype=dtype)
    out.isherm = True
    return out


def tunneling(N, m=1, *, dtype=_data.CSR):
    r"""
    Tunneling operator with elements of the form
    :math:`\\sum |N><N+m| + |N+m><N|`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    m : int (default = 1)
        Number of excitations in tunneling event.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    T : Qobj
        Tunneling operator.

    Notes
    -----
    .. versionadded:: 3.2

    """
    diags = [np.ones(N-m, dtype=int), np.ones(N-m, dtype=int)]
    T = qdiags(diags, [m, -m], dtype=dtype)
    T.isherm = True
    return T


# Gate-like operators, imported from the 4.x qutip.qip for convenience, since
# qip is split out into its own package now.


def controlled(operator: Qobj, *, dtype=_data.CSR):
    """
    Making a simple controlled gate out of a given operator.  For a given
    object :math:`X` this produces :math:`\\ket0\\bra0\\otimes I +
    \\ket1\\bra1\\otimes X`.

    If you need more advanced usage, including arbitrary controls and targets,
    consider using this function in conjunction with :obj:`.expand_operator`,
    or use the full ``qutip-qip`` package.

    Examples
    --------
    >>> controlled(rx_gate(np.pi/3))
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=False
    Qobj data =
    [[1.       +0.j  0.       +0.j  0.       +0.j  0.       +0.j ]
     [0.       +0.j  1.       +0.j  0.       +0.j  0.       +0.j ]
     [0.       +0.j  0.       +0.j  0.8660254+0.j  0.       -0.5j]
     [0.       +0.j  0.       +0.j  0.       -0.5j 0.8660254+0.j ]]
    """
    # This function cannot import tensor because it is evaluated during
    # initialisation of the module, and there would be a circular dependency.
    # Similarly, we cannot import states, so we have to reinvent the wheel with
    # projection operators as well.
    dtype = _data.to.parse(dtype)
    _0 = _data.one_element[dtype]((2, 2), (0, 0), 1)
    _1 = _data.one_element[dtype]((2, 2), (1, 1), 1)
    if operator.dims[0] != operator.dims[1]:
        raise ValueError(
            "expected a square operator but got {}".format(operator.dims)
        )
    return Qobj(
        _data.add(
            _data.kron(_0, _data.identity[dtype](operator.shape[0])),
            _data.kron(_1, operator.data),
            dtype=dtype,
        ),
        dims=[[2] + operator.dims[0]]*2, copy=False, type='oper',
        isherm=operator.isherm, isunitary=operator.isunitary,
    )


def rx_gate(angle: float, *, dtype=_data.Dense):
    r"""
    Single-qubit rotation of ``angle`` radians around the :math:`x`-axis.  In
    the :math:`\{\ket0,\ket1\}` basis this has the matrix representation

    .. math::
        R_x(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -i\sin(\theta/2)\\
            -i\sin(\theta/2) & \cos(\theta/2)\\
        \end{pmatrix}
    """
    if not isinstance(angle, numbers.Real):
        raise TypeError(
            "'angle' must be a real number, but got {!r}".format(angle)
        )
    diag, off = np.cos(0.5*angle), -1j*np.sin(0.5*angle)
    data = _data.Dense(np.array([[diag, off], [off, diag]]))
    herm = 2*abs(off.imag) < _settings.core['atol']
    return Qobj(
        _data.to(dtype, data),
        copy=False, dims=[[2], [2]], type='oper', isunitary=True, isherm=herm,
    )


def ry_gate(angle: float, *, dtype=_data.Dense):
    r"""
    Single-qubit rotation of ``angle`` radians around the :math:`y`-axis.  In
    the :math:`\{\ket0,\ket1\}` basis this has the matrix representation

    .. math::
        R_y(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2)\\
            \sin(\theta/2) & \cos(\theta/2)\\
        \end{pmatrix}
    """
    if not isinstance(angle, numbers.Real):
        raise TypeError(
            "'angle' must be a real number, but got {!r}".format(angle)
        )
    diag, off = np.cos(0.5*angle), np.sin(0.5*angle)
    data = _data.Dense(
        np.array([[diag, -off], [off, diag]], dtype=np.complex128),
    )
    herm = 2*abs(off) < _settings.core['atol']
    return Qobj(
        _data.to(dtype, data),
        copy=False, dims=[[2], [2]], type='oper', isunitary=True, isherm=herm,
    )


def rz_gate(angle: float, *, dtype=_data.CSR):
    r"""
    Single-qubit rotation of ``angle`` radians around the :math:`z`-axis.  In
    the :math:`\{\ket0,\ket1\}` basis this has the matrix representation

    .. math::
        R_z(\theta) = \begin{pmatrix}
            e^{-i\theta/2} & 0\\
            0 & e^{i\theta/2}
        \end{pmatrix}
    """
    if not isinstance(angle, numbers.Real):
        raise TypeError(
            "'angle' must be a real number, but got {!r}".format(angle)
        )
    diag = np.exp(-0.5j * angle)
    herm = abs(diag.imag) < _settings.core['atol']
    return Qobj(
        _data.diag[dtype](np.array([[diag, diag.conjugate()]]), [0]),
        copy=False, dims=[[2], [2]], type='oper', isunitary=True, isherm=herm,
    )


def u3_gate(theta: float, phi: float, lambda_: float, *, dtype=_data.Dense):
    r"""
    OpenQASM :math:`U_3` gate, parametrised by three angles.  In terms of
    single-qubit rotations, this is equivalent (up to a global phase) to

    .. math::
        U_3(\theta, \phi, \lambda) = R_z(\phi)R_y(\theta)R_z(\gamma)

    or in matrix form

    .. math::
        U_3(\theta, \phi, \lambda) = \begin{pmatrix}
            \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2)\\
            e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
        \end{pmatrix}
    """
    cos, sin = np.cos(0.5*theta), np.sin(0.5*theta)
    ephi, elambda = np.exp(1j*phi), np.exp(1j*lambda_)
    data = _data.Dense(
        np.array([[cos, -elambda*sin], [ephi*sin, ephi*elambda*cos]],
                 dtype=np.complex128),
    )
    return Qobj(
        _data.to(dtype, data),
        copy=False, dims=[[2], [2]], type='oper', isunitary=True,
    )


def phase_gate(angle: float, *, dtype=_data.Dense):
    r"""
    Single-qubit phase shift of the :math:`\ket1` state relative to the
    :`math:`\ket0` state.  This is equivalent, up to a global phase, as
    :obj:`.rz_gate`.
    """
    if not isinstance(angle, numbers.Real):
        raise TypeError(
            "'angle' must be a real number, but got {!r}".format(angle)
        )
    diag = np.exp(1j * angle)
    data = _data.Dense(
        np.array([[1, 0], [0, diag]], dtype=np.complex128),
    )
    herm = abs(diag.imag) < _settings.core['atol']
    return Qobj(
        _data.to(dtype, data),
        copy=False, dims=[[2], [2]], type='oper', isunitary=True, isherm=herm,
    )


def swapalpha_gate(alpha: float, *, dtype=_data.Dense):
    r"""
    Unitary :math:`\text{SWAP}^\alpha` gate.  In matrix form, this is

    .. math::
        \text{SWAP}^\alpha = \begin{pmatrix}
            1 & 0 & 0 & 0\\
            0 & \frac12(1 + e^{i\pi\alpha}) & \frac12(1 - e^{i\pi\alpha}) & 0\\
            0 & \frac12(1 - e^{i\pi\alpha}) & \frac12(1 + e^{i\pi\alpha}) & 0\\
            0 & 0 & 0 & 1
        \end{pmatrix}
    """
    if not isinstance(alpha, numbers.Real):
        raise TypeError(
            "'alpha' must be a real number, but got {!r}".format(alpha)
        )
    phase = np.exp(1j * np.pi * alpha)
    data = _data.Dense(np.array([
        [1, 0, 0, 0],
        [0, 0.5*(1 + phase), 0.5*(1 - phase), 0],
        [0, 0.5*(1 - phase), 0.5*(1 + phase), 0],
        [0, 0, 0, 1],
    ]))
    return Qobj(
        _data.to(dtype, data),
        copy=False, dims=[[2, 2], [2, 2]], type='oper', isunitary=True,
    )


def hadamard_transform(N: int, *, dtype=_data.Dense):
    """
    General Hadamard transformation on many qubits.  This the Hadamard matrix
    of size :math:`2^n`, scaled to be unitary.

    See Also
    --------
    hadamard_gate :
        convenience function for a single-qubit Hadamard transformation.
    """
    if not isinstance(N, numbers.Integral) or N <= 0:
        raise (ValueError if isinstance(N, numbers.Integral) else TypeError)(
            "'N' must be a positive integer, but is {}".format(N)
        )
    size = 2**N
    return Qobj(
        (1/np.sqrt(size)) * scipy.linalg.hadamard(2**N, dtype=np.complex128),
        dims=[[2]*N, [2]*N], type='oper', isunitary=True, isherm=True,
    ).to(dtype)


def qft(N: int, *, dtype=_data.Dense):
    """
    Quantum Fourier transform operator on ``N`` qubits.
    """
    if not isinstance(N, numbers.Integral) or N <= 0:
        raise (ValueError if isinstance(N, numbers.Integral) else TypeError)(
            "'N' must be a positive integer, but is {}".format(N)
        )
    size = 2**N
    ns = np.arange(size)
    data = np.exp(2j*np.pi/size * np.outer(ns, ns))/np.sqrt(size)
    return Qobj(
        _data.Dense(data, copy=False),
        copy=False, type='oper', isherm=True, isunitary=True, dims=[[2]*N]*2,
    ).to(dtype)


x_gate = _constant_operator_function('x_gate', sigmax(), sigmax.__doc__)
# Name alias - CNOT is the more common name, so we use that in the docs.
cnot_gate = cx_gate = _constant_operator_function(
    'cnot_gate',
    controlled(sigmax()),
    """
    Controlled-NOT gate.  This is a Pauli-X operation on the second qubit if
    the first qubit is in the :math:`\\ket1` state.

    Examples
    --------
    >>> cnot_gate()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=True
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """,
)
y_gate = _constant_operator_function('y_gate', sigmay(), sigmay.__doc__)
cy_gate = _constant_operator_function(
    'cy_gate',
    controlled(sigmay()),
    """
    Controlled Pauli spin-:math:`\\frac12` :math:`\\sigma_y` gate.

    Examples
    --------
    >>> cy_gate()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=True
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]]
    """,
)
z_gate = _constant_operator_function('z_gate', sigmaz(), sigmaz.__doc__)
# Name alias - CZ is more common, so use that in docs.
csign_gate = cz_gate = _constant_operator_function(
    'cz_gate',
    controlled(sigmaz()),
    """
    Controlled Pauli spin-:math:`\\frac12` :math:`\\sigma_z` gate.

    Examples
    --------
    >>> cz_gate()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', \
isherm=True
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """,
)
s_gate = _constant_operator_function(
    's_gate',
    Qobj(
        [
            [1, 0],
            [0, -1j],
        ],
        type='oper', dims=[[2], [2]], isherm=False, isunitary=False,
    ),
    r"""
    Spin-:math:`\frac12` Pauli :math:`\sqrt Z`-gate, equivalent to a rotation
    of :math:`\pi/2` around the :math:`z`-axis.
    """,
)
cs_gate = _constant_operator_function(
    'cs_gate',
    controlled(s_gate()),
    r"""
    Controlled spin-:math:`\frac12` Pauli :math:`\sqrt Z`-gate, equivalent to a
    rotation of :math:`\pi/2` around the :math:`z`-axis on the second qubit, if
    the first is in state :math:`\ket1`.
    """,
)
t_gate = _constant_operator_function(
    't_gate',
    Qobj(
        [[1, 0], [0, np.exp(0.25j*np.pi)]],
        type='oper', dims=[[2], [2]], isherm=False, isunitary=False,
    ),
    r"""
    Spin-:math:`\frac12` Pauli :math:`\sqrt[4]Z`-gate, equivalent to a rotation
    of :math:`\pi/4` around the :math:`z`-axis.
    """,
)
ct_gate = _constant_operator_function(
    'ct_gate',
    controlled(t_gate()),
    r"""
    Controlled spin-:math:`\frac12` Pauli :math:`\sqrt[4]Z`-gate, equivalent to
    a rotation of :math:`\pi/4` around the :math:`z`-axis on the second qubit,
    if the first is in state :math:`\ket1`.
    """,
)
sqrtnot_gate = _constant_operator_function(
    'sqrtnot_gate',
    Qobj(
        [
            [0.5+0.5j, 0.5-0.5j],
            [0.5-0.5j, 0.5+0.5j],
        ],
        type='oper', dims=[[2], [2]], isherm=False, isunitary=False,
    ),
    r"""
    Spin-:math:`\frac12` Pauli :math:`\sqrt X`-gate, equivalent to a rotation
    of :math:`\pi/2` around the :math:`x`-axis.
    """,
)
hadamard_gate = _constant_operator_function(
    'hadamard_gate', hadamard_transform(1),
    r"""
    Single-qubit Hadamard gate.  In terms of Pauli matrices, this has the
    representation :math:`(\sigma_x + \sigma_z)/\sqrt2`.

    See Also
    --------
    hadamard_transform : Generalised Hadamard transform on many qubits.
    """,
)
berkeley_gate = _constant_operator_function(
    'berkeley_gate',
    Qobj(
        [
            [np.cos(0.125*np.pi), 0, 0, 1j*np.sin(0.125*np.pi)],
            [0, np.cos(0.375*np.pi), 1j*np.sin(0.375*np.pi), 0],
            [0, 1j*np.sin(0.375*np.pi), np.cos(0.375*np.pi), 0],
            [1j*np.sin(0.125*np.pi), 0, 0, np.cos(0.125*np.pi)],
        ],
        type='oper', dims=[[2, 2], [2, 2]], isherm=False, isunitary=True,
    ),
    r"""
    Berkeley :math:`B` gate.  In terms of Pauli operators this is

    .. math::
        B = \exp\Bigl[\frac{i\pi}{8}\bigl(2X\otimes X + Y\otimes Y\bigr)\Bigr]

    and explicitly in matrix form it is

    .. math::

        B = \begin{pmatrix}
            \cos(\pi/8) & 0 & 0 & i\sin(\pi/8)\\
            0 & \cos(3\pi/8) & i\sin(3\pi/8) & 0\\
            0 & i\sin(3\pi/8) * \cos(3\pi/8) & 0\\
            i\sin(\pi/8) & 0 & 0 & \cos(\pi/8)
        \end{pmatrix}
    """,
)
swap_gate = _constant_operator_function(
    'swap_gate',
    Qobj(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        type='oper', dims=[[2, 2], [2, 2]], isherm=True, isunitary=True,
    ),
    """Swap gate, swapping the states of two qubits.""",
)
iswap_gate = _constant_operator_function(
    'iswap_gate',
    Qobj(
        [
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ],
        type='oper', dims=[[2, 2], [2, 2]], isherm=False, isunitary=True,
    ),
    r"""
    iSWAP gate, swapping the states of two qubits with a phase of :math:`i` to
    the :math:`\ket{01}\bra{10}` and :math:`\ket{10}\bra{01}` terms.
    """,
)
sqrtswap_gate = _constant_operator_function(
    'sqrtswap_gate',
    Qobj(
        [
            [1, 0, 0, 0],
            [0, 0.5+0.5j, 0.5-0.5j, 0],
            [0, 0.5-0.5j, 0.5+0.5j, 0],
            [0, 0, 0, 1],
        ],
        type='oper', dims=[[2, 2], [2, 2]], isherm=False, isunitary=True,
    ),
    """
    Square root of the swap gate, such that two applications of this gate is
    the same as :obj:`.swap_gate`.
    """,
)
sqrtiswap_gate = _constant_operator_function(
    'sqrtiswap_gate',
    Qobj(
        np.sqrt(0.5) * np.array([
            [np.sqrt(2), 0, 0, 0],
            [0, 1, 1j, 0],
            [0, 1j, 1, 0],
            [0, 0, 0, np.sqrt(2)],
        ]),
        type='oper', dims=[[2, 2], [2, 2]], isherm=False, isunitary=True,
    ),
    """
    Square root of the iSWAP gate, such that two applications of this gate is
    the same as :obj:`.iswap_gate`.
    """,
)
fredkin_gate = _constant_operator_function(
    'fredkin_gate',
    controlled(swap_gate()),
    r"""
    3-qubit Fredkin gate.  This is the same as a controlled-swap gate  In
    matrix form it is

    .. math::

        \text{Fredkin} = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{pmatrix}
    """,
)
toffoli_gate = _constant_operator_function(
    'toffoli_gate',
    controlled(cnot_gate()),
    r"""
    3-qubit Toffoli gate.  This is the same as a controlled-controlled-X gate
    (or controlled-CNOT),  In matrix form it is

    .. math::

        \text{Toffoli} = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
            0 & 0 & 0 & 0 & 0 & 0 & l & 0
        \end{pmatrix}
    """,
)
