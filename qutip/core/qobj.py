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
"""The Quantum Object (Qobj) class, for representing quantum states and
operators, and related functions.
"""

__all__ = [
    'Qobj', 'isbra', 'isket', 'isoper', 'issuper', 'isoperbra', 'isoperket',
    'isherm', 'ptrace',
]

import functools
import numbers

import numpy as np
import scipy.sparse

from .. import __version__, settings
from . import data as _data
from .cy.ptrace import _ptrace
from .permute import _permute
from .sparse import (sp_eigs, sp_expm, sp_fro_norm, sp_max_norm, sp_one_norm,
                     sp_L2_norm)
from .dimensions import (type_from_dims, enumerate_flat, collapse_dims_super)
from .cy.sparse_utils import cy_tidyup

# OPENMP stuff
from .cy.openmp.utilities import use_openmp
if settings.has_openmp:
    from .cy.openmp.omp_sparse_utils import omp_tidyup


_ADJOINT_TYPE_LOOKUP = {
    'oper': 'oper',
    'super': 'super',
    'ket': 'bra',
    'bra': 'ket',
    'operator-ket': 'operator-bra',
    'operator-bra': 'operator-ket',
}

_MATMUL_TYPE_LOOKUP = {
    ('oper', 'ket'): 'ket',
    ('oper', 'oper'): 'oper',
    ('ket', 'bra'): 'oper',
    ('bra', 'oper'): 'bra',
    ('super', 'oper'): 'oper',
    ('super', 'super'): 'super',
}


def isbra(x):
    return isinstance(x, Qobj) and x.type == 'bra'


def isket(x):
    return isinstance(x, Qobj) and x.type == 'ket'


def isoper(x):
    return isinstance(x, Qobj) and x.type == 'oper'


def isoperbra(x):
    return isinstance(x, Qobj) and x.type == 'operator-bra'


def isoperket(x):
    return isinstance(x, Qobj) and x.type == 'operator-ket'


def issuper(x):
    return isinstance(x, Qobj) and x.type == 'super'


def isherm(x):
    return isinstance(x, Qobj) and x.isherm


def _require_equal_type(method):
    """
    Decorate a binary Qobj method to ensure both operands are Qobj and of the
    same type and dimensions.  Promote numeric scalar to identity matrices of
    the same type and shape.
    """
    @functools.wraps(method)
    def out(self, other):
        if (
            self.type in ('oper', 'super')
            and self.dims[0] == self.dims[1]
            and isinstance(other, numbers.Number)
        ):
            scale = complex(other)
            other = Qobj(_data.csr.identity(self.shape[0], scale),
                         dims=self.dims,
                         type=self.type,
                         isherm=(scale.imag == 0),
                         isunitary=(abs(abs(scale) - 1) < settings.atol),
                         copy=False)
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other, type=self.type)
            except TypeError:
                return NotImplemented
        if (
            self.dims != other.dims
            or self.type != other.type
            or (self.type == 'super' and self.superrep != other.superrep)
        ):
            return NotImplemented
        return method(self, other)
    return out


class Qobj:
    """
    A class for representing quantum objects, such as quantum operators and
    states.

    The Qobj class is the QuTiP representation of quantum operators and state
    vectors. This class also implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator/state operations.  The Qobj constructor optionally takes a
    dimension ``list`` and/or shape ``list`` as arguments.

    Parameters
    ----------
    inpt: array_like
        Data for vector/matrix representation of the quantum object.
    dims: list
        Dimensions of object used for tensor products.
    type: {'bra', 'ket', 'oper', 'operator-ket', 'operator-bra', 'super'}
        The type of quantum object to be represented.
    shape: list
        Shape of underlying data structure (matrix shape).
    copy: bool
        Flag specifying whether Qobj should get a copy of the
        input data, or use the original.


    Attributes
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form) or 'choi' (Choi matrix with tr = dimension).
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
    ishp : bool
        Indicates if the quantum object represents a map, and if that map is
        hermicity preserving (HP).
    istp : bool
        Indicates if the quantum object represents a map, and if that map is
        trace preserving (TP).
    iscptp : bool
        Indicates if the quantum object represents a map that is completely
        positive and trace preserving (CPTP).
    isket : bool
        Indicates if the quantum object represents a ket.
    isbra : bool
        Indicates if the quantum object represents a bra.
    isoper : bool
        Indicates if the quantum object represents an operator.
    issuper : bool
        Indicates if the quantum object represents a superoperator.
    isoperket : bool
        Indicates if the quantum object represents an operator in column vector
        form.
    isoperbra : bool
        Indicates if the quantum object represents an operator in row vector
        form.

    Methods
    -------
    copy()
        Create copy of Qobj
    conj()
        Conjugate of quantum object.
    cosm()
        Cosine of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    dnorm()
        Diamond norm of quantum operator.
    dual_chan()
        Dual channel of quantum object representing a CP map.
    eigenenergies(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies (eigenvalues) of a quantum object.
    eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies and eigenstates of quantum object.
    expm()
        Matrix exponential of quantum object.
    full(order='C')
        Returns dense array of quantum object `data` attribute.
    groundstate(sparse=False, tol=0, maxiter=100000)
        Returns eigenvalue and eigenket for the groundstate of a quantum
        object.
    inv()
        Return a Qobj corresponding to the matrix inverse of the operator.
    matrix_element(bra, ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns norm of a ket or an operator.
    permute(order)
        Returns composite qobj with indices reordered.
    proj()
        Computes the projector for a ket or bra vector.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
    sinm()
        Sine of quantum object.
    sqrtm()
        Matrix square root of quantum object.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    tr()
        Trace of quantum object.
    trans()
        Transpose of quantum object.
    transform(inpt, inverse=False)
        Performs a basis transformation defined by `inpt` matrix.
    trunc_neg(method='clip')
        Removes negative eigenvalues and returns a new Qobj that is
        a valid density operator.
    unit(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns normalized quantum object.

    """
    def _initialize_data(self, arg, dims, copy):
        if isinstance(arg, _data.Data):
            self.dims = dims or [[arg.shape[0]], [arg.shape[1]]]
            self._data = arg.copy() if copy else arg
        elif isinstance(arg, Qobj):
            self.dims = dims or arg.dims.copy()
            self._data = arg.copy() if copy else arg
        elif arg is None or isinstance(arg, numbers.Number):
            self.dims = dims or [[1], [1]]
            self._data = _data.csr.identity(1, scale=complex(arg or 0))
        else:
            self._data = _data.create(arg)
            self.dims = dims or [[self._data.shape[0]], [self._data.shape[1]]]
        #if self._data.shape != (np.prod(self.dims[0]), np.prod(self.dims[1])):
        #    raise ValueError("".join([
        #        "given dimensions ",
        #        repr(self.dims),
        #        " are not compatible with data shape ",
        #        repr(self.shape),
        #    ]))

    def __init__(self, arg=None, dims=None, type=None,
                 copy=True, superrep=None, isherm=None, isunitary=None):
        self._initialize_data(arg, dims, copy)
        self.type = type or type_from_dims(self.dims)
        self._isherm = isherm
        self._isunitary = isunitary

        if self.type == 'super' and type_from_dims(self.dims) == 'oper':
            if self._data.shape[0] != self._data.shape[1]:
                raise ValueError("".join([
                    "cannot build superoperator from nonsquare data of shape ",
                    repr(self._data.shape),
                ]))
            root = int(np.sqrt(self._data.shape[0]))
            if root * root != self._data.shape[0]:
                raise ValueError("".join([
                    "cannot build superoperator from nonsquare subspaces ",
                    "of size ",
                    repr(self._data.shape[0]),
                ]))
            self.dims = [[[root]]*2]*2
        if self.type == 'super':
            superrep = superrep or 'super'
        self.superrep = superrep

    def copy(self):
        """Create identical copy"""
        return Qobj(arg=self._data,
                    dims=self.dims,
                    type=self.type,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=True)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, _data.Data):
            raise TypeError('Qobj data must be a data-layer format.')
            self._data = data

    @_require_equal_type
    def __add__(self, other):
        """
        ADDITION with Qobj on LEFT [ ex. Qobj+4 ]
        """
        isherm = (self._isherm and other._isherm) or None
        return Qobj(_data.add_csr(self._data, other._data),
                    dims=self.dims,
                    type=self.type,
                    isherm=isherm,
                    copy=False)

    def __radd__(self, other): return self.__add__(other)

    @_require_equal_type
    def __sub__(self, other):
        """
        SUBTRACTION with Qobj on LEFT [ ex. Qobj-4 ]
        """
        isherm = (self._isherm and other._isherm) or None
        return Qobj(_data.sub_csr(self._data, other._data),
                    dims=self.dims,
                    type=self.type,
                    isherm=isherm,
                    copy=False)

    def __rsub__(self, other): return self.__neg__().__add__(other)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return self.__matmul__(other)
        multiplier = complex(other)
        isherm = (self._isherm and multiplier.imag == 0) or None
        isunitary = (self._isunitary and abs(multiplier) == 1) or None
        return Qobj(_data.mul_csr(self._data, complex(other)),
                    dims=self.dims,
                    type=self.type,
                    isherm=isherm,
                    isunitary=isunitary,
                    copy=False)

    def __rmul__(self, other):
        # Shouldn't be here unless `other.__mul__` has already been tried, so
        # we _shouldn't_ check that `other` is `Qobj`.
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self.__mul__(complex(other))

    def __matmul__(self, other):
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other)
            except TypeError:
                return NotImplemented
        if self.dims[1] != other.dims[0]:
            raise TypeError("".join([
                "incompatible dimensions ",
                repr(self.dims),
                " and ",
                repr(other.dims),
            ]))
        if (
            self.type == 'super'
            and other.type == 'super'
            and self.superrep != other.superrep
        ):
            raise TypeError("".join([
                "incompatible superoperator representations ",
                repr(self.superrep),
                " and ",
                repr(other.superrep),
            ]))
        if self.type == 'bra' and other.type == 'ket':
            return _data.inner_csr(self.data, other.data)
        return Qobj(_data.matmul_csr(self.data, other.data),
                    dims=[self.dims[0], other.dims[1]],
                    isherm=self._isherm and other._isherm,
                    isunitary=self._isunitary and other._isunitary,
                    superrep=self.superrep,
                    copy=False)

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self.__mul__(complex(other))

    def __neg__(self):
        return Qobj(_data.neg_csr(self._data),
                    dims=self.dims.copy(),
                    type=self.type,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def __getitem__(self, ind):
        out = self._data.as_scipy()[ind]
        return out.toarray() if scipy.sparse.issparse(out) else out

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Qobj) or self.dims != other.dims:
            return False
        diff = _data.sub_csr(self._data, other._data)
        return np.all(np.abs(diff.as_scipy().data) < settings.atol)

    def __ne__(self, other):
        return not (self == other)

    def __pow__(self, n, m=None):  # calculates powers of Qobj
        if (
            self.type not in ('oper', 'super')
            or self.dims[0] != self.dims[1]
            or m is not None
            or not isinstance(n, numbers.Integral)
            or n < 0
        ):
            return NotImplemented
        if n == 0:
            return Qobj(_data.csr.identity(self._data.shape[0]),
                        dims=self.dims,
                        type=self.type,
                        superrep=self.superrep,
                        isherm=True,
                        isunitary=True,
                        copy=False)
        elif n == 1:
            return self.copy()
        # TODO: move this to data layer.
        powers_two = [self._data]
        for i in range(n.bit_length() - 1):
            powers_two.append(_data.matmul_csr(powers_two[i], powers_two[i]))
        i = 0
        out_data = None
        for power in powers_two:
            if n % 2:
                out_data = (power if out_data is None
                            else _data.matmul_csr(out_data, power))
            n //= 2
        return Qobj(out_data,
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def __str__(self):
        s = ""
        t = self.type
        shape = self.shape
        if self.type in ['oper', 'super']:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t +
                  ", isherm = " + str(self.isherm) +
                  (
                      ", superrep = {0.superrep}".format(self)
                      if t == "super" and self.superrep != "super"
                      else ""
                  ) + "\n")
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t + "\n")
        s += "Qobj data =\n"

        if shape[0] > 10000 or shape[1] > 10000:
            # if the system is huge, don't attempt to convert to a
            # dense matrix and then to string, because it is pointless
            # and is likely going to produce memory errors. Instead print the
            # sparse data string representation
            s += str(self.data.as_scipy())

        elif all(np.imag(self.data.as_scipy().data) == 0):
            s += str(np.real(self.full()))

        else:
            s += str(self.full())

        return s

    def __repr__(self):
        # give complete information on Qobj without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    def __call__(self, other):
        """
        Acts this Qobj on another Qobj either by left-multiplication,
        or by vectorization and devectorization, as
        appropriate.
        """
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")

        if self.type == "super":
            if other.type == "ket":
                other = other.proj()

            if other.type == "oper":
                return vector_to_operator(self * operator_to_vector(other))
            else:
                raise TypeError("Can only act super on oper or ket.")

        elif self.type == "oper":
            if other.type == "ket":
                return self * other
            else:
                raise TypeError("Can only act oper on ket.")

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)

    def _repr_latex_(self):
        """
        Generate a LaTeX representation of the Qobj instance. Can be used for
        formatted output in ipython notebook.
        """
        t = self.type
        shape = self.shape
        s = r''
        if self.type in ['oper', 'super']:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t +
                  ", isherm = " + str(self.isherm) +
                  (
                      ", superrep = {0.superrep}".format(self)
                      if t == "super" and self.superrep != "super"
                      else ""
                  ))
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t)

        M, N = self.data.shape

        s += r'\begin{equation*}\left(\begin{array}{*{11}c}'

        def _format_float(value):
            if value == 0.0:
                return "0.0"
            elif abs(value) > 1000.0 or abs(value) < 0.001:
                return ("%.3e" % value).replace("e", r"\times10^{") + "}"
            elif abs(value - int(value)) < 0.001:
                return "%.1f" % value
            else:
                return "%.3f" % value

        def _format_element(m, n, d):
            s = " & " if n > 0 else ""
            if type(d) == str:
                return s + d
            else:
                if abs(np.imag(d)) < settings.atol:
                    return s + _format_float(np.real(d))
                elif abs(np.real(d)) < settings.atol:
                    return s + _format_float(np.imag(d)) + "j"
                else:
                    s_re = _format_float(np.real(d))
                    s_im = _format_float(np.imag(d))
                    if np.imag(d) > 0.0:
                        return (s + "(" + s_re + "+" + s_im + "j)")
                    else:
                        return (s + "(" + s_re + s_im + "j)")

        if M > 10 and N > 10:
            # truncated matrix output
            for m in range(5):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

            for n in range(5):
                s += _format_element(m, n, r'\vdots')
            s += r' & \ddots'
            for n in range(N - 5, N):
                s += _format_element(m, n, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        elif M > 10 and N <= 10:
            # truncated vertically elongated matrix output
            for m in range(5):
                for n in range(N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

            for n in range(N):
                s += _format_element(m, n, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                for n in range(N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        elif M <= 10 and N > 10:
            # truncated horizontally elongated matrix output
            for m in range(M):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        else:
            # full output
            for m in range(M):
                for n in range(N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        s += r'\end{array}\right)\end{equation*}'
        return s

    def dag(self):
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        return Qobj(_data.adjoint_csr(self._data),
                    dims=[self.dims[1], self.dims[0]],
                    type=_ADJOINT_TYPE_LOOKUP[self.type],
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def conj(self):
        """Get the element-wise conjugation of the quantum object."""
        return Qobj(_data.conj_csr(self._data),
                    dims=self.dims.copy(),
                    type=self.type,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def trans(self):
        """Get the matrix transpose of the quantum operator.

        Returns
        -------
        oper : :class:`.Qobj`
            Transpose of input operator.
        """
        return Qobj(_data.transpose_csr(self._data),
                    dims=[self.dims[1], self.dims[0]],
                    type=_ADJOINT_TYPE_LOOKUP[self.type],
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def dual_chan(self):
        """Dual channel of quantum object representing a completely positive
        map.
        """
        # Uses the technique of Johnston and Kribs (arXiv:1102.0948), which
        # is only valid for completely positive maps.
        if not self.iscp:
            raise ValueError("Dual channels are only implemented for CP maps.")
        J = to_choi(self)
        tensor_idxs = enumerate_flat(J.dims)
        J_dual = tensor_swap(J, *(
                list(zip(tensor_idxs[0][1], tensor_idxs[0][0])) +
                list(zip(tensor_idxs[1][1], tensor_idxs[1][0]))
        )).trans()
        J_dual.superrep = 'choi'
        return J_dual

    def norm(self, norm=None, sparse=False, tol=0, maxiter=100000):
        """Norm of a quantum object.

        Default norm is L2-norm for kets and trace-norm for operators.
        Other ket and operator norms may be specified using the `norm` and
        argument.

        Parameters
        ----------
        norm : str
            Which norm to use for ket/bra vectors: L2 'l2', max norm 'max',
            or for operators: trace 'tr', Frobius 'fro', one 'one', or max
            'max'.

        sparse : bool
            Use sparse eigenvalue solver for trace norm.  Other norms are not
            affected by this parameter.

        tol : float
            Tolerance for sparse solver (if used) for trace norm. The sparse
            solver may not converge if the tolerance is set too low.

        maxiter : int
            Maximum number of iterations performed by sparse solver (if used)
            for trace norm.

        Returns
        -------
        norm : float
            The requested norm of the operator or state quantum object.


        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        if self.type in ['oper', 'super']:
            if norm is None or norm == 'tr':
                _op = self*self.dag()
                vals = sp_eigs(_op.data, _op.isherm, vecs=False,
                               sparse=sparse, tol=tol, maxiter=maxiter)
                return np.sum(np.sqrt(np.abs(vals)))
            elif norm == 'fro':
                return sp_fro_norm(self.data)
            elif norm == 'one':
                return sp_one_norm(self.data)
            elif norm == 'max':
                return sp_max_norm(self.data)
            else:
                raise ValueError(
                    "For matrices, norm must be 'tr', 'fro', 'one', or 'max'.")
        else:
            if norm is None or norm == 'l2':
                return sp_L2_norm(self.data)
            elif norm == 'max':
                return sp_max_norm(self.data)
            else:
                raise ValueError("For vectors, norm must be 'l2', or 'max'.")

    def proj(self):
        """Form the projector from a given ket or bra vector.

        Parameters
        ----------
        Q : :class:`qutip.Qobj`
            Input bra or ket vector

        Returns
        -------
        P : :class:`qutip.Qobj`
            Projection operator.
        """
        if self.type != 'ket' and self.type != 'bra':
            raise TypeError("projection is only defined for bras and kets")
        dims = ([self.dims[0], self.dims[0]] if self.type == 'ket'
                else [self.dims[1], self.dims[1]])
        return Qobj(_data.project_csr(self._data),
                    dims=dims,
                    type='oper',
                    isherm=True,
                    copy=False)

    def tr(self):
        """Trace of a quantum object.

        Returns
        -------
        trace : float
            Returns the trace of the quantum object.

        """
        out = _data.trace_csr(self._data)
        return out.real if self.isherm else out

    def purity(self):
        """Calculate purity of a quantum object.

        Returns
        -------
        state_purity : float
            Returns the purity of a quantum object.
            For a pure state, the purity is 1.
            For a mixed state of dimension `d`, 1/d<=purity<1.

        """
        rho = self
        if (rho.type == "super"):
            raise TypeError('Purity is defined on a density matrix or state.')

        if rho.type == "ket" or rho.type == "bra":
            state_purity = (rho*rho.dag()).tr()

        if rho.type == "oper":
            state_purity = (rho*rho).tr()

        return state_purity

    def full(self, order='C', squeeze=False):
        """Dense array from quantum object.

        Parameters
        ----------
        order : str {'C', 'F'}
            Return array in C (default) or Fortran ordering.
        squeeze : bool {False, True}
            Squeeze output array.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.
        """
        out = np.asarray(self.data.to_array(), order=order)
        return out.squeeze() if squeeze else out

    def diag(self):
        """Diagonal elements of quantum object.

        Returns
        -------
        diags : array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.

        """
        out = self.data.diagonal()
        if np.any(np.imag(out) > settings.atol) or not self.isherm:
            return out
        else:
            return np.real(out)

    def expm(self, method='dense'):
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Parameters
        ----------
        method : str {'dense', 'sparse'}
            Use set method to use to calculate the matrix exponentiation. The
            available choices includes 'dense' and 'sparse'.  Since the
            exponential of a matrix is nearly always dense, method='dense'
            is set as default.s

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.

        """
        if self.dims[0][0] != self.dims[1][0]:
            raise TypeError('Invalid operand for matrix exponential')

        if method == 'dense':
            F = sp_expm(self.data.as_scipy(), sparse=False)

        elif method == 'sparse':
            F = sp_expm(self.data.as_scipy(), sparse=True)

        else:
            raise ValueError("method must be 'dense' or 'sparse'.")

        out = Qobj(F, dims=self.dims)
        return out.tidyup() if settings.auto_tidyup else out

    def check_herm(self):
        """Check if the quantum object is hermitian.

        Returns
        -------
        isherm : bool
            Returns the new value of isherm property.
        """
        self._isherm = None
        return self.isherm

    def sqrtm(self, sparse=False, tol=0, maxiter=100000):
        """Sqrt of a quantum operator.

        Operator must be square.

        Parameters
        ----------
        sparse : bool
            Use sparse eigenvalue/vector solver.
        tol : float
            Tolerance used by sparse solver (0 = machine precision).
        maxiter : int
            Maximum number of iterations used by sparse solver.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix square root of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        if self.dims[0][0] == self.dims[1][0]:
            evals, evecs = sp_eigs(self.data, self.isherm, sparse=sparse,
                                   tol=tol, maxiter=maxiter)
            numevals = len(evals)
            dV = scipy.sparse.spdiags(np.sqrt(evals, dtype=complex), 0,
                                      numevals, numevals,
                                      format='csr')
            if self.isherm:
                spDv = dV.dot(evecs.T.conj().T)
            else:
                spDv = dV.dot(np.linalg.inv(evecs.T))

            out = Qobj(evecs.T.dot(spDv), dims=self.dims)
            return out.tidyup() if settings.auto_tidyup else out

        else:
            raise TypeError('Invalid operand for matrix square root')

    def cosm(self):
        """Cosine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix cosine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.

        """
        if self.dims[0][0] == self.dims[1][0]:
            return 0.5 * ((1j * self).expm() + (-1j * self).expm())
        else:
            raise TypeError('Invalid operand for matrix square root')

    def sinm(self):
        """Sine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix sine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.
        """
        if self.dims[0][0] == self.dims[1][0]:
            return -0.5j * ((1j * self).expm() - (-1j * self).expm())
        else:
            raise TypeError('Invalid operand for matrix square root')

    def inv(self, sparse=False):
        """Matrix inverse of a quantum operator

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix inverse of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.
        """
        if self.shape[0] != self.shape[1]:
            raise TypeError('Invalid operand for matrix inverse')
        if sparse:
            inv_mat = scipy.sparse.linalg.inv(self.data.as_scipy().tocsc())
        else:
            inv_mat = np.linalg.inv(self.full())
        return Qobj(inv_mat, dims=self.dims[::-1])

    def unit(self, inplace=False, norm=None, sparse=False, tol=0, maxiter=100000):
        """Operator or state normalized to unity.

        Uses norm from Qobj.norm().

        Parameters
        ----------
        inplace : bool
            Do an in-place normalization
        norm : str
            Requested norm for states / operators.
        sparse : bool
            Use sparse eigensolver for trace norm. Does not affect other norms.
        tol : float
            Tolerance used by sparse eigensolver.
        maxiter : int
            Number of maximum iterations performed by sparse eigensolver.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Normalized quantum object if not in-place,
            else None.

        """
        if inplace:
            nrm = self.norm(norm=norm, sparse=sparse, tol=tol, maxiter=maxiter)

            self.data /= nrm
        elif not inplace:
            out = self / self.norm(norm=norm, sparse=sparse,
                                   tol=tol, maxiter=maxiter)
            if settings.auto_tidyup:
                return out.tidyup()
            else:
                return out
        else:
            raise Exception('inplace kwarg must be bool.')

    def ptrace(self, sel, sparse=None):
        """Partial trace of the quantum object.

        Parameters
        ----------
        sel : int/list
            An ``int`` or ``list`` of components to keep after partial trace.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Quantum object representing partial trace with selected components
            remaining.

        Notes
        -----
        This function is identical to the :func:`qutip.qobj.ptrace` function
        that has been deprecated.

        """
        if sparse is None:
            if self.isket:
                sparse = False
            elif (self.data.nnz / (self.shape[0] * self.shape[1])) >= 0.1:
                sparse = False
        if sparse:
            q = Qobj()
            q.data, q.dims, _ = _ptrace(self, sel)
            return q.tidyup() if settings.auto_tidyup else q
        else:
            return _ptrace_dense(self, sel)

    def permute(self, order):
        """Permutes a composite quantum object.

        Parameters
        ----------
        order : list/array
            List specifying new tensor order.

        Returns
        -------
        P : :class:`qutip.Qobj`
            Permuted quantum object.

        """
        q = Qobj()
        q.data, q.dims = _permute(self, order)
        q.data.sort_indices()
        return q

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup. Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Quantum object with small elements removed.

        """
        atol = atol or settings.auto_tidyup_atol
        nnz = _data.csr.nnz(self.data)
        if nnz:
            # This does the tidyup and returns True if
            # The sparse data needs to be shortened
            if use_openmp() and nnz > 500:
                if omp_tidyup(self.data.data, atol, nnz,
                              settings.num_cpus):
                    self.data.eliminate_zeros()
            else:
                if cy_tidyup(self.data.data, atol, nnz):
                    self.data.eliminate_zeros()
            return self
        else:
            return self

    def transform(self, inpt, inverse=False, sparse=True):
        """Basis transform defined by input array.

        Input array can be a ``matrix`` defining the transformation,
        or a ``list`` of kets that defines the new basis.

        Parameters
        ----------
        inpt : array_like
            A ``matrix`` or ``list`` of kets defining the transformation.
        inverse : bool
            Whether to return inverse transformation.
        sparse : bool
            Use sparse matrices when possible. Can be slower.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Operator in new basis.

        Notes
        -----
        This function is still in development.
        """
        if isinstance(inpt, list) or (isinstance(inpt, np.ndarray) and
                                      len(inpt.shape) == 1):
            if len(inpt) != max(self.shape):
                raise TypeError(
                    'Invalid size of ket list for basis transformation')
            if sparse:
                S = scipy.sparse.hstack([psi.data.as_scipy() for psi in inpt],
                                        format='csr', dtype=complex).conj().T
            else:
                S = np.hstack([psi.full() for psi in inpt],
                              dtype=complex).conj().T
        elif isinstance(inpt, Qobj) and inpt.isoper:
            S = inpt.data
        elif isinstance(inpt, np.ndarray):
            S = inpt.conj()
            sparse = False
        else:
            raise TypeError('Invalid operand for basis transformation')

        # transform data
        if inverse:
            if self.isket:
                data = (S.conj().T) * self.data
            elif self.isbra:
                data = self.data.dot(S)
            else:
                if sparse:
                    data = (S.conj().T) * self.data * S
                else:
                    data = (S.conj().T).dot(self.data.dot(S))
        else:
            if self.isket:
                data = S * self.data
            elif self.isbra:
                data = self.data.dot(S.conj().T)
            else:
                if sparse:
                    data = S * self.data * (S.conj().T)
                else:
                    data = S.dot(self.data.dot(S.conj().T))
        return Qobj(data,
                    dims=self.dims,
                    type=self.type,
                    isherm=self._isherm,
                    superrep=self.superrep)

    def trunc_neg(self, method="clip"):
        """Truncates negative eigenvalues and renormalizes.

        Returns a new Qobj by removing the negative eigenvalues
        of this instance, then renormalizing to obtain a valid density
        operator.


        Parameters
        ----------
        method : str
            Algorithm to use to remove negative eigenvalues. "clip"
            simply discards negative eigenvalues, then renormalizes.
            "sgs" uses the SGS algorithm (doi:10/bb76) to find the
            positive operator that is nearest in the Shatten 2-norm.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            A valid density operator.
        """
        if not self.isherm:
            raise ValueError("Must be a Hermitian operator to remove negative "
                             "eigenvalues.")

        if method not in ('clip', 'sgs'):
            raise ValueError("Method {} not recognized.".format(method))

        eigvals, eigstates = self.eigenstates()
        if all(eigval >= 0 for eigval in eigvals):
            # All positive, so just renormalize.
            return self.unit()
        idx_nonzero = eigvals != 0
        eigvals = eigvals[idx_nonzero]
        eigstates = eigstates[idx_nonzero]

        if method == 'clip':
            eigvals[eigvals < 0] = 0
        elif method == 'sgs':
            eigvals = eigvals[::-1]
            eigstates = eigstates[::-1]
            acc = 0.0
            n_eigs = len(eigvals)
            for idx in reversed(range(n_eigs)):
                if eigvals[idx] + acc / (idx + 1) >= 0:
                    break
                acc += eigvals[idx]
                eigvals[idx] = 0.0
            eigvals[:idx+1] += acc / (idx + 1)
        out_data = _data.csr.zeros(*self.shape)
        for value, state in zip(eigvals, eigstates):
            if value:
                out_data = _data.add_csr(out_data, state.data, value)
        return Qobj(out_data,
                    dims=self.dims.copy(),
                    type=self.type,
                    isherm=True,
                    copy=False).unit()

    def matrix_element(self, bra, ket):
        """Calculates a matrix element.

        Gives the matrix element for the quantum object sandwiched between a
        `bra` and `ket` vector.

        Parameters
        -----------
        bra : :class:`qutip.Qobj`
            Quantum object of type 'bra' or 'ket'

        ket : :class:`qutip.Qobj`
            Quantum object of type 'ket'.

        Returns
        -------
        elem : complex
            Complex valued matrix element.

        Note
        ----
        It is slightly more computationally efficient to use a ket
        vector for the 'bra' input.

        """
        if self.type != 'oper':
            raise TypeError("Can only get matrix elements for an operator.")
        if bra.type not in ('bra', 'ket') or ket.type not in ('bra', 'ket'):
            msg = "Can only calculate matrix elements between a bra and a ket."
            raise TypeError(msg)
        left, op, right = bra._data, self._data, ket._data
        if ket.type == 'bra':
            right = _data.adjoint_csr(right)
        scalar_is_ket = bra.type != 'bra'
        return _data.inner_op_zcsr(left, op, right, scalar_is_ket)

    def overlap(self, other):
        """Overlap between two state vectors or two operators.

        Gives the overlap (inner product) between the current bra or ket Qobj
        and and another bra or ket Qobj. It gives the Hilbert-Schmidt overlap
        when one of the Qobj is an operator/density matrix.

        Parameters
        -----------
        other : :class:`qutip.Qobj`
            Quantum object for a state vector of type 'ket', 'bra' or density
            matrix.

        Returns
        -------
        overlap : complex
            Complex valued overlap.

        Raises
        ------
        TypeError
            Can only calculate overlap between a bra, ket and density matrix
            quantum objects.

        Notes
        -----
        Since QuTiP mainly deals with ket vectors, the most efficient inner
        product call is the ket-ket version that computes the product
        <self|other> with both vectors expressed as kets.
        """
        if not isinstance(other, Qobj):
            raise TypeError("".join([
                "cannot calculate overlap with non-quantum object ",
                repr(other),
            ]))
        if (
            self.type not in ('ket', 'bra', 'oper')
            or other.type not in ('ket', 'bra', 'oper')
        ):
            msg = "only bras, kets and density matrices have defined overlaps"
            raise TypeError(msg)
        left, right = _data.adjoint_csr(self._data), other._data
        if self.type == 'oper' or other.type == 'oper':
            if self.type != 'oper':
                left = _data.project_csr(left)
            if other.type != 'oper':
                right = _data.project_csr(right)
            return _data.trace_csr(_data.matmul_csr(left, right))
        scalar_is_ket = self.type != 'bra'
        if other.type == 'bra':
            right = _data.adjoint_csr(right)
        return _data.inner_csr(left, right, scalar_is_ket)

    def eigenstates(self, sparse=False, sort='low', eigvals=0,
                    tol=0, maxiter=100000, phase_fix=None):
        """Eigenstates and eigenenergies.

        Eigenstates and eigenenergies are defined for operators and
        superoperators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver

        sort : str
            Sort eigenvalues (and vectors) 'low' to high, or 'high' to low.

        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.

        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.

        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        phase_fix : int, None
            If not None, set the phase of each kets so that ket[phase_fix,0]
            is real positive.

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        eigvecs : array
            Array of quantum operators representing the oprator eigenkets.
            Order of eigenkets is determined by order of eigenvalues.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        evals, evecs = sp_eigs(self.data, self.isherm, sparse=sparse,
                               sort=sort, eigvals=eigvals, tol=tol,
                               maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        ekets = np.array([Qobj(vec, dims=new_dims) for vec in evecs],
                         dtype=object)
        norms = np.array([ket.norm() for ket in ekets])
        if phase_fix is None:
            phase = np.array([1] * len(ekets))
        else:
            phase = np.array([np.abs(ket[phase_fix, 0]) / ket[phase_fix, 0]
                              if ket[phase_fix, 0] else 1
                              for ket in ekets])
        return evals, ekets / norms * phase

    def eigenenergies(self, sparse=False, sort='low',
                      eigvals=0, tol=0, maxiter=100000):
        """Eigenenergies of a quantum object.

        Eigenenergies (eigenvalues) are defined for operators or superoperators
        only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        sort : str
            Sort eigenvalues 'low' to high, or 'high' to low.
        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.
        tol : float
            Tolerance used by sparse Eigensolver (0=machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        return sp_eigs(self.data, self.isherm, vecs=False, sparse=sparse,
                       sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)

    def groundstate(self, sparse=False, tol=0, maxiter=100000, safe=True):
        """Ground state Eigenvalue and Eigenvector.

        Defined for quantum operators or superoperators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).
        safe : bool (default=True)
            Check for degenerate ground state

        Returns
        -------
        eigval : float
            Eigenvalue for the ground state of quantum operator.
        eigvec : :class:`qutip.Qobj`
            Eigenket for the ground state of quantum operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        if safe:
            evals = 2
        else:
            evals = 1
        grndval, grndvec = sp_eigs(self.data, self.isherm, sparse=sparse,
                                   eigvals=evals, tol=tol, maxiter=maxiter)
        if safe:
            tol = tol or settings.atol
            if (grndval[1]-grndval[0]) <= 10*tol:
                print("WARNING: Ground state may be degenerate. "
                        "Use Q.eigenstates()")
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        grndvec = Qobj(grndvec[0], dims=new_dims)
        grndvec = grndvec / grndvec.norm()
        return grndval[0], grndvec

    def dnorm(self, B=None):
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`qutip.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return mts.dnorm(self, B)

    @property
    def ishp(self):
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type in ["super", "oper"]:
            try:
                J = to_choi(self)
                return J.isherm
            except:
                return False
        else:
            return False

    @property
    def iscp(self):
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type not in ["super", "oper"]:
            return False
        # We can test with either Choi or chi, since the basis
        # transformation between them is unitary and hence preserves
        # the CP and TP conditions.
        J = self if self.superrep in ('choi', 'chi') else to_choi(self)
        # If J isn't hermitian, then that could indicate either that J is not
        # normal, or is normal, but has complex eigenvalues.  In either case,
        # it makes no sense to then demand that the eigenvalues be
        # non-negative.
        return J.isherm and np.all(J.eigenenergies() >= -settings.atol)

    @property
    def istp(self):
        if self.type not in ['super', 'oper']:
            return False
        # Normalize to a super of type choi or chi.
        # We can test with either Choi or chi, since the basis
        # transformation between them is unitary and hence
        # preserves the CP and TP conditions.
        if self.type == "super" and self.superrep in ('choi', 'chi'):
            qobj = self
        else:
            qobj = to_choi(self)
        # Possibly collapse dims.
        if any([len(index) > 1
                for super_index in qobj.dims
                for index in super_index]):
            qobj = Qobj(qobj, dims=collapse_dims_super(qobj.dims))

        # We use the condition from John Watrous' lecture notes,
        # Tr_1(J(Phi)) = identity_2.
        tr_oper = qobj.ptrace([0])
        return np.allclose(tr_oper.full(), np.eye(2), atol=settings.atol)

    @property
    def iscptp(self):
        if self.type != 'super' and self.type != 'oper':
            return False
        reps = ('choi', 'chi')
        q_oper = to_choi(self) if self.superrep not in reps else self
        return q_oper.iscp and q_oper.istp

    @property
    def isherm(self):
        if self._isherm is not None:
            return self._isherm
        self._isherm = _data.isherm_csr(self._data)
        return self._isherm

    @isherm.setter
    def isherm(self, isherm):
        self._isherm = isherm

    def _calculate_isunitary(self):
        """
        Checks whether qobj is a unitary matrix
        """
        if self.type != 'oper' or self._data.shape[0] != self._data.shape[1]:
            return False
        iden = _data.csr.identity(self.shape[0])
        cmp = _data.matmul_csr(self._data, _data.adjoint_csr(self._data))
        diff = _data.sub_csr(cmp, iden)
        return np.all(np.abs(diff.as_scipy().data) < settings.atol)

    @property
    def isunitary(self):
        if self._isunitary is not None:
            return self._isunitary
        self._isunitary = self._calculate_isunitary()
        return self._isunitary

    @property
    def shape(self): return self.data.shape

    isbra = property(isbra)

    @property
    def isket(self): return self.type == 'ket'

    @property
    def isoperbra(self): return self.type == 'operator-bra'

    @property
    def isoperket(self): return self.type == 'operator-ket'

    @property
    def isoper(self): return self.type == 'oper'

    @property
    def issuper(self): return self.type == 'super'


def _ptrace_dense(Q, sel):
    rd = np.asarray(Q.dims[0], dtype=np.int32).ravel()
    nd = rd.shape[0]
    sel = [sel] if isinstance(sel, int) else list(np.sort(sel))
    dkeep = rd[sel].tolist()
    qtrace = list(set(np.arange(nd)) - set(sel))
    dtrace = rd[qtrace].tolist()
    rd = list(rd)
    if Q.type == 'ket':
        vmat = (Q.full()
                .reshape(rd)
                .transpose(sel + qtrace)
                .reshape([np.prod(dkeep), np.prod(dtrace)]))
        rhomat = vmat.dot(vmat.conj().T)
    else:
        rhomat = np.trace(Q.full()
                          .reshape(rd + rd)
                          .transpose(qtrace + [nd + q for q in qtrace] +
                                     sel + [nd + q for q in sel])
                          .reshape([np.prod(dtrace),
                                    np.prod(dtrace),
                                    np.prod(dkeep),
                                    np.prod(dkeep)]))
    return Qobj(rhomat, dims=[dkeep, dkeep])


def ptrace(Q, sel):
    """Partial trace of the Qobj with selected components remaining.

    Parameters
    ----------
    Q : :class:`qutip.Qobj`
        Composite quantum object.
    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    Returns
    -------
    oper : :class:`qutip.Qobj`
        Quantum object representing partial trace with selected components
        remaining.

    Notes
    -----
    This function is for legacy compatibility only. It is recommended to use
    the ``ptrace()`` Qobj method.

    """
    if not isinstance(Q, Qobj):
        raise TypeError("Input is not a quantum object")
    return Q.ptrace(sel)


# TRAILING IMPORTS
# We do a few imports here to avoid circular dependencies.
from qutip.core.superop_reps import to_choi
from qutip.core.superoperator import vector_to_operator, operator_to_vector
from qutip.core.tensor import tensor_swap
from qutip.core import metrics as mts
