import numpy as np
import scipy.sparse
import pytest

import qutip


def shuffle_indices_scipy_csr(matrix):
    """
    Given a scipy.sparse.csr_matrix, shuffle the indices within each row and
    return a new array.  This should represent the same matrix, but in the less
    efficient, "unsorted" manner.  All mathematical operations should still
    work the same after this, but may be slower.

    This is not guaranteed to change the order of the indices in every case.
    If there is at most one value per row, there is no unsorted order.  In
    general, we attempt to shuffle, and if this returns the same order as
    before, we just reverse it to ensure it's different.
    """
    out = matrix.copy()
    for row in range(out.shape[0]):
        ptr = (out.indptr[row], out.indptr[row + 1])
        if ptr[1] - ptr[0] > 1:
            order = np.argsort(np.random.rand(ptr[1] - ptr[0]))
            # If sorted, reverse it.
            order = np.flip(order) if np.all(order[:-1] < order[1:]) else order
            out.indices[ptr[0]:ptr[1]] = out.indices[ptr[0]:ptr[1]][order]
            out.data[ptr[0]:ptr[1]] = out.data[ptr[0]:ptr[1]][order]
    return out


def random_scipy_csr(shape, density, sorted_):
    """
    Generate a random scipy CSR matrix with the given shape, nnz density, and
    with indices that are either sorted or unsorted.  The nnz elements will
    always be at least one.
    """
    nnz = int(shape[0] * shape[1] * density) or 1
    data = np.random.rand(nnz) + 1j*np.random.rand(nnz)
    rows = np.random.choice(np.arange(shape[0]), nnz)
    cols = np.random.choice(np.arange(shape[1]), nnz)
    sci = scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    if not sorted_:
        shuffle_indices_scipy_csr(sci)
    return sci


def random_numpy_dense(shape, fortran):
    """Generate a random numpy dense matrix with the given shape."""
    out = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    if fortran:
        out = np.asfortranarray(out)
    return out


def random_csr(shape, density, sorted_):
    """
    Generate a random qutip CSR matrix with the given shape, nnz density, and
    with indices that are either sorted or unsorted.  The nnz elements will
    always be at least one (use data.csr.zeros otherwise).
    """
    return qutip.core.data.CSR(random_scipy_csr(shape, density, sorted_))


def random_dense(shape, fortran):
    """Generate a random qutip Dense matrix of the given shape."""
    return qutip.core.data.Dense(random_numpy_dense(shape, fortran))


# Set up some fixtures for automatic parametrisation.

@pytest.fixture(params=[
    pytest.param((1, 5), id='ket'),
    pytest.param((5, 1), id='bra'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
def shape(request): return request.param


@pytest.fixture(params=[True, False], ids=['Fortran', 'C'])
def fortran(request): return request.param


@pytest.fixture(params=[0.001, 1], ids=['sparse', 'dense'])
def density(request): return request.param


@pytest.fixture(params=[True, False], ids=['sorted', 'unsorted'])
def sorted_(request): return request.param


@pytest.fixture(scope='function')
def numpy_dense(shape, fortran):
    return random_numpy_dense(shape, fortran)


@pytest.fixture(scope='function')
def data_dense(shape, fortran):
    return random_dense(shape, fortran)


@pytest.fixture(scope='function')
def scipy_csr(shape, density, sorted_):
    return random_scipy_csr(shape, density, sorted_)


@pytest.fixture(scope='function')
def data_csr(shape, density, sorted_):
    return random_csr(shape, density, sorted_)
