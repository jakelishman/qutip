from . import dense
from . import csr

from .dense import Dense
from .csr import CSR
from .base import Data

from .add import *
from .adjoint import *
from .inner import *
from .kron import *
from .matmul import *
from .mul import *
from .project import *
from .properties import *
from .reshape import *
from .sub import *
from .trace import *


# This is completely temporary - it will actually get replaced by a proper
# dispatcher in its own module, but for now I'm just trying to get CSR working
# within Qobj, and this is the fastest way to stub out this creation.

def create(arg):
    import numpy as np
    import scipy.sparse

    if isinstance(arg, CSR):
        return arg.copy()
    if scipy.sparse.issparse(arg):
        return CSR(arg.tocsr())
    return CSR(scipy.sparse.csr_matrix(np.array(arg)))
