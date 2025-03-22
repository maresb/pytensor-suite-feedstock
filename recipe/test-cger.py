import numpy as np
import pytensor
from pytensor import tensor as pt

x = pt.dvector("x")
y = pt.dvector("y")
A = pt.dmatrix("A")
a = pt.dscalar("a")
out = pt.blas_c.cger_no_inplace(A, a, x, y)
func = pytensor.function([A, a, x, y], out)
func(np.ones((2, 2)), 1, np.ones(2), np.ones(2))

out = pt.blas_c.cger_inplace(A, a, x, y)
func = pytensor.function([a, x, y, A], out, accept_inplace=True)
