import numpy as np
import matrix_multiplier as mm

n = 2
N = 4

local_operator = np.random.rand(2**n, 2**n)
state = np.random.rand(2**N)
#density = np.random.rand(2**N, 2**N)
qidxs = np.random.choice(N, n, replace=False)

mm.sv_propagate_local_qubit_operator(local_operator, state, qidxs, N)