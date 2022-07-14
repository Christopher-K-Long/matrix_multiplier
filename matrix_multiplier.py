import numpy as np


def sv_propagate_local_qubit_operator_v2(local_operator, state, qidxs, N):
    # Find qubit order
    qidxs = np.array(qidxs)
    idx = np.argsort(qidxs)
    # Sort qubits
    sorted = qidxs[idx]
    # Calculate shape that seperates the dimensions upon which qubits are acted
    rshift = np.append(sorted, N)
    lshift = np.insert(sorted + 1, 0, 0)
    del sorted
    n = len(qidxs)
    shape = np.empty(2*n + 1, np.int64)
    shape[1::2] = 2
    shape[::2] = np.power(2, rshift - lshift)
    del rshift, lshift
    # Inversing the sort
    idx = idx.argsort()*2+1
    # Reorder qubits
    nR = np.arange(n)
    state = np.moveaxis(state.reshape(shape), idx, nR)
    # Record shape for later
    shape = state.shape
    # Reshape into matrix
    state = state.reshape((2**n, 2**(N-n)))

    # Apply operator
    state = local_operator @ state

    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), nR, idx).reshape(2**N)
    return state

def dm_propagate_local_qubit_operator_v2(local_operator, state, qidxs, N):
    # Find qubit order
    qidxs = np.array(qidxs)
    idx = np.argsort(qidxs)
    # Sort qubits
    sorted = qidxs[idx]
    # Calculate shape that seperates the dimensions upon which qubits are acted
    rshift = np.append(sorted, N)
    lshift = np.insert(sorted + 1, 0, 0)
    del sorted
    n = len(qidxs)
    M = 2*n
    shape = np.empty(M + 1, np.int64)
    shape[1::2] = 2
    shape[::2] = np.power(2, rshift - lshift)
    del lshift, rshift
    shape = np.concatenate([shape, shape])
    # Inversing the sort
    idx = idx.argsort()
    idx = np.concatenate([idx*2+1, (idx+n)*2+2])
    # Reorder qubits
    MR = np.arange(M)
    state = np.moveaxis(state.reshape(shape), idx, MR)
    # Record shape for later
    shape = state.shape
    # Reshape into matrix
    m = 2**(M)
    state = state.reshape((m, 2**(2*(N-n))))
    local_operator = local_operator.reshape((m, m))

    # Apply operator
    state = local_operator @ state

    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), MR, idx).reshape((2**N,)*2)
    return state


def sv_propagate_local_qubit_operator_v3(local_operator, state, qidxs, N):
    # Reshape and reorder qubits
    shape = (2,)*N
    state = state.reshape(shape)
    n = len(qidxs)
    ns = np.arange(n)
    state = np.moveaxis(state, qidxs, ns).reshape((2**n, 2**(N-n)))
    # Apply operator
    state = local_operator @ state
    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), ns, qidxs).reshape(2**N)
    return state

def dm_propagate_local_qubit_operator_v3(local_operator, state, qidxs, N):
    # Reshape and reorder qubits
    Nt = 2*N
    shape = (2,)*(Nt)
    state = state.reshape(shape)
    n = len(qidxs)
    nt = 2*n
    ns = np.arange(nt)
    axes = np.concatenate([qidxs, N+qidxs])
    state = np.moveaxis(state, axes, ns)
    M = 2**nt
    state = state.reshape((M, 2**(Nt-nt)))
    # Apply operator
    state = local_operator.reshape((M,)*2) @ state
    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), ns, axes)
    state = state.reshape((2**N,)*2)
    return state



def sv_propagate_local_qubit_operator_v4(local_operator, state, qidxs, N):
    # Reshape and reorder qubits
    shape = (2,)*N
    state = state.reshape(shape)
    n = len(qidxs)
    ns = np.arange(n)
    order = [k for k in range(N) if k not in qidxs]
    for dest, src in zip(ns, qidxs):
        order.insert(dest, src)
    state = state.transpose(order).reshape((2**n, 2**(N-n)))
    # Apply operator
    state = local_operator @ state
    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), ns, qidxs).reshape(2**N)
    return state

def dm_propagate_local_qubit_operator_v4(local_operator, state, qidxs, N):
    # Reshape and reorder qubits
    Nt = 2*N
    shape = (2,)*(Nt)
    state = state.reshape(shape)
    n = len(qidxs)
    nt = 2*n
    ns = np.arange(nt)
    axes = np.concatenate([qidxs, N+qidxs])
    state = np.moveaxis(state, axes, ns)
    M = 2**nt
    state = state.reshape((M, 2**(Nt-nt)))
    # Apply operator
    state = local_operator.reshape((M,)*2) @ state
    # Reshape and regorder qubits
    state = np.moveaxis(state.reshape(shape), ns, axes)
    state = state.reshape((2**N,)*2)
    return state