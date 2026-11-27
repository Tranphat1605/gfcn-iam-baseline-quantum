"""Small, batched, differentiable ideal-state quantum simulations in PyTorch.

Wire 0 is the most significant computational-basis bit, matching PennyLane.
No shots, quantum hardware, or claimed quantum speedup are involved.
"""
import math
import torch


def complex_dtype(dtype):
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def ry(angle):
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    return torch.stack((c, -s, s, c)).reshape(2, 2).to(complex_dtype(angle.dtype))


def rz(angle):
    values = torch.stack((torch.exp(-0.5j * angle), torch.exp(0.5j * angle)))
    return torch.diag(values)


def on_wire(gate, wire, n_qubits):
    result = torch.ones((1, 1), device=gate.device, dtype=gate.dtype)
    identity = torch.eye(2, device=gate.device, dtype=gate.dtype)
    for k in range(n_qubits):
        result = torch.kron(result, gate if k == wire else identity)
    return result


def cnot_matrix(n_qubits, control, target, dtype=torch.complex64):
    indices = torch.arange(2 ** n_qubits)
    active = (indices >> (n_qubits - 1 - control)) & 1
    permutation = indices ^ (active << (n_qubits - 1 - target))
    result = torch.zeros((len(indices), len(indices)), dtype=dtype)
    result[permutation, indices] = 1
    return result


def z_signs(n_qubits):
    indices = torch.arange(2 ** n_qubits)
    return torch.stack([1 - 2 * ((indices >> (n_qubits - 1 - k)) & 1)
                        for k in range(n_qubits)], dim=1).float()


def product_ry_state(angles, hadamard=False):
    """RY encoding for (N,n) angles; optional H before RY on each wire."""
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    if hadamard:
        local = torch.stack(((c - s) / math.sqrt(2), (c + s) / math.sqrt(2)), dim=-1)
    else:
        local = torch.stack((c, s), dim=-1)
    state = torch.ones((len(angles), 1), dtype=angles.dtype, device=angles.device)
    for wire in range(angles.shape[1]):
        state = (state.unsqueeze(-1) * local[:, wire].unsqueeze(1)).reshape(len(angles), -1)
    return state.to(complex_dtype(angles.dtype))


def circuit_unitary(theta, cnots):
    """Each layer applies RZ then RY to each wire, followed by a CNOT ring.

    theta[...,0] is the Y angle, theta[...,1] the Z angle. Putting RY after
    RZ avoids a final trainable Z-only phase invisible to Z measurements.
    """
    n_qubits = theta.shape[1]
    dtype = complex_dtype(theta.dtype)
    unitary = torch.eye(2 ** n_qubits, dtype=dtype, device=theta.device)
    for layer in theta:
        for wire in range(n_qubits):
            local = ry(layer[wire, 0]) @ rz(layer[wire, 1])
            unitary = on_wire(local, wire, n_qubits) @ unitary
        for gate in cnots:
            unitary = gate.to(device=theta.device, dtype=dtype) @ unitary
    return unitary


def measure_z(state, unitary, signs):
    output = state @ unitary.transpose(0, 1)
    probabilities = output.real.square() + output.imag.square()
    return probabilities @ signs.to(device=probabilities.device, dtype=probabilities.dtype)
