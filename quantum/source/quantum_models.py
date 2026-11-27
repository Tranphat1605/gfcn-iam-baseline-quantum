"""Quanvolution + upstream GFCN trunk + dressed VQC classification head.

The GFCN trunk is inherited from the upstream CeCILL-C implementation.
Quantum circuit details and the HTR adaptation are experiment design choices.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from source.models import GFCN
from source.quantum_ops import (circuit_unitary, cnot_matrix, measure_z,
                                product_ry_state, z_signs)


class FixedQuanvolution4(nn.Module):
    """Recover pixels from baseline normalization, then apply a fixed 4q filter.

    Patch order: top-left, top-right, bottom-left, bottom-right. Stride 1;
    pad only the bottom/right with white background (ink value zero).
    The trainable GFCN receives 4 channels of ideal Z expectation values.
    """
    def __init__(self, input_mean, input_std, depth=2, seed=0, chunk_size=4096):
        super().__init__()
        if input_std <= 0 or depth < 1 or chunk_size < 1:
            raise ValueError("Invalid normalization/circuit configuration")
        generator = torch.Generator().manual_seed(seed)
        angles = 2 * math.pi * torch.rand((depth, 4, 2), generator=generator)
        cnots = torch.stack([cnot_matrix(4, k, (k + 1) % 4) for k in range(4)])
        self.register_buffer("angles", angles)
        self.register_buffer("unitary", circuit_unitary(angles.double(), cnots).to(torch.complex64))
        self.register_buffer("z_signs", z_signs(4))
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        self.chunk_size = chunk_size

    @torch.no_grad()
    def forward(self, normalized):
        if normalized.ndim != 4 or normalized.shape[1] != 1:
            raise ValueError("Quanvolution expects normalized grayscale BCHW input")
        b, _, h, w = normalized.shape
        # Original loader uses normalized padding value 300; clamp maps it to white.
        pixels = ((normalized * self.input_std + self.input_mean) / 255).clamp(0, 1)
        ink = 1 - pixels
        patches = F.unfold(F.pad(ink, (0, 1, 0, 1), value=0), kernel_size=2)
        patches = patches.transpose(1, 2).reshape(-1, 4)
        outputs = normalized.new_empty((len(patches), 4))
        for start in range(0, len(patches), self.chunk_size):
            batch = patches[start:start + self.chunk_size]
            state = product_ry_state(math.pi * batch)
            outputs[start:start + len(batch)] = measure_z(
                state, self.unitary.to(state.dtype), self.z_signs)
        return outputs.reshape(b, h, w, 4).permute(0, 3, 1, 2).contiguous()


class DressedVQCHead(nn.Module):
    """A shared 4-qubit classifier for every GFCN time frame, before CTC.

    256 -> 4 -> H/RY encoding -> trainable RZ/RY/CNOT layers -> <Z> -> 80.
    This is inspired by Mari's dressed circuit, not their exact circuit/task.
    """
    def __init__(self, in_features, classes, n_qubits=4, depth=2):
        super().__init__()
        if n_qubits != 4 or depth < 1:
            raise ValueError("This experiment uses four qubits and a positive circuit depth")
        self.n_qubits = n_qubits
        self.pre_net = nn.Linear(in_features, n_qubits)
        self.theta = nn.Parameter(0.1 * torch.randn(depth, n_qubits, 2))
        self.post_net = nn.Linear(n_qubits, classes)
        self.register_buffer("cnots", torch.stack([
            cnot_matrix(n_qubits, k, (k + 1) % n_qubits) for k in range(n_qubits)]))
        self.register_buffer("z_signs", z_signs(n_qubits))

    def quantum_features(self, angles):
        state = product_ry_state(angles, hadamard=True)
        unitary = circuit_unitary(self.theta.to(angles.dtype), self.cnots)
        return measure_z(state, unitary, self.z_signs)

    def forward(self, features):
        if features.ndim != 4 or features.shape[2] != 1:
            raise ValueError("VQC head expects B x features x 1 x time")
        b, channels, _, steps = features.shape
        frames = features.squeeze(2).transpose(1, 2).reshape(-1, channels)
        angles = (math.pi / 2) * torch.tanh(self.pre_net(frames))
        logits = self.post_net(self.quantum_features(angles))
        return logits.reshape(b, steps, -1).transpose(1, 2).unsqueeze(2).contiguous()


class DeviceGaussianNoise(nn.Module):
    """Same noise distribution as upstream, also usable by CPU smoke checks."""
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std if self.training else x


class QuantumGFCN(GFCN):
    def __init__(self, params):
        trunk_params = dict(params, input_channels=4)
        super().__init__(trunk_params)
        self.quanvolution = FixedQuanvolution4(
            params["quantum_input_mean"], params["quantum_input_std"],
            depth=params.get("quanvolution_depth", 2),
            seed=params.get("quanvolution_seed", 0),
            chunk_size=params.get("quantum_chunk_size", 4096))
        self.gn = DeviceGaussianNoise()
        self.end_conv = DressedVQCHead(256, params["vocab_size"] + 1,
                                     depth=params.get("vqc_depth", 2))

    def forward(self, x):
        return super().forward(self.quanvolution(x))


class QuanvolutionOnlyGFCN(GFCN):
    """Frozen 4q input filter with the upstream classical GFCN/CTC head."""
    def __init__(self, params):
        super().__init__(dict(params, input_channels=4))
        self.quanvolution = FixedQuanvolution4(
            params['quantum_input_mean'], params['quantum_input_std'],
            depth=params.get('quanvolution_depth', 2),
            seed=params.get('quanvolution_seed', 0),
            chunk_size=params.get('quantum_chunk_size', 4096))
        self.gn = DeviceGaussianNoise()

    def forward(self, x):
        return super().forward(self.quanvolution(x))
