import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None

PAULIS = (
    torch.from_numpy(np.identity(2, dtype=complex)).to(C_DTYPE),
    torch.from_numpy(np.array([[0, 1], [1, 0]], dtype=complex)).to(C_DTYPE),
    torch.from_numpy(np.array([[0, -1j], [1j, 0]], dtype=complex)).to(C_DTYPE),
    torch.from_numpy(np.array([[1, 0], [0, -1]], dtype=complex)).to(C_DTYPE),
)

def dp1_matrix(params):
    """Compute unitary matrix for SU2 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    amat = (1-params.sum())*torch.kron(PAULIS[0], PAULIS[0])
    for ii in range(3):
        pp = torch.kron(PAULIS[ii+1], PAULIS[ii+1])
        amat += pp*params[ii].type(C_DTYPE)

    return amat

def dp2_matrix(params):
    """Compute unitary matrix for SU4 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    amat = (1-params.sum())*torch.kron(PAULIS[0], PAULIS[0])
    amat = torch.kron(amat, torch.kron(PAULIS[0], PAULIS[0]))
    for ii in range(1, 16):
        idxs = np.unravel_index(ii, (4, 4))
        pp = torch.kron(PAULIS[idxs[0]], PAULIS[idxs[1]]).unsqueeze(0)
        pp = torch.tensordot(pp, pp, ([0], [0])).reshape([2]*8).permute(0, 6, 1, 7, 2, 4, 3, 5).reshape(16, 16)
        amat += pp*params[ii-1].type(C_DTYPE)
    return amat



_dp_mat_dict = {
    "dp1": dp1_matrix,
    "dp2": dp2_matrix,
}


def dp1(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the u2 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "dp1"
    mat = _dp_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        paramnum=3,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


def dp2(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the su4 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "dp2"
    mat = _dp_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        paramnum=15,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )
