import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm
from .rx import rx_matrix, rxx_matrix
from .ry import ry_matrix, ryy_matrix
from .rz import rz_matrix, rzz_matrix



if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None

def tk1_matrix(params):
    """Compute unitary matrix for TK1 gate.
    (https://docs.quantinuum.com/tket/api-docs/optype.html#pytket.circuit.OpType)

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    amat = torch.eye(2, dtype=C_DTYPE).unsqueeze(0).repeat(params.shape[0], 1, 1)
    for ii in range(params.shape[0]):
        amat[ii] = rz_matrix(params[:, 0]) @ rx_matrix(params[:, 1]) @ ry_matrix(params[:,2])

    return amat.squeeze(0)

def tk2_matrix(params):
    """Compute unitary matrix for TK2 gate.
    (https://docs.quantinuum.com/tket/api-docs/optype.html#pytket.circuit.OpType)

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    amat = torch.eye(4, dtype=C_DTYPE).unsqueeze(0).repeat(params.shape[0], 1, 1)
    for ii in range(params.shape[0]):
        amat[ii] = rxx_matrix(params[:, :1]) @ ryy_matrix(params[:, 1:2]) @ rzz_matrix(params[:,2:3])

    return amat.squeeze(0)



_tk_mat_dict = {
    "tk1": tk1_matrix,
    "tk2": tk2_matrix,
}


def tk1(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the tk1 gate.

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
    name = "tk1"
    mat = _tk_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        paramnum=2,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


def tk2(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the tk2 gate.

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
    name = "tk2"
    mat = _tk_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        paramnum=2,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )
