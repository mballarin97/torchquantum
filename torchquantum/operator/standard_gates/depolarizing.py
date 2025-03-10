from ..op_types import Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
import torchquantum.functional as tqf


class DP1(Operation, metaclass=ABCMeta):
    """Class for single-qubit depolarizing channel."""

    num_params = 1
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    op_name = "dp1"
    func = staticmethod(tqf.dp1)

    @classmethod
    def _matrix(cls, params):
        return tqf.dp1_matrix(params)

class DP2(Operation, metaclass=ABCMeta):
    """Class for CSX Gate."""

    num_params = 1
    num_wires = 2
    op_name = "csx"
    func = staticmethod(tqf.dp2)

    @classmethod
    def _matrix(cls, params):
        return tqf.dp2_matrix(params)
