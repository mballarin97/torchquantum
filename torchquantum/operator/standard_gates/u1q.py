from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class U1q(Operation, metaclass=ABCMeta):
    """Class for Quantinuum's U1q gate."""

    num_params = 2
    num_wires = 1
    op_name = "u1q"
    func = staticmethod(tqf.u1q)

    @classmethod
    def _matrix(cls, params):
        return tqf.u1q_matrix(params)
