from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class SU2(Operation, metaclass=ABCMeta):
    """Class for SU2 gate."""

    num_params = 3
    num_wires = 1
    op_name = "su2"
    func = staticmethod(tqf.su2)

    @classmethod
    def _matrix(cls, params):
        return tqf.su2_matrix(params)


class SU4(Operation, metaclass=ABCMeta):
    """Class for controlled SU4 gate."""

    num_params = 15
    num_wires = 2
    op_name = "su4"
    func = staticmethod(tqf.su4)

    @classmethod
    def _matrix(cls, params):
        return tqf.su4_matrix(params)
