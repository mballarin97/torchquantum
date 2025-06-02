from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class TK1(Operation, metaclass=ABCMeta):
    """Class for TK1 gate."""

    num_params = 3
    num_wires = 1
    op_name = "tk1"
    func = staticmethod(tqf.tk1)

    @classmethod
    def _matrix(cls, params):
        return tqf.tk1_matrix(params)


class TK2(Operation, metaclass=ABCMeta):
    """Class for controlled TK@ gate."""

    num_params = 3
    num_wires = 2
    op_name = "tk2"
    func = staticmethod(tqf.tk2)

    @classmethod
    def _matrix(cls, params):
        return tqf.tk2_matrix(params)
