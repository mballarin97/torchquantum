import numpy as np
import torch
import pytket as tk
from qiskit import QuantumCircuit
import torchquantum as tq


def tket_to_qtorch_model(circ : tk.Circuit, noisy=False):
    """
    Map the decomposition to a torchquantum model
    to perform an optimization of the parameters
    through automatic differentiation.
    The gates allowed at the moment are:
    ``("rx", "rz", "rzz")``

    Returns
    -------
    QuantumModel
        torchquantum model
    """
    okgates = (
            "h",
            "x",
            "z",
            "rx",
            "rz",
            "ry",
            "rzz",
            "cry",
            "cx",
            "tk1",
            "tk2",
            "xxphase",
            "yyphase",
            "zzphase",
        )
    ops = []
    for cmd in circ.get_commands():
        op = cmd.op.type.name.lower()
        sites = [ qq.index[0] for qq in  cmd.qubits]
        # Defined in radiants in tket
        params = [pp*np.pi for pp in cmd.op.params]

        if op not in okgates:
            raise ValueError(
                f"Only Decompositions of {okgates} can be mapped to torch model, not {op}"
            )

        if op == "rzz" and noisy:
            op = "nrzz"
        if "phase" in op:
            op = "r"+op[:2]

        if len(params) == 0:
            params = None
        else:
            params = torch.from_numpy(np.array(params))

        ops += [
            {
                "name": op,
                "wires": sites,
                "params": params,
                "trainable": params is not None,
            }
        ]
    module = tq.QuantumModule.from_op_history(ops)
    module.n_wires = circ.n_qubits

    return module

def qiskit_to_qtorch_model(circ : QuantumCircuit, noisy=False):
    okgates = (
            "h",
            "x",
            "z",
            "rx",
            "rz",
            "ry",
            "rzz",
            "cry",
            "su2",
            "su4",
            "cx",
            "id",
            "dp1",
            "dp2",
        )
    ops = []
    for instr in circ.data:
        sites = [circ.find_bit(qq)[0] for qq in instr.qubits]
        name = instr.name
        params = instr.operation.params

        if op not in okgates:
            raise ValueError(
                f"Only Decompositions of {okgates} can be mapped to torch model, not {op}"
            )

        if op == "rzz" and noisy:
            op = "nrzz"

        if len(params) == 0:
            params = None
        else:
            params = torch.from_numpy(np.array(params))

        ops += [
                {
                    "name": name,
                    "wires": sites,
                    "params": params,
                    "trainable": params is not None,
                }
            ]
    module = tq.QuantumModule.from_op_history(ops)
    module.n_wires = circ.n_qubits

    return module