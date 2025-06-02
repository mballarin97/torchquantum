"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import numpy as np

from torchquantum.macro import C_DTYPE, F_DTYPE
from torchquantum.functional import func_name_dict, func_name_dict_collect
from torchquantum.linalg import truncated_svd_gesdd as svd_decomposition

from typing import Union, List

__all__ = ["MPSDevice"]


class MPSDevice(nn.Module):
    def __init__(
        self,
        n_wires: int,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
        dtype : str = "complex",
        ctol : float = 1e-9,
        cmbd : int = 64,
        trace_qubits : List[int] = None,
    ):
        """A quantum device that contains the quantum state vector.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = "mps"
        self.trace_qubits = [] if trace_qubits is None else trace_qubits
        self.device = device
        self.iso_center = 0
        self.dtype = C_DTYPE if dtype == "complex" else F_DTYPE
        self.ctol = ctol
        self.cmbd = cmbd

        self.states = []
        for ii in range(self.n_wires):
            _state = torch.zeros(2, dtype=self.dtype)
            _state[0] = 1 + 0j  # type: ignore
            new_shape = [1, 2, 1]
            _state = torch.reshape(_state, new_shape).to(self.device)

            self.register_buffer(f"state_{ii}", _state)
            self.states.append(_state)

        self.record_op = record_op
        self.op_history = []
        self.iso_towards(n_wires-1)

    def __getitem__(self, idx):
        for tidx, tt in enumerate(self.states):
            if torch.isnan(tt).any():
                raise RuntimeError(f"NaN appeared in tensor {tidx}")
        return self.states[idx]

    def __setitem__(self, idx, val):
        self.states[idx] = val

    def reset_op_history(self):
        """Resets the all Operation of the quantum device"""
        self.op_history = []

    def get_states_1d(self):
        """Return the states in a 1d tensor."""
        tens = torch.squeeze(self[0], dim=0)
        for ii in range(1, self.n_wires):
            tens = torch.tensordot(
                tens, self[ii], ([-1], [0])
            )
        return torch.reshape(tens, [2**self.n_wires])

    def evaluate(self, point):
        """Return the states in a 1d tensor."""
        tens_list = []
        for idx, pp in enumerate(point):
            tt = self[idx][:, int(pp), :]
            tens_list += [tt]

        tens = tens_list[0]
        for tt in tens_list[1:]:
            tens = torch.tensordot(
                tens, tt, ([1], [0])
            )

        return torch.squeeze(tens)

    def sample(self, num_samples):
        probs = {}
        bitstrings = []

        self.iso_towards(0)
        for _ in range(num_samples):
            bitstring, prob = self.single_sample()
            probs[bitstring] = prob
            bitstrings.append(bitstring)

        uniq, count = np.unique(bitstrings, return_counts=True)
        counts = dict(zip(uniq, count))
        return probs, counts


    def single_sample(self):
        tens = self[0]
        for ii in range(1, self.n_wires):
            dm = torch.tensordot(
                tens, tens.conj(),
                ([0, 2], [0, 2])
            )
            # Measured 0
            if np.random.rand() < dm[0, 0].abs():
                meas = 0
                tmp_val += "0"
                prob *= dm[0, 0].abs()
                fact = 1/torch.sqrt(dm[0, 0].abs())
            else:
                meas = 1
                tmp_val += "1"
                prob *= dm[1, 1].abs()
                fact = 1/torch.sqrt(dm[1, 1].abs())

            tens = tens[:, meas, :]*fact
            tens = torch.tensordot(
                tens,
                self[ii],
                ([2], [0])
            )
        val += tmp_val[::-1]
        return val, prob.detach().item()

    def overlap(self, other):
        tm = torch.ones((1, 1), dtype=self.dtype)
        for ii in range(self.n_wires):
            tm = torch.tensordot(
                tm, self[ii], ([0], [0])
            )
            tm = torch.tensordot(
                tm, other[ii].conj(), ([0, 1], [0, 1])
            )
        return torch.squeeze(tm)

    def to_mpo(self):
        """
        Map to an MPO, possibly tracing away some qubits
        """
        mpo_tensors = []
        nextt =  torch.ones((1, 1))
        for ii in range(self.n_wires):
            tens = torch.tensordot(nextt, self[ii], ([1], [0]))
            if ii in self.trace_qubits:
                nextt = torch.tensordot(
                    tens, tens.conj(), ([1], [1])
                ).permute(0, 2, 1, 3).reshape(tens.shape[0]**2, -1)
            else:

                tmp = torch.unsqueeze(tens, 0)
                mat = torch.tensordot(
                    tmp, tmp.conj(), ([0], [0])
                ).permute(0, 3, 1, 4, 2, 5).reshape(tens.shape[0]**2*4, -1)

                if ii < self.n_wires-1:
                    uu, ss, vv = svd_decomposition(mat, self.cmbd, rel_tol=self.ctol)
                    ss = ss.to(vv.dtype)
                    vv = vv.T.conj()
                    mpo_tensors += [ uu.reshape(self[ii].shape[0]**2, 4, len(ss)) ]
                    nextt = torch.matmul( torch.diag(ss), vv )
        if self.n_wires-1 in self.trace_qubits:
            mpo_tensors[-1] = torch.tensordot(
                mpo_tensors[-1], nextt, ([-1], [0])
            )
        return mpo_tensors

    def trace_overlap(self, other):
        """
        Return the overlap after tracing out some degrees of freedom,
        effectively taking the trace distance with respect to an input MPO.
        """
        mpo_tensors = self.to_mpo()
        mpsdev = MPSDevice(
            self.n_wires - len(self.trace_qubits),
            self.device,
            dtype=self.dtype,
            ctol=self.ctol,
            cmbd=self.cmbd,
            trace_qubits=None
        )
        mpsdev.states = mpo_tensors
        other = [oo.reshape(oo.shape[0], 4, oo.reshape[-1]) for oo in other]

        return mpsdev.overlap(other)


    def norm(self):
        states = [ss.conj() for ss in self.states]
        return self.overlap(states)

    def get_state_1d(self):
        """Return the state in a 1d tensor."""
        return torch.reshape(self.state, [2**self.n_wires])

    def copy(self):
        copy_dev = MPSDevice(
            self.n_wires,
            self.device,
            self.record_op,
            self.dtype,
            self.ctol,
            self.cmbd,
            self.trace_qubits
            )
        for ii in range(self.n_wires):
            copy_dev[ii] = self[ii].detach().clone()

        copy_dev.iso_center = self.iso_center
        return copy_dev

    @property
    def name(self):
        """Return the name of the device."""
        return self.__class__.__name__

    def __repr__(self):
        return f" class: {self.name} \n device name: {self.device_name} \n number of qubits: {self.n_wires} \n batch size: {self.bsz} \n current computing device: {self.state.device} \n recording op history: {self.record_op} \n current states: {repr(self.get_states_1d().cpu().detach().numpy())}"

    def apply_one_site_operator(self, wires, matrix):
        state = self[wires]
        matrix = matrix.to(dtype=self.dtype)
        state = torch.tensordot(
            state,
            matrix,
            ([1], [1])
        )
        state = torch.permute(state, [0, 2, 1])
        self[wires] = state

    def _apply_two_sites_operator(self, idx, jdx, matrix, dirc="R"):
        matrix = matrix.reshape(2, 2, 2, 2).to(dtype=self.dtype)
        if idx > jdx:
            matrix = torch.permute(matrix, [1, 0, 3, 2])

        minid = min(idx, jdx)
        maxid = max(idx, jdx)
        to_iso = minid if np.abs(self.iso_center-minid) < np.abs(self.iso_center-maxid) else maxid
        self.iso_towards(to_iso)
        mint = self[minid]
        maxt = self[maxid]

        two_tens = torch.tensordot(
            mint, maxt, ([2], [0])
        )
        two_tens = torch.tensordot(
            two_tens, matrix,
            ([1, 2], [2, 3])
        ).permute(0, 2, 3, 1).reshape(np.prod(mint.shape[:2]), -1)
        uu, ss, vv = svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
        ss = ss.to(vv.dtype)
        vv = vv.T.conj()
        if dirc == "R":
            rr = torch.matmul(torch.diag(ss), vv)
        else:
            uu = torch.matmul(uu, torch.diag(ss))
            rr = vv
        chi = uu.shape[1]
        mint = uu.reshape((*mint.shape[:2], chi))
        maxt = rr.reshape((chi, *maxt.shape[1:]))

        self[minid] = mint
        self[maxid] = maxt
        if dirc == "R":
            self.iso_center = maxid
        else:
            self.iso_center = minid

    def swap(self, idx, jdx, dirc="R"):
        minid = min(idx, jdx)
        maxid = max(idx, jdx)
        to_iso = minid if np.abs(self.iso_center-minid) < np.abs(self.iso_center-maxid) else maxid
        self.iso_towards(to_iso)
        mint = self[minid]
        maxt = self[maxid]

        # The permutation here is the swap
        two_tens = torch.tensordot(
            mint, maxt, ([2], [0])
        ).permute(0, 2, 1, 3).reshape(np.prod(mint.shape[:2]), -1)
        uu, ss, vv = svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
        ss = ss.to(vv.dtype)
        vv = vv.T.conj()
        if dirc == "R":
            rr = torch.matmul(torch.diag(ss), vv)
        else:
            uu = torch.matmul(uu, torch.diag(ss))
            rr = vv
        chi = uu.shape[1]
        mint = uu.reshape((*mint.shape[:2], chi))
        maxt = rr.reshape((chi, *maxt.shape[1:]))

        self[minid] = mint
        self[maxid] = maxt
        if dirc == "R":
            self.iso_center = maxid
        else:
            self.iso_center = minid

    def apply_two_sites_operator(self, idx, jdx, matrix, dirc="R"):
        qubits = [idx, jdx]
        new_qubits = [qq for qq in qubits]

        # Apply swaps to bring qubits adjacents
        if np.abs(idx - jdx)>1:
            for q1 in range(min(qubits), max(qubits)-1):
                self.swap(q1, q1+1, dirc="R")

        if qubits[0] < qubits[1]:
            new_qubits[0] = max(qubits)-1
        else:
            new_qubits[1] = max(qubits)-1
        # Apply two site operator
        self._apply_two_sites_operator(new_qubits[0], new_qubits[1], matrix, dirc)

        # Bring back qubits in the original position
        if np.abs(qubits[0] - qubits[1])>1:
            for q1 in range(max(qubits)-1, min(qubits), -1):
                self.swap(q1-1, q1, dirc="L")

    def iso_towards(self, jdx):
        step = 1 if jdx > self.iso_center else -1
        idxs = [ii for ii in range(self.iso_center, jdx+step, step)]

        for idx in idxs:
            self.move_iso_one_step(idx)

    def move_iso_one_step(self, jdx):
        idx = self.iso_center
        if jdx == idx:
            return

        it = self[idx]
        jt = self[jdx]

        if idx < jdx:
            qq, rr = torch.linalg.qr(it.reshape(-1, it.shape[-1]))
            #qq, ss, vv, _ = svd_decomposition(it.reshape(-1, it.shape[-1]))
            #rr = torch.diag(ss) @ vv
            self[idx] = qq.reshape(*it.shape[:-1], -1)
            self[jdx] = torch.tensordot(
                rr, jt, ([1], [0])
            )
        else:
            qq, rr = torch.linalg.qr(it.reshape(it.shape[0], -1).T )
            #uu, ss, qq, _ = svd_decomposition(it.reshape(it.shape[0], -1))
            #rr = uu @ torch.diag(ss)
            self[idx] = qq.T.reshape(-1, *it.shape[1:] )
            self[jdx] = torch.tensordot(
                jt, rr.T, ([-1], [0])
            )
        self.iso_center = jdx


for func_name, func in func_name_dict.items():
    setattr(MPSDevice, func_name, func)

for func_name, func in func_name_dict_collect.items():
    setattr(MPSDevice, func_name, func)
