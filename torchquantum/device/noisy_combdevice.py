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

from string import ascii_lowercase
import torch
import torch.nn as nn
import numpy as np

from torchquantum.macro import C_DTYPE, F_DTYPE
from torchquantum.functional import func_name_dict, func_name_dict_collect
from .combdevice import svd_decomposition

from typing import Union

__all__ = ["NoisyCombTNDevice"]


class NoisyCombTNDevice(nn.Module):
    """
    Notice that we always treat the device as a comb TTN
    with local dimension 4 instead of 2. the real nature as
    density matrix is just used in the computations.
    """
    def __init__(
        self,
        n_wires: int,
        n_dims: int,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
        dtype : str = "complex"
    ):
        """A quantum device that contains the quantum density matrix.
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
        self.n_wires = n_wires*n_dims
        self.n_wires_per_dim = n_wires
        self.n_dims = n_dims
        self.device_name = "looplesstn"
        self.device = device
        self.iso_center = 0
        self.dtype = C_DTYPE if dtype == "complex" else F_DTYPE

        self.states = []
        for ii in range(self.n_wires):
            _state = torch.zeros(4, dtype=self.dtype)
            _state[0] = 1 + 0j  # type: ignore
            new_shape = [1, 4, 1, 1] if ii%n_wires == 0 else [1,4,1]
            _state = torch.reshape(_state, new_shape).to(self.device)

            self.register_buffer(f"state_{ii}", _state)
            self.states.append(_state)

        self.record_op = record_op
        self.op_history = []
        self.iso_towards(n_wires-1)

    def __getitem__(self, idx):
        for tidx, tt in enumerate(self.states):
            if torch.isnan(tt).any():
                raise RuntimeError(tidx)
        return self.states[idx]

    def __setitem__(self, idx, val):
        self.states[idx] = val

    def reset_op_history(self):
        """Resets the all Operation of the quantum device"""
        self.op_history = []

    def get_states_1d(self):
        """Return the states in a 1d tensor."""
        tens = torch.squeeze(self[0], dim=(0, 3))
        for ii in range(1, self.n_wires):
            tens = torch.tensordot(
                tens, self[ii], ([-1], [0])
            )
        tens = tens.reshape([2]*(2*self.n_wires))
        order = np.arange(2*self.n_wires).reshape(-1, 2).reshape(-1, order="F")
        tens = tens.permute(*order).reshape(2**self.n_wires, -1)
        return tens

    def evaluate(self, point):
        """Return the states in a 1d tensor."""
        zerot = torch.zeros(4, dtype=self[0].dtype)
        zerot[0] = 1
        onet = torch.zeros(4, dtype=self[0].dtype)
        onet[-1] = 1
        tens_lists = [[] for _ in range(self.n_dims)]
        dim = -1
        for idx, pp in enumerate(point):
            if idx%self.n_wires_per_dim == 0:
                dim += 1
            tt = self[idx]
            tt = torch.tensordot(
                tt,
                onet if pp else zerot,
                ([1], [0])
            )
            tens_lists[dim] += [tt]

        zero_dims = []
        for dim in range(self.n_dims):
            tens = tens_lists[dim][0]
            for tt in tens_lists[dim][1:]:
                tens = torch.tensordot(
                    tens, tt, ([1], [0])
                ).permute(0, 2, 1)
            zero_dims.append(tens)
        tens = zero_dims[0]
        for dim in range(1, self.n_dims):
            tens = torch.tensordot(
                tens,
                zero_dims[dim],
                ([-1], [0])
            )

        return torch.squeeze(tens)

    def overlap(self, other):
        """
        Other should be in the pure state-density matrix form,
        i.e. local dimensions 4
        """
        dim_tens = []
        for dd in range(self.n_dims):
            for ii in range(self.n_wires_per_dim-1, -1, -1):
                idx = ii + self.n_wires_per_dim*dd
                if ii == self.n_wires_per_dim-1:
                    tens = torch.tensordot(
                        self[idx],
                        other[idx].conj(),
                        ([1, 2], [1, 2])
                    )
                elif ii > 0:
                    tens = torch.tensordot(
                        self[idx],
                        tens,
                        ([2], [0])
                    )
                    tens = torch.tensordot(
                        tens,
                        other[idx].conj(),
                        ([1, 2], [1, 2])
                    )
                else: # Case ii==0
                    tens = torch.tensordot(
                        self[idx],
                        tens,
                        ([2], [0])
                    )
                    tens = torch.tensordot(
                        tens,
                        other[idx].conj(),
                        ([1, 3], [1, 2])
                    ).permute(0, 2, 1, 3)
                    dim_tens.append(tens)

        tens = dim_tens[0].reshape(dim_tens[0].shape[2:])
        for tt in dim_tens[1:]:
            tens = torch.tensordot(
                tens, tt, ([0, 1], [0, 1])
            )

        return tens

    def norm(self):
        states = [ss.conj() for ss in self.states]
        return self.overlap(states)

    def get_state_1d(self):
        """Return the state in a 1d tensor."""
        return torch.reshape(self.state, [2**self.n_wires])

    @property
    def name(self):
        """Return the name of the device."""
        return self.__class__.__name__

    def __repr__(self):
        return f" class: {self.name} \n device name: {self.device_name} \n number of qubits: {self.n_wires} \n batch size: {self.bsz} \n current computing device: {self.state.device} \n recording op history: {self.record_op} \n current states: {repr(self.get_states_1d().cpu().detach().numpy())}"

    def apply_one_site_operator(self, wires, matrix):
        state = self[wires]
        matrix = matrix.to(dtype=self.dtype)
        # Promote to the U^\dagger rho U
        if matrix.shape[0] == 2:
            matp = matrix.T.conj().contiguous()
            matrix = torch.kron(matp, matrix)
        state = torch.tensordot(
            state,
            matrix,
            ([1], [1])
        )
        if state.ndim == 3:
            state = torch.permute(state, [0, 2, 1])
        elif state.ndim == 4:
            state = torch.permute(state, [0, 3, 1, 2])
        self[wires] = state

    def apply_two_sites_operator(self, idx, jdx, matrix, dirc="R"):
        if matrix.shape[0] == 4:
            matrix = matrix.unsqueeze(0)
            matp = matrix.conj().permute(0, 2, 1).contiguous()
            matrix = torch.tensordot(matp, matrix, ([0], [0])).reshape([2]*8).permute(0, 6, 1, 7, 2, 4, 3, 5).reshape(16, 16)

        matrix = matrix.reshape(4, 4, 4, 4).to(dtype=self.dtype)
        if idx < jdx:
            matrix = torch.permute(matrix, [1, 0, 3, 2])

        minid = min(idx, jdx)
        maxid = max(idx, jdx)
        to_iso = minid if np.abs(self.iso_center-minid) < np.abs(self.iso_center-maxid) else maxid
        self.iso_towards(to_iso)
        mint = self[minid]
        maxt = self[maxid]

        if idx % self.n_wires_per_dim == 0 and jdx % self.n_wires_per_dim == 0:
            two_tens = torch.tensordot(
                mint, maxt, ([3], [0])
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 3], [2, 3])
            ).permute(0, 4, 1, 2, 5, 3).reshape(-1, np.prod(maxt.shape[1:]) )
            uu, ss, vv, _ = svd_decomposition(
                two_tens, tol=0, max_rank=64
            )
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            #uu, rr = torch.linalg.qr(two_tens)
            chi = uu.shape[1]
            mint = uu.reshape( *mint.shape[:3], chi )
            maxt = rr.reshape((chi, *maxt.shape[1:]))
        elif minid % self.n_wires_per_dim == 0:
            two_tens = torch.tensordot(
                mint, maxt, ([2], [0])
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 3], [2, 3])
            ).permute(0, 3, 1, 4, 2).reshape(-1, np.prod(maxt.shape[1:]) )
            uu, ss, vv, _ = svd_decomposition(
                two_tens, tol=0, max_rank=64
            )
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            #uu, rr = torch.linalg.qr(two_tens)
            chi = uu.shape[1]
            mint = uu.reshape( *mint.shape[:2], mint.shape[3], chi )
            mint = torch.permute(mint, (0, 1, 3, 2))
            maxt = rr.reshape((chi, *maxt.shape[1:]))
        else:
            two_tens = torch.tensordot(
                mint, maxt, ([2], [0])
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 2], [2, 3])
            ).permute(0, 2, 3, 1).reshape(np.prod(mint.shape[:2]), -1)
            uu, ss, vv, _ = svd_decomposition(
                two_tens, tol=1e-12, max_rank=64
            )
            if dirc == "R":
                rr = torch.matmul(torch.diag(ss), vv)
            else:
                uu = torch.matmul(uu, torch.diag(ss))
                rr = vv
            #uu, rr = torch.linalg.qr(two_tens)
            chi = uu.shape[1]
            mint = uu.reshape((*mint.shape[:2], chi))
            maxt = rr.reshape((chi, *maxt.shape[1:]))

        self[minid] = mint
        self[maxid] = maxt
        if dirc == "R":
            self.iso_center = maxid
        else:
            self.iso_center = minid

    def iso_towards(self, jdx):
        nwd = self.n_wires_per_dim
        if jdx%nwd == 0 and self.iso_center%nwd == 0:
            # Case of just moving between physical dimensions
            step = 1 if jdx > self.iso_center else -1
            idxs = [nwd*ii for ii in range(self.iso_center//nwd, jdx//nwd+step, step)]
        elif jdx//nwd == self.iso_center//nwd:
            # Case of moving between the same physical dimension
            step = 1 if jdx > self.iso_center else -1
            idxs = [ii for ii in range(self.iso_center, jdx+step, step)]
        else:
            # First go to the zeroth of your dimension
            idxs = [ii for ii in range(self.iso_center, -1, -1)]
            new_iso = idxs[-1]
            # Then go to the zeroth of the new dimension
            step = 1 if jdx > new_iso else -1
            idxs += [nwd*ii for ii in range(new_iso//nwd, jdx//nwd+step, step)]
            new_iso = idxs[-1]
            # Finally go the desired index
            idxs += [ii for ii in range(new_iso, jdx+1)]

        for idx in idxs:
            self.move_iso_one_step(idx)

    def move_iso_one_step(self, jdx):
        idx = self.iso_center
        if jdx == idx:
            return

        it = self[idx]
        jt = self[jdx]

        if idx % self.n_wires_per_dim == 0 and jdx % self.n_wires_per_dim != 0:
            tt = it.permute(0, 1, 3, 2)
            qq, rr = torch.linalg.qr(tt.reshape(-1, tt.shape[-1]))
            self[idx] = qq.reshape(*tt.shape[:-1], -1).permute(0, 1, 3, 2)
            self[jdx] = torch.tensordot(
                    rr, jt, ([1], [0])
                )
            #print(idx, jdx, self[idx].shape, it.shape)
        elif jdx % self.n_wires_per_dim == 0:
            qq, rr = torch.linalg.qr(it.reshape(it.shape[0], -1).T )
            self[idx] = qq.T.reshape(-1, *it.shape[1:] )
            self[jdx] = torch.tensordot(
                jt, rr.T, ([2], [1])
            ).permute(0, 1, 3, 2)
        else:
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
                    jt, rr.T, ([-1], [1])
                )
        self.iso_center = jdx


for func_name, func in func_name_dict.items():
    setattr(NoisyCombTNDevice, func_name, func)

for func_name, func in func_name_dict_collect.items():
    setattr(NoisyCombTNDevice, func_name, func)

