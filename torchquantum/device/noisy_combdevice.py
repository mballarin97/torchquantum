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
from torchquantum.linalg import truncated_svd_gesdd as svd_decomposition

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
        dtype : str = "complex",
        ctol : float = 1e-9,
        otol : float = 1e-12,
        cmbd : int = 64,
        ombd : int = 16,
        noise_model : dict = None,
    ):
        """A quantum device that contains the quantum density matrix.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
            dtype : str
                Type of the TN
            ctol : float
                Tolerance for SVD of the closed system
            otol : float
                Tolerance for the SVD of the open system
            cmbd : int
                Maximum bond dimension for the SVD of the closed system
            ombd : int
                Maximum bond dimension for the SVD of the closed system
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires*n_dims
        self.n_wires_per_dim = n_wires
        self.n_dims = n_dims
        self.device_name = "noisy_comb_tn"
        self.device = device
        self.iso_center = 0
        self.dtype = C_DTYPE if dtype == "complex" else F_DTYPE
        self.osys_dim = [1]*self.n_wires

        # Truncation params
        self.ctol = ctol
        self.otol = otol
        self.cmbd = cmbd
        self.ombd = ombd

        self.states = []
        for ii in range(self.n_wires):
            _state = torch.zeros(2, dtype=self.dtype)
            _state[0] = 1  # type: ignore
            new_shape = [1, 2, 1, 1] if ii%n_wires == 0 else [1, 2,1]
            _state = torch.reshape(_state, new_shape).to(self.device)

            self.register_buffer(f"state_{ii}", _state)
            self.states.append(_state)

        self.record_op = record_op
        self.op_history = []
        self.iso_towards(n_wires-1)

        # Initialize noise model
        if noise_model is None:
            noise_model = {}
        self.err_slope = noise_model.get("err_slope", 1.43e-3)
        self.err_intercept = noise_model.get("err_intercept", 2.1e-4)
        self.use_constant = noise_model.get("use_constant", False)
        self.zero_dim_noisy = noise_model.get("zero_dim_noisy", False)

    def __getitem__(self, idx):
        for tidx, tt in enumerate(self.states):
            if torch.isnan(tt).any():
                raise RuntimeError(f"Nans in the {tidx} tensor")
        return self.states[idx]

    def __setitem__(self, idx, val):
        self.states[idx] = val

    def reset_op_history(self):
        """Resets the all Operation of the quantum device"""
        self.op_history = []

    def _zz_noise(self, errangle):
        """
        Get the probability of error for a ZZ gate given
        the angle.
        """
        if errangle > 1e-4:
            p_err = self.err_slope * errangle
            if self.use_constant:
                p_err += self.err_intercept
            p_err = 1-torch.sqrt(1-5/4*p_err)
        else:
            p_err = torch.zeros_like(errangle)

        return p_err

    def get_states_1d(self):
        """Return the states in a 1d tensor."""
        tens = torch.squeeze(self[0], dim=(0, 3))
        tens = tens.reshape(2, self.osys_dim[0], -1)
        tens = torch.tensordot(tens, tens.conj(), ([1], [1])).permute(0, 2, 1, 3).reshape(4, -1)
        for ii in range(1, self.n_wires):
            tmp = self[ii].reshape(self[ii].shape[0], 2, self.osys_dim[ii], -1)
            tmp = torch.tensordot(tmp, tmp.conj(), ([2], [2])).permute(0, 3, 1, 4, 2, 5).reshape(tmp.shape[0]**2, 4, -1)

            tens = torch.tensordot(
                tens, tmp, ([-1], [0])
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
            tt = tt.reshape(tt.shape[0], 2, self.osys_dim[idx], *tt.shape[2:])
            tt = torch.tensordot(
                tt, tt.conj(), ([2], [2])
            )
            if idx%self.n_wires_per_dim == 0:
                tt = tt.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
                    self[idx].shape[0]**2,
                    4,
                    self[idx].shape[2]**2,
                    self[idx].shape[3]**2,
                )
            else:
                tt = tt.permute(0, 3, 1, 4, 2, 5).reshape(
                    self[idx].shape[0]**2,
                    4,
                    self[idx].shape[2]**2,
                )
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
        Other should be in the pure state form,
        i.e. local dimensions 2
        """
        dim_tens = []
        for dd in range(self.n_dims):
            for ii in range(self.n_wires_per_dim-1, -1, -1):
                idx = ii + self.n_wires_per_dim*dd
                if ii == self.n_wires_per_dim-1:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    tens = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1, 3], [1, 2])
                    )
                    # Tensor (left_o, o_leg, left_c)
                    tens = tens.permute(0, 2, 1).reshape(-1, self.osys_dim[idx])
                    tens = torch.tensordot(
                        tens,
                        tens.conj(),
                        ([1], [1])
                    )
                elif ii > 0:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    st = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1], [1])
                    ).permute(0, 3, 1, 2, 4).reshape(
                        self[idx].shape[0]*other[idx].shape[0],
                        self.osys_dim[idx],
                        self[idx].shape[2]*other[idx].shape[2],
                    )
                    stc = st.conj()
                    tens = torch.tensordot(
                        st,
                        tens,
                        ([2], [0])
                    )
                    tens = torch.tensordot(
                        tens,
                        stc,
                        ([1, 2], [1, 2])
                    )
                else: # Case ii==0
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], *st.shape[2:])
                    st = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1], [1])
                    ).permute(0, 4, 1, 2, 5, 3, 6).reshape(
                        self[idx].shape[0]*other[idx].shape[0],
                        self.osys_dim[idx],
                        self[idx].shape[2]*other[idx].shape[2],
                        self[idx].shape[3]*other[idx].shape[3],
                    )
                    stc = st.conj()
                    tens = torch.tensordot(
                        st,
                        tens,
                        ([2], [0])
                    ).permute(0, 1, 3, 2)
                    tens = torch.tensordot(
                        tens,
                        stc,
                        ([1, 2], [1, 2])
                    ).permute(0, 2, 1, 3)

                    dim_tens.append(tens)

        tens = dim_tens[0].reshape(dim_tens[0].shape[2:])
        for tt in dim_tens[1:]:
            tens = torch.tensordot(
                tens, tt, ([0, 1], [0, 1])
            )

        return tens

    def overlap_(self, other):
        """
        Other should be in the pure state form,
        i.e. local dimensions 2
        """
        new = [None for _ in range(self.n_wires)]
        for dd in range(self.n_dims):
            for ii in range(self.n_wires_per_dim-1, -1, -1):
                idx = ii + self.n_wires_per_dim*dd
                if ii == self.n_wires_per_dim-1:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    tens = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1], [1])
                    )
                    # Tensor (left_o, o_leg, left_c)
                    tens = tens.permute(0, 3, 1, 2, 4).reshape(-1, self.osys_dim[idx])
                    uu, ss, vv = svd_decomposition(tens, self.cmbd, rel_tol=self.ctol)
                    vv = vv.T.conj()
                    ss = ss.to(vv.dtype)
                    new[idx] = vv.reshape(len(ss), self.osys_dim[idx], 1)
                    tens = torch.matmul(uu, torch.diag(ss))
                elif ii > 0:
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], st.shape[2])
                    st = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1], [1])
                    ).permute(0, 3, 1, 2, 4).reshape(
                        self[idx].shape[0]*other[idx].shape[0],
                        self.osys_dim[idx], self[idx].shape[2]*other[idx].shape[2],
                    )
                    st = torch.tensordot(st, tens, ([-1], [0])).reshape(self[idx].shape[0]*other[idx].shape[0], -1)
                    uu, ss, vv = svd_decomposition(st, self.cmbd, rel_tol=self.ctol)
                    vv = vv.T.conj()
                    ss = ss.to(vv.dtype)
                    new[idx] = vv.reshape(len(ss), self.osys_dim[idx], -1)
                    tens = torch.matmul(uu, torch.diag(ss)).reshape(-1, len(ss))
                else: # Case ii==0
                    st = self[idx]
                    st = st.reshape(st.shape[0], 2, self.osys_dim[idx], *st.shape[2:])
                    st = torch.tensordot(
                        st,
                        other[idx].conj(),
                        ([1], [1])
                    ).permute(0, 4, 1, 2, 5, 3, 6).reshape(
                        self[idx].shape[0]*other[idx].shape[0],
                        self.osys_dim[idx],
                        self[idx].shape[2]*other[idx].shape[2],
                        self[idx].shape[3]*other[idx].shape[3],
                    )
                    st = torch.tensordot(st, tens, ([-2], [0]))
                    st = st.permute(0, 1, 3, 2)
                    new[idx] = st

        dim_tens = []
        for dd in range(self.n_dims):
            for ii in range(self.n_wires_per_dim-1, -1, -1):
                idx = ii + self.n_wires_per_dim*dd
                if ii == self.n_wires_per_dim-1:
                    tens = torch.tensordot(
                        new[idx],
                        new[idx].conj(),
                        ([1, 2], [1, 2])
                    )
                elif ii > 0:
                    tens = torch.tensordot(
                        new[idx],
                        tens,
                        ([2], [0])
                    )
                    tens = torch.tensordot(
                        tens,
                        new[idx].conj(),
                        ([1, 2], [1, 2])
                    )
                else: # Case ii==0
                    tens = torch.tensordot(
                        new[idx],
                        tens,
                        ([2], [0])
                    )
                    tens = torch.tensordot(
                        tens,
                        new[idx].conj(),
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
        sshape = state.shape
        matrix = matrix.to(dtype=self.dtype)
        state = state.reshape(sshape[0], 2, self.osys_dim[wires], *sshape[2:])

        state = torch.tensordot(
            state,
            matrix,
            ([1], [1])
        )
        if len(sshape) == 3:
            state = torch.permute(state, [0, 3, 1, 2])
        elif len(sshape) == 4:
            state = torch.permute(state, [0, 4, 1, 2, 3])
        self[wires] = state.reshape(sshape[0], 2*self.osys_dim[wires], *sshape[2:])

    def apply_one_site_noise(self, wires, p_err):
        if torch.isclose(p_err, torch.zeros_like(p_err)):
            return

        # Having the exact same probability is an issue for the SVD,
        # leading to singular values with multiplicity >1. Thus, we slightly
        # modify the probabilities to ensure the singular values are different.
        if self.dtype == C_DTYPE:
            dtype = C_DTYPE
            tensor = torch.zeros((2, 2, 4), dtype=dtype)
            dp = p_err/4
        else:
            dtype = F_DTYPE
            tensor = torch.zeros((2, 2, 3), dtype=dtype)
            dp = p_err/2

        tensor[:, :, 1] = torch.sqrt(dp)*torch.tensor([[0, 1.0], [1, 0]], dtype=dtype)
        tensor[:, :, -1] = torch.sqrt(dp)*torch.tensor([[1.0, 0], [0, -1]], dtype=dtype)
        if self.dtype == C_DTYPE:
            tensor[:, :, 2] = torch.sqrt(dp)*torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)

        tensor[:, :, 0] = torch.sqrt(1-3*dp)*torch.tensor([[1, 0], [0, 1]], dtype=dtype)

        self.iso_towards(wires)
        state = self[wires]
        sshape = state.shape
        if len(sshape) == 4 and not self.zero_dim_noisy:
            return
        state = state.reshape(sshape[0], 2, self.osys_dim[wires], *sshape[2:])

        state = torch.tensordot(
            state,
            tensor.to(state.dtype),
            ([1], [1])
        )
        if len(sshape) == 3:
            state = torch.permute(state, [0, 3, 2, 1, 4])
        elif len(sshape) == 4:
            state = torch.permute(state, [0, 4, 2, 3, 1, 5])

        state = state.reshape(-1, self.osys_dim[wires]*4 )
        uu, ss, _ = svd_decomposition(state, self.ombd, rel_tol=self.otol)
        ss = ss.to(uu.dtype)
        state = torch.matmul(uu, torch.diag(ss))
        odim = len(ss)

        self.osys_dim[wires] = odim
        state = state.reshape(sshape[0], 2, *sshape[2:], odim)

        if len(sshape) == 3:
            state = torch.permute(state, [0, 1, 3, 2])
        elif len(sshape) == 4:
            state = torch.permute(state, [0, 1, 4, 2, 3])

        self[wires] = state.reshape(sshape[0], 2*self.osys_dim[wires], *sshape[2:])

    def apply_two_sites_operator(self, idx, jdx, matrix, dirc="R"):
        apply_noise = False
        if isinstance(matrix, list):
            apply_noise = True
            p_err = self._zz_noise(matrix[1])
            matrix = matrix[0]
        matrix = matrix.reshape(2, 2, 2, 2).to(dtype=self.dtype)
        if idx > jdx:
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
            two_tens = two_tens.reshape(
                mint.shape[0], 2, self.osys_dim[minid], mint.shape[-2], 2, self.osys_dim[maxid], maxt.shape[2], maxt.shape[3]
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 4], [2, 3])
            ).permute(0, 6, 1, 2, 3, 7, 4, 5).reshape(-1, np.prod(maxt.shape[1:]) )
            #uu, ss, vv, _ = svd_decomposition(
            #    two_tens, tol=self.ctol, max_rank=self.cmbd
            #)
            uu, ss, vv = svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
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
            two_tens = two_tens.reshape(
                mint.shape[0], 2, self.osys_dim[minid], mint.shape[-1], 2, self.osys_dim[maxid], maxt.shape[2]
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 4], [2, 3])
            ).permute(0, 5, 1, 2, 6, 3, 4).reshape(-1, np.prod(maxt.shape[1:]) )
            #uu, ss, vv, _ = svd_decomposition(
            #    two_tens, tol=self.ctol, max_rank=self.cmbd
            #)
            uu, ss, vv = svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
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
            two_tens = two_tens.reshape(
                mint.shape[0], 2, self.osys_dim[minid], 2, self.osys_dim[maxid], maxt.shape[2]
            )
            two_tens = torch.tensordot(
                two_tens, matrix,
                ([1, 3], [2, 3])
            ).permute(0, 4, 1, 5, 2, 3).reshape(np.prod(mint.shape[:2]), -1)
            #uu, ss, vv, _ = svd_decomposition(
            #    two_tens, tol=self.ctol, max_rank=self.cmbd
            #)
            uu, ss, vv = svd_decomposition(two_tens, self.cmbd, rel_tol=self.ctol)
            vv = vv.T.conj()
            ss = ss.to(vv.dtype)
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
        if apply_noise:
            self.apply_one_site_noise(idx, p_err)
            self.apply_one_site_noise(jdx, p_err)

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
            idxs = [ii for ii in range(self.iso_center, self.iso_center//nwd*nwd-1, -1)]
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
        elif jdx % self.n_wires_per_dim == 0 and idx % self.n_wires_per_dim != 0:
            if idx < jdx:
                qq, rr = torch.linalg.qr(it.reshape(-1, it.shape[-1]) )
                self[idx] = qq.reshape(*it.shape[:-1], -1 )
                self[jdx] = torch.tensordot(
                    rr, jt, ([1], [0])
                )
            else:
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

