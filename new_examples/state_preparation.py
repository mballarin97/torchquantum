import torch
import numpy as np
import torch.optim as opt
import pytket as tk
from tqdm import tqdm
from torchquantum.device.mps_device import C_DTYPE, F_DTYPE, MPSDevice
import matplotlib.pyplot as plt
from utils import tket_to_qtorch_model

class StatePreparation:
    """
    Perform the interpolation of a quantum circuit.

    Parameters
    ----------
    func : callable
        The function to be approximated
    num_qubits : int
        The number of qubits in the circuit
    num_dims : int
        The number of dimensions in the circuit
    target : LooplessTN | np.ndarray
        The target loopless tensor network
    qdevice : str, optional
        The quantum device to be used, by default "sv".
        Available options are "sv" for statevector, "comb" for comb tensor network,
        and "noisy_comb" for noisy comb tensor network.
    device : str, optional
        The device to be used, by default "cpu"
    dtype : str, optional
        The data type to be used, by default "real"
    obd : int, optional
        The bond dimension for the open system link
    mbd : int, optional
        The maximum bond dimension of the TN
    """

    def __init__(
        self,
        num_qubits,
        target,
        ansatz,
        device="cpu",
        dtype="complex",
        mbd=None,
    ):
        self.num_qubits = num_qubits

        # True value of the function everywhere
        self.target = [
            tt.to(F_DTYPE if dtype == "real" else C_DTYPE) for tt in target
        ]

        # Quantum circuit ansatz
        self.ansatz = ansatz
        self.old_params = None
        self.state = None
        self.dtype = dtype
        self.device = device
        self.mbd = 64 if mbd is None else mbd

    def _initialize_qdevice(self):
        """
        Initialize the torchquantum device.
        """
        qdev = MPSDevice(
            self.num_qubits,
            dtype=self.dtype,
            device=self.device,
            cmbd=self.mbd,
        )

        return qdev

    def train(self, model, optimizer):
        """
        Perform a training step for the target unitary
        with the ansatz circuit and a given optimizer

        Parameters
        ----------
        model : torchquantum.Model
            The quantum circuit model to be optimized
        optimizer : torch.optimizer
            The optimizer to use in the optimization

        Returns
        -------
        float
            The value of the cost function
        float
            The average magnitude of the gradients over
            all parameters
        """

        def closure():
            """Closure function, needed for LBFGS"""
            qdev = self._initialize_qdevice()
            model.forward(qdev)

            # compute the infidelity as cost function
            loss = 2*(1-qdev.overlap(self.target).real)

            # Compute the gradiens
            optimizer.zero_grad()
            loss.backward()
            return loss

        loss = closure()

        # Update params
        optimizer.step(closure)

        return loss.item()

    def optimize(self, n_epochs, optim_params=None):
        """
        Decompose a target unitary in a
        Hardware efficient decomposition

        Parameters
        ----------
        n_epochs : int
            The number of epochs in the training
        optim_params : dict, optional
            Additional arguments for the optimizer.
            Default to None.
        """
        if optim_params is None:
            optim_params = {}
        output = optim_params.get("output", "tqdm+print")
        torch_model = tket_to_qtorch_model( self.ansatz )
        if optim_params.get("optmizer", "adam") == "adam":
            optimizer = opt.Adam(
                torch_model.parameters(), lr=optim_params.get("lr", 1e-3)
            )
        elif optim_params["optmizer"] == "lbfgs":
            optimizer = opt.LBFGS(
                torch_model.parameters(), lr=optim_params.get("lr", 1)
            )
        else:
            optimizer = self.optim_params["optmizer"]
            raise ValueError(
                f"Optimizer {optimizer} not available. Choose between adam and lbfgs"
            )

        losses = []
        tqdmf = tqdm if "tqdm" in output else lambda x: x
        for _ in tqdmf(range(n_epochs)):
            loss = self.train(
                torch_model,
                optimizer,
            )
            # Save results of the epoch
            losses.append(loss)
        qdev = self._initialize_qdevice()
        torch_model.forward(qdev)
        print(qdev.get_states_1d())

        # Update the parameters of the ansatz
        self._update_ansatz_params(optimizer)
        if "print" in output:
            print(
                f"Optimization on {self.num_qubits} sites has L2 norm error {losses[-1]}"
            )

        return np.array(losses)

    def _update_ansatz_params(self, optimizer):
        """
        Update the ansatz parameters with the optimized ones

        Parameters
        ----------
        optimizer : torch.optimizer
            The optimizer used
        """
        new_circ = tk.Circuit(self.ansatz.n_qubits)

        params = []
        for group in optimizer.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is not None:
                    params.append(np.squeeze(p.detach().numpy()))
        params = np.array(params)

        cnt = 0
        for cmd in self.ansatz.get_commands():
            op = cmd.op.type
            sites = [ qq.index[0] for qq in  cmd.qubits]
            # Defined in radiants in tket
            this_params = params[cnt]/np.pi
            new_circ.add_gate(op, this_params, sites)
            cnt += 1

        self.ansatz = new_circ

def to_mps(vect, num_sites, local_dim = 2):
    tensors = []
    state_tensor = vect.reshape([1] + [local_dim] * num_sites + [1])
    for ii in range(num_sites - 1):
        legs = state_tensor.shape
        mat = state_tensor.reshape(np.prod(legs[:2]), np.prod(legs[2:]))
        uu, ss, vv = torch.linalg.svd( mat, full_matrices=False )
        ss = ss.to(vv.dtype)
        tensors.append( uu.reshape(legs[0], local_dim, len(ss)) )
        state_tensor = (torch.diag(ss) @ vv).reshape(len(ss), 2, -1)
    tensors.append(state_tensor.reshape(len(ss), 2, 1))
    return tensors


if __name__ == "__main__":

    num_qubs = 4
    num_epochs = 2000
    tket = False

    ansatz = tk.Circuit(num_qubs)
    for ii in range(num_qubs):
        #ansatz.TK1(*np.random.rand(3), ii)
        ansatz.Rx(np.random.rand(), ii)
        ansatz.Ry(np.random.rand(), ii)
        ansatz.Rz(np.random.rand(), ii)
    for ii in range(num_qubs-1):
        #ansatz.TK2(*np.random.rand(3), ii, ii+1)
        ansatz.XXPhase(np.random.rand(), ii, ii+1)
        ansatz.YYPhase(np.random.rand(), ii, ii+1)
        ansatz.ZZPhase(np.random.rand(), ii, ii+1)
    for ii in range(num_qubs):
        #ansatz.TK1(*np.random.rand(3), ii)
        ansatz.Rx(np.random.rand(), ii)
        ansatz.Ry(np.random.rand(), ii)
        ansatz.Rz(np.random.rand(), ii)

    # A sine as target state
    target = torch.linspace(0, 1, 2**num_qubs, dtype=C_DTYPE)
    target = torch.sin(target)
    target /= torch.sqrt( torch.vdot(target, target) )
    target_mps = to_mps(target, num_qubs)

    optimizer = StatePreparation(
        num_qubs,
        target_mps,
        ansatz
    )
    losses = optimizer.optimize(num_epochs)

    res = optimizer.ansatz.get_statevector()
    fidelity = 1-np.abs(np.vdot( res, target.numpy() ))**2

    plt.plot(losses, ls="dashed", label="loss")
    plt.plot(len(losses)-1, fidelity, "o", label="Final fidelity")
    plt.legend()
    plt.yscale("log")
    plt.show()
