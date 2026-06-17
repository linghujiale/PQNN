from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Literal, Optional

import pennylane as qml
import torch
import torch.nn as nn

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


AnsatzType = Literal["alternate", "cascade", "cross_mesh", "layered"]
EncodingType = Literal["angle", "amplitude"]
DEFAULT_DTYPE = torch.float32


def get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def build_lamd_tag(value: float) -> str:
    return f"{value:.1f}"


def get_config() -> Dict[str, Any]:
    return {
        # ---------------- Basic settings ----------------
        "default_dtype": torch.float32,
        "save_dir": "figures",
        "model_filename": "poisson_dv_qnn_angle_cascade_h20_q8_l15_fp32_val100x100.pt",
        "plot_filename_prefix": "poisson_dv_qnn_angle_cascade_h20_q8_l15_fp32_val100x100_QNN",
        "rel_l2_history_filename": "poisson_dv_qnn_angle_cascade_h20_q8_l15_fp32_val100x100_QNN_rel_l2_history",
        "arch_name_serial": "DV-QNN Angle-Cascade 2D Poisson PINN (h=20, q=8, L=15, fp32, val=100x100)",
        # ---------------- Quantum architecture ----------------
        "num_qubits": 10,
        "num_quantum_layers": 9,
        "encoding": "angle",
        "q_ansatz": "cascade",
        "shots": None,
        "diff_method": "backprop",
        # ---------------- Classical architecture ----------------
        "classical_input_dim": 2,
        "classical_output_dim": 1,
        "hidden_dim": 20,
        # ---------------- PDE settings (2D Poisson) ----------------
        "domain_min": -1.0,
        "domain_max": 1.0,
        # ---------------- Sampling ----------------
        "num_interior_points": 256,
        "num_bnd_per_edge": 128,
        "validation_grid_n": 100,
        "jitter_scale": 0.02,
        "resample_every": 100,
        "sobol_scramble": True,
        # ---------------- Adam training ----------------
        "adam_lr": 5e-3,
        "adam_steps": 1000,
        "scheduler_step_size": 800,
        "scheduler_gamma": 0.7,
        # ---------------- LBFGS training ----------------
        "lbfgs_steps": 200,
        "lbfgs_lr": 0.1,
        "lbfgs_max_iter": 20,
        "lbfgs_line_search_fn": "strong_wolfe",
        "lbfgs_history_size": 100,
        "lbfgs_tolerance_grad": 1e-7,
        "lbfgs_tolerance_change": 1e-9,
        "boundary_loss_weight": 50.0,
        "grad_clip_max_norm": 0.0,
        "scheduler_factor": 0.9,
        "scheduler_patience": 1000,
        "scheduler_min_lr": 1e-6,
        "log_every": 100,
        "lbfgs_log_every": 20,
        # ---------------- Evaluation ----------------
        "eval_every": 100,
        # ---------------- Plotting ----------------
        "rel_l2_eps": 1e-8,
        "eval_batch_size": 64,
        "plot_grid_n": 60,
        "plot_figsize": (15, 4),
        "history_figsize": (7, 4),
        "plot_dpi": 150,
    }

CONFIG = get_config()
lamd = get_env_float("LAMD", 5.0)
CONFIG["adam_steps"] = get_env_int("ADAM_STEPS", CONFIG["adam_steps"])
CONFIG["lbfgs_steps"] = get_env_int(
    "LBFGS_EPOCHS",
    get_env_int("LBFGS_STEPS", CONFIG["lbfgs_steps"]),
)
CONFIG["lbfgs_max_iter"] = get_env_int("LBFGS_MAX_ITER", CONFIG["lbfgs_max_iter"])
CONFIG["log_every"] = get_env_int("LOG_EVERY", CONFIG["log_every"])
CONFIG["lbfgs_log_every"] = get_env_int("LBFGS_LOG_EVERY", CONFIG["lbfgs_log_every"])

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_default_dtype(CONFIG["default_dtype"])
os.makedirs(CONFIG["save_dir"], exist_ok=True)
LAMD_TAG = build_lamd_tag(lamd)
RUN_SAVE_DIR = os.path.join(CONFIG["save_dir"], f"lamd_{LAMD_TAG}")
os.makedirs(RUN_SAVE_DIR, exist_ok=True)


def get_runtime_device() -> torch.device:
    requested = os.environ.get("QNN_DEVICE", "auto").strip().lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("QNN_DEVICE requests CUDA, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if requested not in {"", "auto"}:
        print(f"Unrecognized QNN_DEVICE='{requested}'. Using auto device detection.")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xavier_init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def should_evaluate(step: int, total_steps: int, eval_every: int) -> bool:
    return step == 1 or step == total_steps or (eval_every > 0 and step % eval_every == 0)


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class DVQuantumLayer(nn.Module):
    """
    DV quantum layer used as the quantum core inside the Poisson PINN.

    This implementation first tries batched QNode execution.
    If the installed PennyLane/backend does not support it for the current setup,
    it falls back to per-sample execution.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        num_quantum_layers: int = 15,
        q_ansatz: AnsatzType = "cascade",
        encoding: EncodingType = "angle",
        shots: Optional[int] = None,
        diff_method: str = "backprop",
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        self.num_qubits = num_qubits
        self.num_quantum_layers = num_quantum_layers
        self.q_ansatz = q_ansatz
        self.encoding = encoding
        self.shots = shots
        self.dtype = dtype

        self.quantum_device = torch.device(device) if device is not None else get_runtime_device()
        if self.quantum_device.type != "cuda":
            raise RuntimeError(
                "DVQuantumLayer is configured for full-GPU training. "
                "Please run with QNN_DEVICE=cuda on a GPU node."
            )

        params_per_layer = self._params_per_layer(q_ansatz, num_qubits)
        self.theta = nn.Parameter(
            torch.empty(
                num_quantum_layers,
                params_per_layer,
                dtype=self.dtype,
                device=self.quantum_device,
            )
        )
        nn.init.xavier_normal_(self.theta)

        self.dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method=diff_method,
        )

        # None = not tested yet; True = batched works; False = fallback to loop
        self._batched_qnode_supported: Optional[bool] = None

    @staticmethod
    def _params_per_layer(q_ansatz: AnsatzType, num_qubits: int) -> int:
        if q_ansatz == "layered":
            return 4 * num_qubits
        if q_ansatz == "alternate":
            return 4 * (num_qubits - 1)
        if q_ansatz == "cascade":
            return 3 * num_qubits
        if q_ansatz == "cross_mesh":
            return 4 * num_qubits + num_qubits * (num_qubits - 1)
        raise ValueError(f"Unsupported q_ansatz: {q_ansatz}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")

        x = x.to(device=self.quantum_device, dtype=self.theta.dtype)

        if self._batched_qnode_supported is not False:
            try:
                q_out = self.qnode(x, self.theta)
                q_out = self._coerce_qnode_output(q_out)

                if q_out.dim() == 1 and x.shape[0] == 1:
                    q_out = q_out.unsqueeze(0)

                # Common cases:
                # 1) [batch, num_qubits]
                # 2) [num_qubits, batch] -> transpose
                if q_out.dim() == 2:
                    if q_out.shape[0] == x.shape[0]:
                        self._batched_qnode_supported = True
                        return q_out
                    if q_out.shape[1] == x.shape[0]:
                        self._batched_qnode_supported = True
                        return q_out.transpose(0, 1)

                # If shape is not recognized, treat as unsupported
                self._batched_qnode_supported = False
            except Exception:
                self._batched_qnode_supported = False

        outputs = []
        for sample in x:
            q_out = self.qnode(sample, self.theta)
            outputs.append(self._coerce_qnode_output(q_out))
        return torch.stack(outputs, dim=0)

    def _coerce_qnode_output(self, output: torch.Tensor | list | tuple) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output.to(device=self.quantum_device, dtype=self.theta.dtype)
        if isinstance(output, (list, tuple)):
            return torch.stack([self._coerce_qnode_output(item) for item in output], dim=0)
        return torch.as_tensor(output, device=self.quantum_device, dtype=self.theta.dtype)

    def _encode(self, x: torch.Tensor) -> None:
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(
                features=x,
                wires=range(self.num_qubits),
                normalize=True,
                pad_with=0.0,
            )
        elif self.encoding == "angle":
            qml.templates.AngleEmbedding(
                features=x,
                wires=range(self.num_qubits),
                rotation="X",
            )
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

    def _circuit(self, x: torch.Tensor, theta: torch.Tensor):
        self._encode(x)

        for layer_idx in range(self.num_quantum_layers):
            params = theta[layer_idx]

            if self.q_ansatz == "layered":
                self._layered(params)
            elif self.q_ansatz == "alternate":
                self._alternate(params)
            elif self.q_ansatz == "cascade":
                self._cascade(params)
            elif self.q_ansatz == "cross_mesh":
                self._cross_mesh(params)
            else:
                raise ValueError(f"Unsupported q_ansatz: {self.q_ansatz}")

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _layered(self, params: torch.Tensor) -> None:
        expected = 4 * self.num_qubits
        if len(params) != expected:
            raise ValueError(f"layered expects {expected} params, got {len(params)}")

        idx = 0
        for qubit_id in range(self.num_qubits):
            qml.RZ(params[idx], wires=qubit_id)
            idx += 1
            qml.RX(params[idx], wires=qubit_id)
            idx += 1

        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])

        for qubit_id in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit_id)
            idx += 1
            qml.RZ(params[idx], wires=qubit_id)
            idx += 1

    def _alternate(self, params: torch.Tensor) -> None:
        expected = 4 * (self.num_qubits - 1)
        if len(params) != expected:
            raise ValueError(f"alternate expects {expected} params, got {len(params)}")

        idx = 0

        def dressed_cnot(ctrl: int, tgt: int) -> None:
            nonlocal idx
            qml.RY(params[idx], wires=ctrl)
            idx += 1
            qml.RY(params[idx], wires=tgt)
            idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[idx], wires=ctrl)
            idx += 1
            qml.RZ(params[idx], wires=tgt)
            idx += 1

        for i in range(0, self.num_qubits - 1, 2):
            dressed_cnot(i, i + 1)
        for i in range(1, self.num_qubits - 1, 2):
            dressed_cnot(i, i + 1)

    def _cascade(self, params: torch.Tensor) -> None:
        expected = 3 * self.num_qubits
        if len(params) != expected:
            raise ValueError(f"cascade expects {expected} params, got {len(params)}")

        idx = 0
        for qubit_id in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit_id)
            idx += 1

        for qubit_id in range(self.num_qubits):
            qml.RZ(params[idx], wires=qubit_id)
            idx += 1

        qml.CRX(params[idx], wires=[self.num_qubits - 1, 0])
        idx += 1
        for qubit_id in range(self.num_qubits - 1):
            qml.CRX(params[idx], wires=[qubit_id, qubit_id + 1])
            idx += 1

    def _cross_mesh(self, params: torch.Tensor) -> None:
        expected = 4 * self.num_qubits + self.num_qubits * (self.num_qubits - 1)
        if len(params) != expected:
            raise ValueError(f"cross_mesh expects {expected} params, got {len(params)}")

        idx = 0
        for qubit_id in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit_id)
            idx += 1

        for qubit_id in range(self.num_qubits):
            qml.RZ(params[idx], wires=qubit_id)
            idx += 1

        for ctrl in range(self.num_qubits - 1, -1, -1):
            for tgt in range(self.num_qubits - 1, -1, -1):
                if ctrl != tgt:
                    qml.CRZ(params[idx], wires=[ctrl, tgt])
                    idx += 1

        for qubit_id in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit_id)
            idx += 1

        for qubit_id in range(self.num_qubits):
            qml.RZ(params[idx], wires=qubit_id)
            idx += 1


class DVQCPINN(nn.Module):
    """
    Classical preprocessor -> DV quantum layer -> classical postprocessor.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 20,
        num_qubits: int = 8,
        num_quantum_layers: int = 15,
        q_ansatz: AnsatzType = "cascade",
        encoding: EncodingType = "angle",
        shots: Optional[int] = None,
        diff_method: str = "backprop",
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else get_runtime_device()

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_qubits, dtype=self.dtype),
        ).to(self.device)

        self.quantum_layer = DVQuantumLayer(
            num_qubits=num_qubits,
            num_quantum_layers=num_quantum_layers,
            q_ansatz=q_ansatz,
            encoding=encoding,
            shots=shots,
            diff_method=diff_method,
            dtype=self.dtype,
            device=self.device,
        )

        self.postprocessor = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, dtype=self.dtype),
        ).to(self.device)

        self.preprocessor.apply(xavier_init_linear)
        self.postprocessor.apply(xavier_init_linear)

    def build_quantum_input(self, x: torch.Tensor) -> torch.Tensor:
        z = self.preprocessor(x)
        z = wrap_to_pi(torch.pi * torch.tanh(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {tuple(x.shape)}")

        x = x.to(device=self.device, dtype=self.dtype)
        z = self.build_quantum_input(x)
        q = self.quantum_layer(z).to(device=self.device, dtype=self.dtype)
        y = self.postprocessor(q)
        return y


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for LBFGS optimization.")

    non_cuda_devices = sorted({str(param.device) for param in trainable_params if param.device.type != "cuda"})
    if non_cuda_devices:
        raise RuntimeError(
            "Full-GPU mode requires all trainable parameters on CUDA, but found: "
            f"{', '.join(non_cuda_devices)}"
        )

    return torch.optim.LBFGS(
        trainable_params,
        lr=CONFIG["lbfgs_lr"],
        max_iter=CONFIG["lbfgs_max_iter"],
        history_size=CONFIG["lbfgs_history_size"],
        tolerance_grad=CONFIG["lbfgs_tolerance_grad"],
        tolerance_change=CONFIG["lbfgs_tolerance_change"],
        line_search_fn=CONFIG["lbfgs_line_search_fn"],
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def u_exact(x: torch.Tensor) -> torch.Tensor:
    wave_number = 0.5 * lamd * torch.pi
    return torch.sin(wave_number * x[:, 0]) * torch.sin(wave_number * x[:, 1])


def f_source(x: torch.Tensor) -> torch.Tensor:
    wave_number = 0.5 * lamd * torch.pi
    return 2.0 * (wave_number ** 2) * u_exact(x)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target) / (torch.linalg.norm(target) + CONFIG["rel_l2_eps"])


def pde_residual(model: nn.Module, x: torch.Tensor, return_u: bool = False):
    x = x.clone().detach().requires_grad_(True)
    u = model(x).squeeze(-1)

    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[:, 0]
    u_y = grad_u[:, 1]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1]

    residual = -u_xx - u_yy - f_source(x)
    if return_u:
        return residual, u
    return residual


def generate_points(
    num_points: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=CONFIG["sobol_scramble"])
    points = engine.draw(num_points).to(device=device, dtype=CONFIG["default_dtype"])
    points = points * (CONFIG["domain_max"] - CONFIG["domain_min"]) + CONFIG["domain_min"]
    points = points + (torch.rand_like(points) - 0.5) * CONFIG["jitter_scale"]
    return points.clamp(CONFIG["domain_min"], CONFIG["domain_max"])


def sample_training_points(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_int = generate_points(CONFIG["num_interior_points"], CONFIG["classical_input_dim"], device)
    t_bnd = generate_points(CONFIG["num_bnd_per_edge"], 1, device).squeeze(-1)

    x_bnd = torch.cat(
        [
            torch.stack([torch.full_like(t_bnd, CONFIG["domain_min"]), t_bnd], dim=1),
            torch.stack([torch.full_like(t_bnd, CONFIG["domain_max"]), t_bnd], dim=1),
            torch.stack([t_bnd, torch.full_like(t_bnd, CONFIG["domain_min"])], dim=1),
            torch.stack([t_bnd, torch.full_like(t_bnd, CONFIG["domain_max"])], dim=1),
        ],
        dim=0,
    )
    return x_int, x_bnd


def make_validation_points(device: torch.device) -> torch.Tensor:
    n = CONFIG["validation_grid_n"]
    xs = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    ys = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).view(
        -1, CONFIG["classical_input_dim"]
    ).to(device)


def clip_gradients_by_device(model: nn.Module, max_norm: float) -> None:
    if max_norm <= 0:
        return

    params_by_device: dict[torch.device, list[torch.nn.Parameter]] = {}
    for param in model.parameters():
        if param.grad is None:
            continue
        params_by_device.setdefault(param.grad.device, []).append(param)

    for params in params_by_device.values():
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def batched_predict(model: DVQCPINN, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
    batch_size = batch_size or CONFIG["eval_batch_size"]
    was_training = model.training
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size]
            pred = model(chunk).squeeze(-1)
            outputs.append(pred.detach().cpu())

    if was_training:
        model.train()

    return torch.cat(outputs, dim=0)


def evaluate_rel_l2(model: DVQCPINN, x_eval: torch.Tensor, u_eval: torch.Tensor) -> float:
    pred = batched_predict(model, x_eval)
    rel_l2 = relative_l2_error(pred, u_eval.cpu())
    return float(rel_l2.item())


def plot_rel_l2_history(history: list[tuple[int, float]]) -> None:
    if plt is None or not history:
        print("Skipping Rel L2 history plot because matplotlib is unavailable or history is empty.")
        return

    epochs = [epoch for epoch, _ in history]
    rel_l2_values = [value for _, value in history]

    fig, ax = plt.subplots(figsize=CONFIG["history_figsize"])
    ax.plot(epochs, rel_l2_values, color="tab:blue", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(f"Relative L2 Error During Training (every {CONFIG['eval_every']} steps)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['rel_l2_history_filename']}.png")
    pdf_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['rel_l2_history_filename']}.pdf")
    fig.savefig(png_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Rel L2 history plot to {png_path}")
    print(f"Saved Rel L2 history plot to {pdf_path}")


def plot_solution_comparison(model: DVQCPINN, device: torch.device, arch_name: str) -> float | None:
    if plt is None or np is None:
        print("Skipping final solution plot because matplotlib or numpy is unavailable.")
        return None

    n = CONFIG["plot_grid_n"]
    xs = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    ys = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).view(-1, 2).to(device)

    pred = batched_predict(model, grid)
    exact = u_exact(grid.cpu())
    rel_l2 = relative_l2_error(pred, exact)

    pred_np = pred.view(n, n).numpy()
    exact_np = exact.view(n, n).numpy()
    err_np = np.abs(pred_np - exact_np)

    fig, axes = plt.subplots(1, 3, figsize=CONFIG["plot_figsize"])
    vmin = float(exact_np.min())
    vmax = float(exact_np.max())

    im0 = axes[0].imshow(
        exact_np,
        origin="lower",
        extent=[CONFIG["domain_min"], CONFIG["domain_max"], CONFIG["domain_min"], CONFIG["domain_max"]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Exact Solution")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        pred_np,
        origin="lower",
        extent=[CONFIG["domain_min"], CONFIG["domain_max"], CONFIG["domain_min"], CONFIG["domain_max"]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("QNN Prediction")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_np,
        origin="lower",
        extent=[CONFIG["domain_min"], CONFIG["domain_max"], CONFIG["domain_min"], CONFIG["domain_max"]],
        cmap="Reds",
    )
    axes[2].set_title("Absolute Error")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Poisson {arch_name} (Rel L2: {rel_l2.item():.2e})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    png_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['plot_filename_prefix']}.png")
    pdf_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['plot_filename_prefix']}.pdf")
    fig.savefig(png_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFinal Grid Relative L2 Error: {rel_l2.item():.4e}")
    print(f"Saved solution plot to {png_path}")
    print(f"Saved solution plot to {pdf_path}")
    return float(rel_l2.item())


def print_structure_summary() -> None:
    print(
        f"Code structure: (x, y) -> MLP({CONFIG['hidden_dim']}) -> "
        f"DV-QNN({CONFIG['num_qubits']} qubits, {CONFIG['num_quantum_layers']} layers, "
        f"{CONFIG['encoding']}+{CONFIG['q_ansatz']}) -> "
        f"MLP({CONFIG['hidden_dim']}) -> u(x, y)"
    )
    print(
        "Quantum core settings: "
        f"num_qubits={CONFIG['num_qubits']}, "
        f"num_quantum_layers={CONFIG['num_quantum_layers']}, "
        f"encoding={CONFIG['encoding']}, "
        f"q_ansatz={CONFIG['q_ansatz']}"
    )
    print(
        f"Runtime changes: dtype={CONFIG['default_dtype']}, "
        f"eval_every={CONFIG['eval_every']}, "
        f"validation_grid={CONFIG['validation_grid_n']}x{CONFIG['validation_grid_n']}, "
        "optimizer=Adam+LBFGS, "
        "quantum forward=try batched QNode first"
    )


def run_serial_helmholtz() -> None:
    print("========== Current run [DV-QNN 2D Poisson PINN] ==========\n")
    device = get_runtime_device()
    print(f"Requested device (QNN_DEVICE): {os.environ.get('QNN_DEVICE', 'auto')}")
    print(f"Using device: {device}")
    print(f"lamd = {lamd:.1f}")
    print(f"Output dir = {RUN_SAVE_DIR}")
    print(f"Adam epochs = {CONFIG['adam_steps']}, LBFGS epochs = {CONFIG['lbfgs_steps']}")
    print(
        f"Validation grid = {CONFIG['validation_grid_n']} x {CONFIG['validation_grid_n']} "
        f"({CONFIG['validation_grid_n'] ** 2} points)"
    )
    if device.type != "cuda":
        raise RuntimeError("Full-GPU mode requires CUDA. Please set QNN_DEVICE=cuda and run on a GPU node.")
    print_structure_summary()

    model = DVQCPINN(
        input_dim=CONFIG["classical_input_dim"],
        output_dim=CONFIG["classical_output_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_qubits=CONFIG["num_qubits"],
        num_quantum_layers=CONFIG["num_quantum_layers"],
        q_ansatz=CONFIG["q_ansatz"],
        encoding=CONFIG["encoding"],
        shots=CONFIG["shots"],
        diff_method=CONFIG["diff_method"],
        dtype=CONFIG["default_dtype"],
        device=device,
    )

    x_eval = make_validation_points(device)
    u_eval = u_exact(x_eval.cpu())
    x_int, x_bnd = sample_training_points(device)

    print(f"Trainable params: {count_trainable_parameters(model)}")

    best_loss = float("inf")
    best_state = None
    rel_l2_history: list[tuple[int, float]] = []
    latest_rel_l2: Optional[float] = None
    train_start_time = time.perf_counter()

    def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_local, _ = pde_residual(model, x_int, return_u=True)
        loss_pde_local = torch.mean(residual_local.square())
        u_bnd_pred_local = model(x_bnd).squeeze(-1)
        loss_bnd_local = torch.mean((u_bnd_pred_local - u_exact(x_bnd)) ** 2)
        loss_local = loss_pde_local + CONFIG["boundary_loss_weight"] * loss_bnd_local
        return loss_local, loss_pde_local, loss_bnd_local

    # Adam warm-up stage
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["adam_lr"])
    adam_scheduler = torch.optim.lr_scheduler.StepLR(
        adam_optimizer,
        step_size=CONFIG["scheduler_step_size"],
        gamma=CONFIG["scheduler_gamma"],
    )

    adam_start_time = time.perf_counter()
    for step in range(1, CONFIG["adam_steps"] + 1):
        if step % CONFIG["resample_every"] == 1 and step != 1:
            x_int, x_bnd = sample_training_points(device)

        loss, loss_pde, loss_bnd = compute_losses()
        adam_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        adam_optimizer.step()
        adam_scheduler.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if step % CONFIG["log_every"] == 0:
            current_lr = adam_optimizer.param_groups[0]["lr"]
            q_param_norm = model.quantum_layer.theta.data.norm().item()
            with torch.no_grad():
                u_int_pred = model(x_int).squeeze(-1)
                u_int_true = u_exact(x_int.cpu()).to(device=device, dtype=CONFIG["default_dtype"])
                rel_l2_now = float(relative_l2_error(u_int_pred, u_int_true).item())
            adam_stage_elapsed = time.perf_counter() - adam_start_time
            print(
                f"[Adam] step {step:05d} | "
                f"Loss: {loss_value:.4e} | "
                f"Loss_pde: {loss_pde.item():.4e} | "
                f"Loss_bnd: {loss_bnd.item():.4e} | "
                f"Rel L2: {rel_l2_now:.4e} | "
                f"q_param_norm: {q_param_norm:.4e} | "
                f"lr: {current_lr:.3e} | "
                f"elapsed: {adam_stage_elapsed:.2f}s ({format_duration(adam_stage_elapsed)})"
            )
    adam_elapsed_seconds = time.perf_counter() - adam_start_time
    print(
        f"Adam training time ({CONFIG['adam_steps']} epochs): {adam_elapsed_seconds:.2f} s "
        f"({format_duration(adam_elapsed_seconds)})"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    # LBFGS stage
    optimizer = make_optimizer(model)
    print("LBFGS optimizer devices: cuda")

    def closure() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        loss_local, _, _ = compute_losses()
        loss_local.backward()
        clip_gradients_by_device(model, CONFIG["grad_clip_max_norm"])
        return loss_local

    lbfgs_start_time = time.perf_counter()
    for step in range(1, CONFIG["lbfgs_steps"] + 1):
        if step % CONFIG["resample_every"] == 1 and step != 1:
            x_int, x_bnd = sample_training_points(device)

        optimizer.step(closure)
        loss, loss_pde, loss_bnd = compute_losses()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if should_evaluate(step, CONFIG["lbfgs_steps"], CONFIG["eval_every"]):
            with torch.no_grad():
                u_int_pred = model(x_int).squeeze(-1)
                u_int_true = u_exact(x_int.cpu()).to(device=device, dtype=CONFIG["default_dtype"])
                latest_rel_l2 = float(relative_l2_error(u_int_pred, u_int_true).item())
            rel_l2_history.append((step, latest_rel_l2))

        if step % CONFIG["lbfgs_log_every"] == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            q_param_norm = model.quantum_layer.theta.data.norm().item()
            with torch.no_grad():
                u_int_pred = model(x_int).squeeze(-1)
                u_int_true = u_exact(x_int.cpu()).to(device=device, dtype=CONFIG["default_dtype"])
                rel_l2_now = float(relative_l2_error(u_int_pred, u_int_true).item())
            lbfgs_stage_elapsed = time.perf_counter() - lbfgs_start_time
            print(
                f"[LBFGS] step {step:05d} | "
                f"Loss: {loss_value:.4e} | "
                f"Loss_pde: {loss_pde.item():.4e} | "
                f"Loss_bnd: {loss_bnd.item():.4e} | "
                f"Rel L2: {rel_l2_now:.4e} | "
                f"q_param_norm: {q_param_norm:.4e} | "
                f"lr: {current_lr:.3e} | "
                f"elapsed: {lbfgs_stage_elapsed:.2f}s ({format_duration(lbfgs_stage_elapsed)})"
            )
    lbfgs_elapsed_seconds = time.perf_counter() - lbfgs_start_time
    print(
        f"LBFGS training time ({CONFIG['lbfgs_steps']} epochs): {lbfgs_elapsed_seconds:.2f} s "
        f"({format_duration(lbfgs_elapsed_seconds)})"
    )

    train_elapsed_seconds = time.perf_counter() - train_start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    final_rel_l2 = evaluate_rel_l2(model, x_eval, u_eval)

    model_path = os.path.join(RUN_SAVE_DIR, CONFIG["model_filename"])
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    print(f"Best training loss: {best_loss:.6e}")
    print(
        f"Training time: {train_elapsed_seconds:.2f} s "
        f"({format_duration(train_elapsed_seconds)})"
    )
    print(
        f"[Timing Summary] lamd={lamd:.1f} | "
        f"Adam({CONFIG['adam_steps']})={adam_elapsed_seconds:.2f}s | "
        f"LBFGS({CONFIG['lbfgs_steps']})={lbfgs_elapsed_seconds:.2f}s"
    )
    print("\n========== Validation (100x100 Uniform Grid) ==========")
    print(f"Validation Rel L2: {final_rel_l2:.4e}")

    timing_filename = f"{os.path.splitext(CONFIG['model_filename'])[0]}_timing_summary.txt"
    timing_path = os.path.join(RUN_SAVE_DIR, timing_filename)
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(f"lamd={lamd:.1f}\n")
        f.write(f"adam_epochs={CONFIG['adam_steps']}\n")
        f.write(f"lbfgs_epochs={CONFIG['lbfgs_steps']}\n")
        f.write(f"adam_seconds={adam_elapsed_seconds:.6f}\n")
        f.write(f"lbfgs_seconds={lbfgs_elapsed_seconds:.6f}\n")
        f.write(f"total_seconds={train_elapsed_seconds:.6f}\n")
    print(f"Saved timing summary to {timing_path}")

    plot_rel_l2_history(rel_l2_history)
    plot_solution_comparison(model, device, CONFIG["arch_name_serial"])


def run_parallel_helmholtz() -> None:
    run_serial_helmholtz()


if __name__ == "__main__":
    run_serial_helmholtz()
