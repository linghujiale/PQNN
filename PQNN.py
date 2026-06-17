"""
PQNN-L2 单分支变体说明（删除 PQNNA，保留 PQNNB）：
1. 保留量子网络前后的经典 MLP。
2. 删除并行双量子分支中的 A 分支（mode_expectations）。
3. 仅保留 B 分支（probs 测量）。
4. 整体结构变为：
   x -> pre-quantum MLP -> simple quantum input projection
     -> single PQNN (probs)
     -> post-quantum MLP -> u
"""

from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import merlin as ML

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


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


# ============================================================
# 0. 配置
# ============================================================
def get_config() -> Dict[str, Any]:
    return {
        # ---------------- 基本设置 ----------------
        "default_dtype": torch.float32,
        "save_dir": "figures",
        "model_filename": "poisson_single_branch_probs_2block.pt",
        "plot_filename_prefix": "poisson_single_branch_probs_2block_PQNN",
        "arch_name_serial": "Single-Branch Photonic QPINN (2D Poisson, keep probs branch, 2 blocks)",

        # ---------------- 光量子网络结构 ----------------
        "n_modes": 4,
        "quantum_input_dim": 4,
        "n_reupload_blocks": 2,
        "measurement_type": "probs",
        "input_state": [1, 1, 0, 0],

        # ---------------- 经典网络宽度 ----------------
        "classical_input_dim": 2,
        "pre_quantum_hidden_dim": 20,
        "post_quantum_hidden_dim": 20,

        # ---------------- 单分支输入仿射初始化 ----------------
        "quantum_input_gain_init": 1.00,

        # ---------------- PDE 参数（2D Poisson） ----------------
        "domain_min": -1.0,
        "domain_max": 1.0,

        # ---------------- 采样参数 ----------------
        "num_interior_points": 256,
        "num_bnd_per_edge": 128,
        "jitter_scale": 0.02,
        "resample_every": 100,
        "sobol_scramble": True,

        # ---------------- Adam 参数 ----------------
        "adam_lr": 5e-3,
        "adam_steps": 1000,
        "boundary_loss_weight": 50.0,
        "scheduler_step_size": 800,
        "scheduler_gamma": 0.7,
        "log_every": 100,

        # ---------------- LBFGS 参数 ----------------
        "lbfgs_lr": 0.1,
        "lbfgs_max_iter": 20,
        "lbfgs_line_search_fn": "strong_wolfe",
        "lbfgs_epochs": 200,
        "lbfgs_log_every": 20,
        "lbfgs_patience": 100,
        "lbfgs_min_delta": 1e-14,

        # ---------------- 可视化参数 ----------------
        "rel_l2_eps": 1e-8,
        "validation_grid_n": 100,
        "plot_grid_n": 100,
        "plot_figsize": (15, 4),
        "plot_dpi": 150,
    }

CONFIG = get_config()
lamd = get_env_float("LAMD", 5.0)
CONFIG["adam_steps"] = get_env_int("ADAM_STEPS", CONFIG["adam_steps"])
CONFIG["lbfgs_epochs"] = get_env_int("LBFGS_EPOCHS", CONFIG["lbfgs_epochs"])
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


# ============================================================
# 1. 基础工具函数
# ============================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def default_input_state(n_modes: int) -> List[int]:
    state = [1 if i % 2 == 0 else 0 for i in range(n_modes)]
    if sum(state) == 0:
        state[0] = 1
    return state


def normalize_input_state(state: List[int] | None, n_modes: int) -> List[int]:
    if state is None:
        return default_input_state(n_modes)

    state = list(state)
    if len(state) < n_modes:
        state = state + [0] * (n_modes - len(state))
    elif len(state) > n_modes:
        state = state[:n_modes]

    if sum(state) == 0:
        state[0] = 1
    return state


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _resolve_measurement_strategy(measurement_type: str) -> Tuple[Any, str]:
    measurement_type = measurement_type.lower().strip()
    ms = ML.MeasurementStrategy
    comp_space = getattr(ML, "ComputationSpace", None)

    if measurement_type in {"probs", "probabilities", "fock_probs"} and hasattr(ms, "probs"):
        attempts = []
        if comp_space is not None and hasattr(comp_space, "FOCK"):
            attempts.extend([
                lambda: ms.probs(computation_space=comp_space.FOCK),
                lambda: ms.probs(comp_space.FOCK),
            ])
        attempts.append(lambda: ms.probs())
        for fn in attempts:
            try:
                return fn(), "probs"
            except TypeError:
                continue
            except Exception:
                continue
        print("[Warning] probs measurement unavailable, fallback to mode_expectations.")

    if hasattr(ms, "mode_expectations"):
        attempts = []
        if comp_space is not None and hasattr(comp_space, "FOCK"):
            attempts.extend([
                lambda: ms.mode_expectations(computation_space=comp_space.FOCK),
                lambda: ms.mode_expectations(comp_space.FOCK),
            ])
        attempts.append(lambda: ms.mode_expectations())
        for fn in attempts:
            try:
                return fn(), "mode_expectations"
            except TypeError:
                continue
            except Exception:
                continue

    raise RuntimeError("Failed to construct a valid MerLin measurement strategy.")


def _try_builder_call(builder, method_name: str, call_specs: list[tuple[tuple, dict]]) -> bool:
    if not hasattr(builder, method_name):
        return False

    method = getattr(builder, method_name)
    for args, kwargs in call_specs:
        try:
            method(*args, **kwargs)
            return True
        except TypeError:
            continue
        except Exception:
            continue
    return False


def _add_rotations(builder, name: str) -> bool:
    return _try_builder_call(
        builder,
        "add_rotations",
        [
            ((), {"trainable": True, "name": name}),
            ((), {"trainable": True}),
            ((), {"name": name}),
            ((), {}),
        ],
    )


def _add_superpositions(builder, name: str) -> bool:
    return _try_builder_call(
        builder,
        "add_superpositions",
        [
            ((), {"trainable": True, "name": name}),
            ((), {"trainable": True}),
            ((), {"name": name}),
            ((), {}),
        ],
    )


def _add_entangling(builder, name: str) -> bool:
    return _try_builder_call(
        builder,
        "add_entangling_layer",
        [
            ((), {"trainable": True, "name": name}),
            ((), {"trainable": True}),
            ((), {"name": name}),
            ((), {}),
        ],
    )


def _add_angle_encoding(builder, modes: list[int], name: str) -> bool:
    return _try_builder_call(
        builder,
        "add_angle_encoding",
        [
            ((), {"modes": modes, "name": name}),
            ((), {"modes": modes}),
            ((modes,), {"name": name}),
            ((modes,), {}),
            ((), {"name": name}),
            ((), {}),
        ],
    )


def build_quantum_layer(
    n_modes: int,
    input_dim: int,
    n_reupload_blocks: int,
    input_state: List[int] | None,
    measurement_type: str,
):
    input_state = normalize_input_state(input_state, n_modes)

    builder = ML.CircuitBuilder(n_modes=n_modes)
    modes = list(range(min(input_dim, n_modes)))

    for i in range(n_reupload_blocks):
        _add_rotations(builder, f"pre_rot_{i+1}")
        _add_angle_encoding(builder, modes, f"encoding_phase_{i+1}")
        _add_superpositions(builder, f"superpose_a_{i+1}")
        _add_entangling(builder, f"entangle_a_{i+1}")

        _add_rotations(builder, f"mid_rot_{i+1}")
        _add_angle_encoding(builder, modes, f"encoding_mix_{i+1}")
        _add_superpositions(builder, f"superpose_b_{i+1}")
        _add_entangling(builder, f"entangle_b_{i+1}")

        _add_rotations(builder, f"post_rot_{i+1}")

    measurement_strategy, actual_measurement = _resolve_measurement_strategy(measurement_type)
    qlayer = ML.QuantumLayer(
        builder=builder,
        input_state=input_state,
        measurement_strategy=measurement_strategy,
    )
    return qlayer, actual_measurement, input_state


# ============================================================
# 2. 单分支网络（删除 A，保留 B）
# ============================================================
class SingleBranchPhotonicPINN(nn.Module):
    """
    结构：
    x -> classical_stem(MLP) -> simple quantum projection
      -> single PQNN (probs) -> readout(MLP) -> u

    和原双分支版本相比：
    1. 删除 PQNNA
    2. 删除 branch_logits 以及双分支加权拼接
    3. 仅保留一套量子输入仿射参数
    """

    def __init__(
        self,
        n_modes: int = CONFIG["n_modes"],
        input_dim: int = CONFIG["quantum_input_dim"],
        n_reupload_blocks: int = CONFIG["n_reupload_blocks"],
        input_state: List[int] | None = None,
        measurement_type: str = CONFIG["measurement_type"],
        device=None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = CONFIG["default_dtype"]

        self.n_modes = n_modes
        self.input_dim = input_dim
        self.n_reupload_blocks = n_reupload_blocks
        self.in_dim = CONFIG["classical_input_dim"]
        self.q_input_total_dim = 2 * self.input_dim * self.n_reupload_blocks

        self.measurement_type = measurement_type

        # 量子网络前的 MLP
        self.classical_stem = nn.Sequential(
            nn.Linear(self.in_dim, CONFIG["pre_quantum_hidden_dim"], dtype=self.dtype),
            nn.Tanh(),
        ).to(self.device)

        # 简单量子输入投影
        self.quantum_input_proj = nn.Linear(
            CONFIG["pre_quantum_hidden_dim"],
            self.q_input_total_dim,
            dtype=self.dtype,
        ).to(self.device)

        # 单分支输入仿射
        self.quantum_input_gain = nn.Parameter(
            CONFIG["quantum_input_gain_init"] * torch.ones(
                self.q_input_total_dim, dtype=self.dtype, device=self.device
            )
        )
        self.quantum_input_bias = nn.Parameter(
            torch.zeros(self.q_input_total_dim, dtype=self.dtype, device=self.device)
        )

        # 单量子分支
        self.qnet, self.actual_measurement, self.input_state = build_quantum_layer(
            n_modes=n_modes,
            input_dim=input_dim,
            n_reupload_blocks=n_reupload_blocks,
            input_state=input_state,
            measurement_type=measurement_type,
        )
        self.qnet = self.qnet.to(self.device)

        with torch.no_grad():
            dummy = torch.zeros(1, self.q_input_total_dim, device=self.device, dtype=self.dtype)
            qout = self.qnet(dummy).to(dtype=self.dtype, device=self.device)
            q_feat_dim = qout.shape[-1]

        # 量子网络后的 MLP
        self.readout = nn.Sequential(
            nn.Linear(q_feat_dim, CONFIG["post_quantum_hidden_dim"], dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(CONFIG["post_quantum_hidden_dim"], 1, dtype=self.dtype),
        ).to(self.device)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.classical_stem, self.quantum_input_proj, self.readout]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def build_quantum_input(self, h: torch.Tensor) -> torch.Tensor:
        return self.quantum_input_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.dtype)
        h = self.classical_stem(x)

        q_in_base = self.build_quantum_input(h)
        q_in = wrap_to_pi(
            q_in_base * self.quantum_input_gain.unsqueeze(0)
            + self.quantum_input_bias.unsqueeze(0)
        )

        q_feat = self.qnet(q_in).to(dtype=self.dtype, device=self.device)
        return self.readout(q_feat).squeeze(-1)


# ============================================================
# 2.1 参数统计工具
# ============================================================
def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def count_tensor_parameters(param: torch.Tensor) -> int:
    return int(param.numel())


def print_parameter_summary(model: nn.Module) -> None:
    print("\n" + "=" * 72)
    print("Network Parameter Summary")
    print("=" * 72)

    classical_stem_params = count_parameters(model.classical_stem, trainable_only=False)
    quantum_input_proj_params = count_parameters(model.quantum_input_proj, trainable_only=False)
    qnet_params = count_parameters(model.qnet, trainable_only=False)
    readout_params = count_parameters(model.readout, trainable_only=False)

    quantum_input_gain_params = count_tensor_parameters(model.quantum_input_gain)
    quantum_input_bias_params = count_tensor_parameters(model.quantum_input_bias)

    total_params = (
        classical_stem_params
        + quantum_input_proj_params
        + quantum_input_gain_params
        + quantum_input_bias_params
        + qnet_params
        + readout_params
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'classical_stem':<32s}: {classical_stem_params:>10d}")
    print(f"{'quantum_input_proj':<32s}: {quantum_input_proj_params:>10d}")
    print(f"{'quantum_input_gain':<32s}: {quantum_input_gain_params:>10d}")
    print(f"{'quantum_input_bias':<32s}: {quantum_input_bias_params:>10d}")
    print(f"{'qnet':<32s}: {qnet_params:>10d}")
    print(f"{'readout':<32s}: {readout_params:>10d}")
    print("-" * 72)
    print(f"{'Total params':<32s}: {total_params:>10d}")
    print(f"{'Trainable params':<32s}: {trainable_params:>10d}")
    print("=" * 72)

    print("\nDetailed named parameters:")
    for name, param in model.named_parameters():
        print(
            f"  {name:<50s} shape={list(param.shape)!s:<18s} "
            f"numel={param.numel():>8d}  requires_grad={param.requires_grad}"
        )
    print()


# ============================================================
# 3. PDE / 可视化
# ============================================================
def u_exact(x: torch.Tensor) -> torch.Tensor:
    wave_number = 0.5 * lamd * torch.pi
    return torch.sin(wave_number * x[:, 0]) * torch.sin(wave_number * x[:, 1])


def f_source(x: torch.Tensor) -> torch.Tensor:
    wave_number = 0.5 * lamd * torch.pi
    return 2.0 * (wave_number ** 2) * u_exact(x)


def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.requires_grad_(True)
    u = model(x)

    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grad_u[:, 0], grad_u[:, 1]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1]

    return -u_xx - u_yy - f_source(x)


def make_validation_points(device: torch.device) -> torch.Tensor:
    n = CONFIG["validation_grid_n"]
    xs = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    ys = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).view(
        -1, CONFIG["classical_input_dim"]
    ).to(device)


def plot_results(model: nn.Module, device: torch.device, arch_name: str):
    if plt is None or np is None:
        return

    n = CONFIG["plot_grid_n"]
    xs = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    ys = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).view(-1, 2).to(device)

    with torch.no_grad():
        pred = model(grid).cpu()
        exact = u_exact(grid.cpu())

    rel_l2 = torch.linalg.norm(pred - exact) / (torch.linalg.norm(exact) + CONFIG["rel_l2_eps"])
    print(f"\nFinal Grid Relative L2 Error: {rel_l2.item():.4e}")

    pred_np = pred.view(n, n).numpy()
    exact_np = exact.view(n, n).numpy()
    err_np = np.abs(pred_np - exact_np)

    fig, axes = plt.subplots(1, 3, figsize=CONFIG["plot_figsize"])
    vmin, vmax = exact_np.min(), exact_np.max()

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
    axes[1].set_title(f"{arch_name} Prediction")
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
    plt.tight_layout()

    png_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['plot_filename_prefix']}.png")
    pdf_path = os.path.join(RUN_SAVE_DIR, f"{CONFIG['plot_filename_prefix']}.pdf")
    plt.savefig(png_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {png_path}")
    print(f"Saved plot to {pdf_path}")


# ============================================================
# 4. 训练
# ============================================================
def print_structure_summary():
    print(
        "Code structure: x -> pre-quantum MLP -> simple quantum input projection -> "
        "single PQNN (probs) -> post-quantum MLP -> u(x, y)"
    )
    print("Difference vs dual-branch version: remove PQNNA, keep only PQNNB.")
    print("Quantum depth setting: n_reupload_blocks = 2, so q_input_total_dim = 2 * 4 * 2 = 16.")


def run_serial_helmholtz():
    print("========== 当前运行 [Single-Branch PQNN: 2D Poisson, keep probs branch only, 2 blocks] ==========\n")
    device = get_device()
    print(f"Using device: {device}")
    print(f"lamd = {lamd:.1f}")
    print(f"Output dir = {RUN_SAVE_DIR}")
    print(f"Adam epochs = {CONFIG['adam_steps']}, LBFGS epochs = {CONFIG['lbfgs_epochs']}")
    print(
        f"Validation grid = {CONFIG['validation_grid_n']} x {CONFIG['validation_grid_n']} "
        f"({CONFIG['validation_grid_n'] ** 2} points)"
    )
    print_structure_summary()
    train_start_time = time.perf_counter()

    model = SingleBranchPhotonicPINN(
        n_modes=CONFIG["n_modes"],
        input_dim=CONFIG["quantum_input_dim"],
        n_reupload_blocks=CONFIG["n_reupload_blocks"],
        input_state=CONFIG["input_state"],
        measurement_type=CONFIG["measurement_type"],
        device=device,
    ).to(device)

    print(f"Requested measurement: {CONFIG['measurement_type']}")
    print(f"Actual measurement   : {model.actual_measurement}")
    print(f"Input state          : {model.input_state}")

    # 输出网络参数量
    print_parameter_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["adam_lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["scheduler_step_size"],
        gamma=CONFIG["scheduler_gamma"],
    )

    def gen_points(num, dim):
        engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=CONFIG["sobol_scramble"])
        pts = engine.draw(num).to(device=device, dtype=CONFIG["default_dtype"])
        pts = pts * (CONFIG["domain_max"] - CONFIG["domain_min"]) + CONFIG["domain_min"]
        pts = pts + (torch.rand_like(pts) - 0.5) * CONFIG["jitter_scale"]
        return pts.clamp(CONFIG["domain_min"], CONFIG["domain_max"])

    def sample_training_points():
        x_int = gen_points(CONFIG["num_interior_points"], CONFIG["classical_input_dim"])

        # 仅修改这里：边界上改为均匀取点，不再经过 jitter
        t_bnd = torch.linspace(
            CONFIG["domain_min"],
            CONFIG["domain_max"],
            CONFIG["num_bnd_per_edge"],
            device=device,
            dtype=CONFIG["default_dtype"],
        )

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

    x_int, x_bnd = sample_training_points()
    x_val = make_validation_points(device)
    u_val = u_exact(x_val)

    best_loss = float("inf")
    best_state = None

    adam_start_time = time.perf_counter()
    for step in range(1, CONFIG["adam_steps"] + 1):
        if step % CONFIG["resample_every"] == 1 and step != 1:
            x_int, x_bnd = sample_training_points()

        loss_pde = torch.mean(pde_residual(model, x_int) ** 2)
        loss_bnd = torch.mean((model(x_bnd) - u_exact(x_bnd)) ** 2)
        loss = loss_pde + CONFIG["boundary_loss_weight"] * loss_bnd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step % CONFIG["log_every"] == 0:
            with torch.no_grad():
                rel_l2 = torch.linalg.norm(model(x_int) - u_exact(x_int)) / (
                    torch.linalg.norm(u_exact(x_int)) + CONFIG["rel_l2_eps"]
                )
                gain_mean = model.quantum_input_gain.mean().item()
            adam_stage_elapsed = time.perf_counter() - adam_start_time

            print(
                f"Step {step:04d} | Loss: {loss_value:.4e} | "
                f"Rel L2: {rel_l2.item():.4e} | "
                f"quantum_input_gain_mean: {gain_mean:.3f} | "
                f"elapsed: {adam_stage_elapsed:.2f}s ({format_elapsed_time(adam_stage_elapsed)})"
            )
    adam_elapsed_seconds = time.perf_counter() - adam_start_time
    print(
        f"Adam training time ({CONFIG['adam_steps']} epochs): {adam_elapsed_seconds:.2f} s "
        f"({format_elapsed_time(adam_elapsed_seconds)})"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=CONFIG["lbfgs_lr"],
        max_iter=CONFIG["lbfgs_max_iter"],
        line_search_fn=CONFIG["lbfgs_line_search_fn"],
    )

    def closure():
        lbfgs_opt.zero_grad()
        loss_pde = torch.mean(pde_residual(model, x_int) ** 2)
        loss_bnd = torch.mean((model(x_bnd) - u_exact(x_bnd)) ** 2)
        loss = loss_pde + CONFIG["boundary_loss_weight"] * loss_bnd
        loss.backward()
        return loss

    print("Starting L-BFGS fine-tuning...", flush=True)
    lbfgs_start_time = time.perf_counter()

    lbfgs_bad_epochs = 0
    lbfgs_best_loss = best_loss

    for epoch in range(1, CONFIG["lbfgs_epochs"] + 1):
        loss = lbfgs_opt.step(closure)
        loss_value = float(loss.item())

        if loss_value + CONFIG["lbfgs_min_delta"] < lbfgs_best_loss:
            lbfgs_best_loss = loss_value
            lbfgs_bad_epochs = 0
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            lbfgs_bad_epochs += 1

        if epoch % CONFIG["lbfgs_log_every"] == 0:
            with torch.no_grad():
                rel_l2 = torch.linalg.norm(model(x_int) - u_exact(x_int)) / (
                    torch.linalg.norm(u_exact(x_int)) + CONFIG["rel_l2_eps"]
                )
                gain_mean = model.quantum_input_gain.mean().item()
            lbfgs_stage_elapsed = time.perf_counter() - lbfgs_start_time

            print(
                f"[LBFGS] epoch {epoch:03d} | "
                f"loss={loss_value:.4e} | "
                f"rel_l2={rel_l2.item():.4e} | "
                f"quantum_input_gain_mean={gain_mean:.3f} | "
                f"elapsed={lbfgs_stage_elapsed:.2f}s ({format_elapsed_time(lbfgs_stage_elapsed)})"
            )

        if lbfgs_bad_epochs >= CONFIG["lbfgs_patience"]:
            print(
                f"[LBFGS] Early stop triggered at epoch {epoch} "
                f"(no improvement for {CONFIG['lbfgs_patience']} epochs)."
            )
            break
    lbfgs_elapsed_seconds = time.perf_counter() - lbfgs_start_time
    print(
        f"LBFGS training time ({CONFIG['lbfgs_epochs']} epochs): {lbfgs_elapsed_seconds:.2f} s "
        f"({format_elapsed_time(lbfgs_elapsed_seconds)})"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        val_rel_l2 = torch.linalg.norm(model(x_val) - u_val) / (
            torch.linalg.norm(u_val) + CONFIG["rel_l2_eps"]
        )

    train_elapsed_seconds = time.perf_counter() - train_start_time

    model_path = os.path.join(RUN_SAVE_DIR, CONFIG["model_filename"])
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    print(
        "Training time: "
        f"{train_elapsed_seconds:.2f} s "
        f"({format_elapsed_time(train_elapsed_seconds)})"
    )
    print(
        f"[Timing Summary] lamd={lamd:.1f} | "
        f"Adam({CONFIG['adam_steps']})={adam_elapsed_seconds:.2f}s | "
        f"LBFGS({CONFIG['lbfgs_epochs']})={lbfgs_elapsed_seconds:.2f}s"
    )
    print("\n========== Validation (100x100 Uniform Grid) ==========")
    print(f"Validation Rel L2: {val_rel_l2.item():.4e}")

    timing_filename = f"{os.path.splitext(CONFIG['model_filename'])[0]}_timing_summary.txt"
    timing_path = os.path.join(RUN_SAVE_DIR, timing_filename)
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(f"lamd={lamd:.1f}\n")
        f.write(f"adam_epochs={CONFIG['adam_steps']}\n")
        f.write(f"lbfgs_epochs={CONFIG['lbfgs_epochs']}\n")
        f.write(f"adam_seconds={adam_elapsed_seconds:.6f}\n")
        f.write(f"lbfgs_seconds={lbfgs_elapsed_seconds:.6f}\n")
        f.write(f"total_seconds={train_elapsed_seconds:.6f}\n")
    print(f"Saved timing summary to {timing_path}")

    plot_results(model, device, CONFIG["arch_name_serial"])


def run_parallel_helmholtz():
    return run_serial_helmholtz()


if __name__ == "__main__":
    run_serial_helmholtz()
