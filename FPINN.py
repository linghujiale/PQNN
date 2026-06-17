"""
经典 PINN 版本（傅里叶特征输入版）：
用于求解二维 Poisson 方程，保留纯 MLP 结构，
并将每个输入变量扩展为：
[x, sin(x), cos(x), sin(3x), cos(3x), sin(5x), cos(5x)]。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

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
# 0. 所有参数统一定义在代码最前面的 def 中
# ============================================================
def get_config() -> Dict[str, Any]:
    return {
        # ---------------- 基本设置 ----------------
        "default_dtype": torch.float32,
        "save_dir": "figures",
        "domain_min": -1.0,
        "domain_max": 1.0,
        "classical_input_dim": 2,
        "model_output_dim": 1,

        # ---------------- 模型名称/输出 ----------------
        "model_name": "MLP Fourier Features Input (2D Poisson)",
        "model_filename": "poisson_mlp_fourier_features.pt",
        "plot_filename_prefix": "poisson_mlp_fourier_features_MLP",

        # ---------------- Fourier 特征设置 ----------------
        # 每个变量展开为：
        # [x, sin(1*x), cos(1*x), sin(3*x), cos(3*x), sin(5*x), cos(5*x)]
        "fourier_freqs": [1.0, 3.0, 5.0],
        "include_raw_input": True,

        # ---------------- MLP 结构 ----------------
        "hidden_dims": [64, 64, 64, 64],

        # ---------------- Poisson 精确解 ----------------
        # 计算域扩展为 [-1,1]×[-1,1] 后，为保持总波动数不变，
        # 将空间波数减半：
        # u(x,y) = sin((lamd*pi/2)*x) * sin((lamd*pi/2)*y)
        # -Δu = 2*(lamd*pi/2)^2 * u

        # ---------------- 训练采样参数 ----------------
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
        "lbfgs_tolerance_grad": 1e-10,
        "lbfgs_tolerance_change": 1e-12,
        "lbfgs_history_size": 20,
        "lbfgs_line_search_fn": "strong_wolfe",
        "lbfgs_epochs": 200,
        "lbfgs_log_every": 20,

        # ---------------- 可视化参数 ----------------
        "rel_l2_eps": 1e-8,
        "validation_grid_n": 100,
        "plot_grid_n": 100,
        "plot_figsize": (15, 4),
        "plot_dpi": 150,
    }

CONFIG = get_config()
lamd = get_env_float("LAMD", 6.0)
CONFIG["adam_steps"] = get_env_int("ADAM_STEPS", CONFIG["adam_steps"])
CONFIG["lbfgs_epochs"] = get_env_int("LBFGS_EPOCHS", CONFIG["lbfgs_epochs"])
CONFIG["lbfgs_max_iter"] = get_env_int("LBFGS_MAX_ITER", CONFIG["lbfgs_max_iter"])
CONFIG["log_every"] = get_env_int("LOG_EVERY", CONFIG["log_every"])
CONFIG["lbfgs_log_every"] = get_env_int("LBFGS_LOG_EVERY", CONFIG["lbfgs_log_every"])

# 自动计算 Fourier 展开后的输入维度
def get_feature_dim() -> int:
    per_var_dim = 0
    if CONFIG["include_raw_input"]:
        per_var_dim += 1
    per_var_dim += 2 * len(CONFIG["fourier_freqs"])  # sin/cos
    return CONFIG["classical_input_dim"] * per_var_dim


CONFIG["input_feature_dim"] = get_feature_dim()

# stdout 行缓冲
sys.stdout.reconfigure(line_buffering=True)

# 默认使用 64 位浮点数，保证 PINN 二阶导数稳定
torch.set_default_dtype(CONFIG["default_dtype"])

os.makedirs(CONFIG["save_dir"], exist_ok=True)
LAMD_TAG = build_lamd_tag(lamd)
RUN_SAVE_DIR = os.path.join(CONFIG["save_dir"], f"lamd_{LAMD_TAG}")
os.makedirs(RUN_SAVE_DIR, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================
# 1. Fourier 特征映射
# ============================================================
def build_fourier_features(x: torch.Tensor) -> torch.Tensor:
    """
    输入:
        x: [N, D]
    输出:
        feats: [N, D * (1 + 2 * len(freqs))]  (若 include_raw_input=True)

    对每个变量 xi 展开为：
        [xi, sin(1*xi), cos(1*xi), sin(3*xi), cos(3*xi), sin(5*xi), cos(5*xi)]
    """
    feat_list: List[torch.Tensor] = []

    for d in range(x.shape[1]):
        xd = x[:, d:d + 1]  # [N,1]

        if CONFIG["include_raw_input"]:
            feat_list.append(xd)

        for w in CONFIG["fourier_freqs"]:
            feat_list.append(torch.sin(w * xd))
            feat_list.append(torch.cos(w * xd))

    return torch.cat(feat_list, dim=1)


# ============================================================
# 2. 传统 MLP 网络（Fourier 特征输入）
# ============================================================
class PoissonMLP(nn.Module):
    """用于二维 Poisson PINN 的纯 MLP（Fourier 特征输入版本）。"""

    def __init__(
        self,
        hidden_dims: Sequence[int] = tuple(CONFIG["hidden_dims"]),
        device=None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = CONFIG["default_dtype"]

        self.input_scale = nn.Parameter(
            torch.ones(CONFIG["input_feature_dim"], dtype=self.dtype, device=self.device)
        )

        dims = [CONFIG["input_feature_dim"], *hidden_dims, CONFIG["model_output_dim"]]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, dtype=self.dtype))
            if out_dim != CONFIG["model_output_dim"]:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers).to(self.device)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.dtype)
        feats = build_fourier_features(x)
        feats = feats * self.input_scale
        y = self.net(feats)
        return y.squeeze(-1)


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

    input_scale_params = count_tensor_parameters(model.input_scale)
    net_params = count_parameters(model.net, trainable_only=False)
    total_params = input_scale_params + net_params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'raw input dim':<32s}: {CONFIG['classical_input_dim']:>10d}")
    print(f"{'feature input dim':<32s}: {CONFIG['input_feature_dim']:>10d}")
    print(f"{'input_scale':<32s}: {input_scale_params:>10d}")
    print(f"{'net':<32s}: {net_params:>10d}")
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
# 2. Poisson 问题
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

    u_x = grad_u[:, 0]
    u_y = grad_u[:, 1]

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


# ============================================================
# 3. 可视化
# ============================================================
def plot_results(model: nn.Module, device: torch.device, arch_name: str):
    if plt is None or np is None:
        return

    n = CONFIG["plot_grid_n"]
    xs = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])
    ys = torch.linspace(CONFIG["domain_min"], CONFIG["domain_max"], n, dtype=CONFIG["default_dtype"])

    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).view(
        -1, CONFIG["classical_input_dim"]
    ).to(device)

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
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        pred_np,
        origin="lower",
        extent=[CONFIG["domain_min"], CONFIG["domain_max"], CONFIG["domain_min"], CONFIG["domain_max"]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"{arch_name} Prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_np,
        origin="lower",
        extent=[CONFIG["domain_min"], CONFIG["domain_max"], CONFIG["domain_min"], CONFIG["domain_max"]],
        cmap="Reds",
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Poisson {arch_name} (Rel L2: {rel_l2.item():.2e})")
    plt.tight_layout()

    prefix = CONFIG["plot_filename_prefix"]
    png_path = os.path.join(RUN_SAVE_DIR, f"{prefix}.png")
    pdf_path = os.path.join(RUN_SAVE_DIR, f"{prefix}.pdf")
    plt.savefig(png_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {png_path}")
    print(f"Saved plot to {pdf_path}")


# ============================================================
# 4. 训练流程
# ============================================================
def run_helmholtz():
    print("========== 当前运行 [纯 MLP + Fourier 特征输入版：2D Poisson] ==========\n")
    device = get_device()
    print(f"Using device: {device}")
    print(f"lamd = {lamd:.1f}")
    print(f"Output dir = {RUN_SAVE_DIR}")
    print(f"Adam epochs = {CONFIG['adam_steps']}, LBFGS epochs = {CONFIG['lbfgs_epochs']}")
    print(f"Fourier freqs = {CONFIG['fourier_freqs']}")
    print(f"Input feature dim = {CONFIG['input_feature_dim']}")
    print(
        f"Validation grid = {CONFIG['validation_grid_n']} x {CONFIG['validation_grid_n']} "
        f"({CONFIG['validation_grid_n'] ** 2} points)"
    )

    model = PoissonMLP(hidden_dims=CONFIG["hidden_dims"], device=device).to(device)
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
        t_bnd = gen_points(CONFIG["num_bnd_per_edge"], 1).squeeze(-1)
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
    train_start_time = time.perf_counter()

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

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step % CONFIG["log_every"] == 0:
            with torch.no_grad():
                u_pred = model(x_int)
                u_true = u_exact(x_int)
                rel_l2 = torch.linalg.norm(u_pred - u_true) / (
                    torch.linalg.norm(u_true) + CONFIG["rel_l2_eps"]
                )
            adam_stage_elapsed = time.perf_counter() - adam_start_time

            print(
                f"Step {step:04d} | "
                f"Loss: {loss.item():.4e} | "
                f"Rel L2: {rel_l2.item():.4e} | "
                f"input_scale_mean: {model.input_scale.mean().item():.4f} | "
                f"elapsed: {adam_stage_elapsed:.2f}s ({format_elapsed_time(adam_stage_elapsed)})"
            )
    adam_elapsed_time = time.perf_counter() - adam_start_time
    print(
        f"Adam training time ({CONFIG['adam_steps']} epochs) = {adam_elapsed_time:.2f} s "
        f"({format_elapsed_time(adam_elapsed_time)})"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=CONFIG["lbfgs_lr"],
        max_iter=CONFIG["lbfgs_max_iter"],
        tolerance_grad=CONFIG["lbfgs_tolerance_grad"],
        tolerance_change=CONFIG["lbfgs_tolerance_change"],
        history_size=CONFIG["lbfgs_history_size"],
        line_search_fn=CONFIG["lbfgs_line_search_fn"],
    )

    def closure():
        lbfgs_opt.zero_grad()
        loss = torch.mean(pde_residual(model, x_int) ** 2) + CONFIG["boundary_loss_weight"] * torch.mean(
            (model(x_bnd) - u_exact(x_bnd)) ** 2
        )
        loss.backward()
        return loss

    print("Starting L-BFGS fine-tuning...", flush=True)
    lbfgs_start_time = time.perf_counter()
    for epoch in range(1, CONFIG["lbfgs_epochs"] + 1):
        loss = lbfgs_opt.step(closure)

        if epoch % CONFIG["lbfgs_log_every"] == 0:
            with torch.no_grad():
                u_pred = model(x_int)
                u_true = u_exact(x_int)
                rel_l2 = torch.linalg.norm(u_pred - u_true) / (
                    torch.linalg.norm(u_true) + CONFIG["rel_l2_eps"]
                )
            lbfgs_stage_elapsed = time.perf_counter() - lbfgs_start_time

            print(
                f"[LBFGS] epoch {epoch:03d} | "
                f"loss={loss.item():.4e} | "
                f"rel_l2={rel_l2.item():.4e} | "
                f"elapsed={lbfgs_stage_elapsed:.2f}s ({format_elapsed_time(lbfgs_stage_elapsed)})",
                flush=True,
            )
    lbfgs_elapsed_time = time.perf_counter() - lbfgs_start_time
    print(
        f"LBFGS training time ({CONFIG['lbfgs_epochs']} epochs) = {lbfgs_elapsed_time:.2f} s "
        f"({format_elapsed_time(lbfgs_elapsed_time)})"
    )

    with torch.no_grad():
        u_pred = model(x_val)
        u_true = u_val
        val_rel_l2 = torch.linalg.norm(u_pred - u_true) / (torch.linalg.norm(u_true) + CONFIG["rel_l2_eps"])

    total_training_time = time.perf_counter() - train_start_time
    print(f"Best training loss = {best_loss:.6e}")
    print(f"Learned input_scale = {model.input_scale.detach().cpu().numpy()}")
    print(
        f"Total training time = {total_training_time:.2f} s "
        f"({format_elapsed_time(total_training_time)})"
    )
    print(
        f"[Timing Summary] lamd={lamd:.1f} | "
        f"Adam({CONFIG['adam_steps']})={adam_elapsed_time:.2f}s | "
        f"LBFGS({CONFIG['lbfgs_epochs']})={lbfgs_elapsed_time:.2f}s"
    )
    print("\n========== Validation (100x100 Uniform Grid) ==========")
    print(f"Validation Rel L2: {val_rel_l2.item():.4e}")

    model_path = os.path.join(RUN_SAVE_DIR, CONFIG["model_filename"])
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    timing_filename = f"{os.path.splitext(CONFIG['model_filename'])[0]}_timing_summary.txt"
    timing_path = os.path.join(RUN_SAVE_DIR, timing_filename)
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(f"lamd={lamd:.1f}\n")
        f.write(f"adam_epochs={CONFIG['adam_steps']}\n")
        f.write(f"lbfgs_epochs={CONFIG['lbfgs_epochs']}\n")
        f.write(f"adam_seconds={adam_elapsed_time:.6f}\n")
        f.write(f"lbfgs_seconds={lbfgs_elapsed_time:.6f}\n")
        f.write(f"total_seconds={total_training_time:.6f}\n")
    print(f"Saved timing summary to {timing_path}")

    plot_results(model, device, CONFIG["model_name"])


if __name__ == "__main__":
    run_helmholtz()
