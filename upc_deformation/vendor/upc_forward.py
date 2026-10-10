"""
physics/upc_forward.py
======================

VMD + Glauber UPC 矢量介子（J/ψ）相干产生的可微 torch 前向模型。

特征
----
- **完全 torch 化**：所有 numpy/scipy 算子换成 torch；FFT 用 torch.fft，
  RegularGridInterpolator 换成 ``F.grid_sample``，``scipy.special.k1`` 换成
  自实现的稳定 K1。
- **Batched θ**：``forward(cluster_pos, sigma_c, y, ...)``,
  ``cluster_pos: (B, n_clusters, 3)``, ``sigma_c: (B,) 或 scalar``,
  ``y: (B,) 或 scalar``。一个 batch 元素对应一组 4-cluster 构型 + 一个 rapidity。
- **非对称碰撞支持**：可通过 ``cluster_pos_2`` 指定第二个核的独立团簇构型，
  配合配置中 ``nuclear_1`` / ``nuclear_2`` 节点实现 p-Pb、d-Au 等非对称系统。
- **WS + alpha_cluster 混合密度模型**：每个核可独立选择 ``woods_saxon`` 或
  ``alpha_cluster`` 密度模型，实现如 p-Pb（核1=WS 质子，核2=alpha_cluster Pb）
  等混合系统。
- **团簇密度**：默认走 **Brink-Bloch 反对称化** 单体密度（含 Pauli 交换修正），
  可通过 ``brink_antisym=False`` 退化到 naive Gaussian-sum 做对照。
- **device/dtype 自动跟随输入**：grid / 物理常数 全部 ``register_buffer``。
- **保留 incoherent 开关**：``incoherent=False`` 走纯相干；``incoherent=True``
  会按相同的物理模型在 image_q_total 上叠加一份 Gaussian 包络（积分用 batched
  torch quadrature 替代原 MC，整个路径可微）。

输出
----
- ``image_q_total: (B, target_size_q, target_size_q)``  动量空间 J/ψ 分布
- ``p_axis_q:      (target_size_q,)``                  q_perp 轴 (GeV)

主要入口
--------
- ``forward(cluster_pos, sigma_c, y, cluster_pos_2=None, density_model_1=None, ...)``
    单一取向的相干前向。底层 / 给出明确取向时直接用。
    当 ``cluster_pos_2=None`` 时退化为对称 AA 碰撞。
    当 ``density_model_n='woods_saxon'`` 时忽略该核的 cluster_pos。
- ``forward_with_orientation_avg(cluster_pos, n_orientations, chunk_size, ...)``
    在给定 cluster_pos 上做 SO(3) Euler-角随机平均。物理上 UPC 出态原子核取向
    无法逐事件标定，必须做平均。**chunk_size 用来防止 B·N 过大时 OOM。**
- ``forward_from_zmatrix(z, n_orientations, ...)``
    Phase 2 主入口：给定 6 维 SE(3) 不变量（z-matrix）→ canonical cartesian →
    旋转平均 → image_avg。全程可微。配合 ``ZMatrixBijection`` 就是
    ``NF 输出 u → image_avg`` 的完整 pipeline。

与 numpy 原版的对应
-----------------
原 ``upc.__call__`` 的 ``output_momentum_space=True, polar=True`` 分支被烧死为
默认行为，只额外提供 ``incoherent`` flag。代码顺序与原版相同：

1. _compute_density_gaussian_sum   → ρ_p, ρ_total            (B, N, N, N)
2. _compute_F_3d                   → F̂(k) form factor         (B, N, N, N) complex
3. _compute_T_prim_and_thickness   → T'(b), T(b)              (B, Nt, Nt)
4. _compute_photon_field           → V_x(b), V_y(b)           (B, No, No) complex
5. _compute_A_matrices             → A1_x, A1_y, A2_x, A2_y    (B, No, No) complex
6. _coherent_image_q               → image_q_total            (B, T, T)
7. (可选) _incoherent_contribution → 加 Gaussian 包络

设计 trade-off
--------------
- 这一版**不在 ``__init__`` 里**缓存密度/F/光子场，所有依赖 cluster_pos 的中间
  量都在 ``forward`` 里 batched 重算。代价是单 batch 内存：
  e.g. B=32, N_WS=64, complex64 → 32 × 64³ × 8 byte ≈ 67 MB（可接受）。
- 1D / 2D / 3D 的 ``grid_sample`` 实现统一封装在文件顶部，单独单元测试就能验证。

"""

from __future__ import annotations

import math
import os
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ============================================================
# 工具：torch.meshgrid 兼容老版本 (torch < 1.10 不接受 indexing= kwarg)
# ============================================================

def _meshgrid_ij(*tensors: torch.Tensor):
    """
    与 ``np.meshgrid(..., indexing='ij')`` / ``torch.meshgrid(..., indexing='ij')``
    数值等价的 wrapper，兼容 torch < 1.10（那时没有 ``indexing`` kwarg，但默认行为
    本身就是 'ij'）。
    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # torch < 1.10：默认行为等价于 indexing='ij'
        return torch.meshgrid(*tensors)


def _complex_dtype_for(real_or_complex: torch.Tensor) -> torch.dtype:
    """Preserve float64 accumulation when the caller supplies float64 density.

    The historical O16 path is float32 and therefore remains complex64
    byte-for-byte.  The Ne20 analytic-density oracle may deliberately supply
    float64; forcing it back to complex64 would discard the requested stable
    accumulation before the first FFT.
    """
    dtype = real_or_complex.real.dtype if real_or_complex.is_complex() else real_or_complex.dtype
    return torch.complex128 if dtype == torch.float64 else torch.complex64


# ============================================================
# 工具：冲量参数 b 窗口积分
# ============================================================

IMPACT_PARAMETER_QUADRATURE_CONTRACT = "normalized_trapezoid_2pi_b_db_v1"
IMPACT_PARAMETER_SURVIVAL_CONTRACT = "unity_within_hard_b_window"
FIXED_B_COMPLEX_AMPLITUDE_CONTRACT = "fixed_b_complex_amplitude_v1"
FIXED_B_EXTERNAL_DENSITY_BRANCH_RESPONSE_CONTRACT = (
    "fixed_b_external_density_branch_response_v1"
)


class FixedBExternalDensityBranchResponse(NamedTuple):
    """Pre-square branch amplitudes plus the legacy empirical comparison.

    Complex amplitudes use public axis order ``(batch,b,y,p_x,p_y)`` on the
    full FFT grid.  The empirical branch images use the same leading axes on
    the frozen output grid and are already intensities.  No configuration
    average or impact-parameter quadrature is applied.
    """

    amplitude_2_to_1_x: torch.Tensor
    amplitude_2_to_1_y: torch.Tensor
    amplitude_1_to_2_x: torch.Tensor
    amplitude_1_to_2_y: torch.Tensor
    empirical_incoherent_2_to_1: torch.Tensor
    empirical_incoherent_1_to_2: torch.Tensor
    p_full_GeV: torch.Tensor
    p_axis_GeV: torch.Tensor


def _normalized_impact_parameter_quadrature(
    b_grid: Sequence[float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """返回 ``∫ 2π b db`` 的归一化复合梯形节点权重。

    这里计算的是 hard-window *average*::

        <I>_b = ∫[b_min,b_max] 2π b I(b) db / ∫[b_min,b_max] 2π b db

    而不是带 ``fm²`` 面积因子的绝对截面。这保持历史 ML 图像尺度，
    同时把旧的离散 ``2πb_i / Σ2πb_i`` 点权重改成真正包含
    ``db`` 的 quadrature。

    单点网格定义为 fixed-b evaluation，权重为 1。多点网格必须有限、
    ``b >= 0`` 且严格递增；不会静默排序或合并重复节点。
    """
    b_values = tuple(float(b) for b in b_grid)
    if not b_values:
        raise ValueError("impact-parameter quadrature requires at least one b node")
    if any(not math.isfinite(b) for b in b_values):
        raise ValueError("impact-parameter nodes must be finite")
    if any(b < 0.0 for b in b_values):
        raise ValueError("impact-parameter nodes must satisfy b >= 0")
    if len(b_values) == 1:
        return b_values, (1.0,)
    if any(right <= left for left, right in zip(b_values[:-1], b_values[1:])):
        raise ValueError("multi-node impact-parameter grid must be strictly increasing")

    # Composite-trapezoid node widths for a possibly non-uniform grid.
    node_widths = [0.5 * (b_values[1] - b_values[0])]
    node_widths.extend(
        0.5 * (b_values[i + 1] - b_values[i - 1])
        for i in range(1, len(b_values) - 1)
    )
    node_widths.append(0.5 * (b_values[-1] - b_values[-2]))
    raw_weights = tuple(
        2.0 * math.pi * b * width for b, width in zip(b_values, node_widths)
    )
    weight_sum = math.fsum(raw_weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("impact-parameter quadrature has zero/non-finite annular measure")
    return b_values, tuple(weight / weight_sum for weight in raw_weights)


# ============================================================
# 工具：稳定的 K_1（modified Bessel function 二类一阶）
# ============================================================

def stable_k1(x: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 K_1(x) (modified Bessel function K, order 1)，
    用于光子等效流。当 x→0+, K_1(x) ~ 1/x；当 x→∞, K_1(x) ~ sqrt(π/(2x)) e^{-x}。

    实现：
      - x < 1: 用 series（K_1(x) ≈ 1/x + (x/2) ln(x/2) + …）
      - x ≥ 1: 用 asymptotic + Chebyshev 改良

    这里直接调用 torch.special.modified_bessel_k1（torch ≥ 2.0 已支持）。
    若 torch 不支持，可以 fallback 到分段实现。本文件不直接用 K1（光子流走 FFT
    路径，不显式 K1），保留此函数以便未来扩展。
    """
    return torch.special.modified_bessel_k1(x.clamp_min(1e-12))


# ============================================================
# 工具：grid_sample 包装
# ============================================================
#
# torch.nn.functional.grid_sample 的坐标系：
#   - 输入: (N, C, H, W)（2D）或 (N, C, D, H, W)（3D）
#   - grid: 归一化坐标 ∈ [-1, 1]
#       2D grid shape (N, H_out, W_out, 2)，最后一维顺序是 (x→W, y→H)
#       3D grid shape (N, D_out, H_out, W_out, 3)，最后一维顺序是 (x→W, y→H, z→D)
#   - 网格上 *单元中心*：align_corners=True 时 [-1, 1] 对齐到 [0, W-1]/[0, H-1]/[0, D-1]
# 我们在所有需要插值的地方都用 align_corners=True，并预先把数据按"单调升序"排好。
# ----------------------------------------------------------------


def _interp2d_batched(
    values: torch.Tensor,         # (B, H, W) 或 (B, H, W) complex
    grid_x: torch.Tensor,         # (B, Q, 2) 物理坐标 (x, y)
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> torch.Tensor:
    """
    2D 双线性插值（batched）。

    Args
    ----
    values : (B, H, W) real 或 complex；axis 1=H(=x), axis 2=W(=y)
             按 *单调升序* 摆放：values[b, j, i] 对应 (x_j, y_i)
    grid_x : (B, Q, 2)  查询的物理坐标，最后一维是 (x, y)
    x_min, x_max : x 轴端点
    y_min, y_max : y 轴端点

    Returns
    -------
    (B, Q) 同 values 的 dtype

    Notes
    -----
    grid_sample 2D 约定: last-dim (x, y) → (W, H)。
    values 的 H 维度对应物理 x，W 维度对应物理 y。
    因此 grid[..., 0] (→ W) 放 y 坐标，grid[..., 1] (→ H) 放 x 坐标。
    代码中通过 ``torch.stack([ny, nx], dim=-1)`` 实现此映射。
    """
    B, H, W = values.shape[0], values.shape[1], values.shape[2]
    Q = grid_x.shape[1]

    # 物理坐标 → 归一化 [-1, 1]
    nx = 2.0 * (grid_x[..., 0] - x_min) / (x_max - x_min) - 1.0  # (B, Q)
    ny = 2.0 * (grid_x[..., 1] - y_min) / (y_max - y_min) - 1.0  # (B, Q)

    # grid_sample 2D 约定: last-dim (x, y) 对应 (W, H)。
    # values 摆放为 (B, H=x_idx, W=y_idx) → unsqueeze(1) 后 (B, C, H=x_idx, W=y_idx)。
    # 所以 grid x (→W=y_idx) 应该是 *物理 y*，grid y (→H=x_idx) 应该是 *物理 x*。
    grid = torch.stack([ny, nx], dim=-1).to(values.real.dtype).unsqueeze(1)  # (B, 1, Q, 2)

    is_complex = values.is_complex()
    if is_complex:
        re = values.real.unsqueeze(1)  # (B, 1, H, W)
        im = values.imag.unsqueeze(1)
        out_re = F.grid_sample(re, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
        out_im = F.grid_sample(im, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
        out = torch.complex(out_re, out_im).squeeze(1).squeeze(1)  # (B, Q)
    else:
        v = values.unsqueeze(1)
        out = F.grid_sample(v, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
        out = out.squeeze(1).squeeze(1)
    return out


def _interp3d_F_at_kz(
    F_3d_sorted: torch.Tensor,   # (B, Nx, Ny, Nz) complex，按 *单调升序* 排好
    kz_value: torch.Tensor,      # (B,)  目标 kz 标量（每个 batch 一个）
    kx_query: torch.Tensor,      # (Q_y, Q_x) 2D 查询的 kx 网格 (单调升序)
    ky_query: torch.Tensor,      # (Q_y, Q_x) 2D 查询的 ky 网格 (单调升序)
    kx_min: float, kx_max: float,
    ky_min: float, ky_max: float,
    kz_min: float, kz_max: float,
) -> torch.Tensor:
    """
    在 batched 3D form factor 上做 "kz 固定 + (kx, ky) 2D 网格" 插值。

    输出 shape: (B, Q_y, Q_x) complex。

    Notes
    -----
    F_3d_sorted 维度顺序为 (B, Nx, Ny, Nz)，对应 grid_sample 3D 的
    (N, C, D, H, W) 中 D→Nx(kx), H→Ny(ky), W→Nz(kz)。
    grid_sample 3D 约定: (x, y, z) → (W, H, D)，因此：
      - grid[..., 0] → W = Nz = kz
      - grid[..., 1] → H = Ny = ky
      - grid[..., 2] → D = Nx = kx
    """
    B = F_3d_sorted.shape[0]
    Qy, Qx = kx_query.shape

    # 归一化坐标
    nx = 2.0 * (kx_query - kx_min) / (kx_max - kx_min) - 1.0  # (Qy, Qx)
    ny = 2.0 * (ky_query - ky_min) / (ky_max - ky_min) - 1.0
    nz = 2.0 * (kz_value - kz_min) / (kz_max - kz_min) - 1.0  # (B,)

    # F_3d_sorted 摆放 (B, Nx, Ny, Nz)，unsqueeze(1) 后
    # (B, C, D=Nx, H=Ny, W=Nz)。grid_sample 3D 约定 (x,y,z)→(W,H,D)。
    # → grid x (W=Nz) = 归一化 kz_value
    #   grid y (H=Ny) = 归一化 ky_query
    #   grid z (D=Nx) = 归一化 kx_query
    grid = torch.empty(B, 1, Qy, Qx, 3, device=F_3d_sorted.device,
                       dtype=F_3d_sorted.real.dtype)
    grid[..., 0] = nz[:, None, None, None].expand(B, 1, Qy, Qx)   # kz → W
    grid[..., 1] = ny[None, None, :, :].expand(B, 1, Qy, Qx)      # ky → H
    grid[..., 2] = nx[None, None, :, :].expand(B, 1, Qy, Qx)      # kx → D

    # F.grid_sample 不支持 complex → real/imag 分开
    re = F_3d_sorted.real.unsqueeze(1)  # (B, 1, Nx, Ny, Nz)
    im = F_3d_sorted.imag.unsqueeze(1)
    out_re = F.grid_sample(re, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    out_im = F.grid_sample(im, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    out = torch.complex(out_re.squeeze(1).squeeze(1),
                        out_im.squeeze(1).squeeze(1))   # (B, Qy, Qx)
    return out


# ============================================================
# 主类
# ============================================================


class UPCForwardModel(nn.Module):
    """
    VMD + Glauber UPC 矢量介子（J/ψ）相干产生的可微 batched torch 前向模型。

    构造时只需要 config（grid / 物理常数 / 碰撞 / 核结构 等静态参数）。
    cluster_pos / sigma_c / y 等"θ"层面的输入都在 ``forward`` 时传入。

    **非对称碰撞**：通过 ``cluster_pos_2`` 参数指定第二个核的独立构型，
    并在配置中提供 ``nuclear_1`` / ``nuclear_2`` 节点定义各核的 Z、A。
    当 ``cluster_pos_2=None`` 时退化为对称 AA 碰撞。

    **混合密度模型**：每个核可独立配置 ``density_model`` 为 ``woods_saxon``
    或 ``alpha_cluster``。当 ``density_model='woods_saxon'`` 时，
    ``cluster_pos`` 参数对该核被忽略，密度由 WS 形变参数 + 取向决定。
    这使得我们可以实现如 p-Pb（核1=WS 质子，核2=alpha_cluster Pb）等混合系统。
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, config: Union[str, dict]):
        super().__init__()
        if isinstance(config, str):
            self._load_config_file(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config 必须是 str 或 dict，收到 {type(config)}")

        self._setup_parameters()
        self._setup_grids()
        self._precompute_static_ws_intermediates()

    # ------------------- 配置加载 -------------------

    def _load_config_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件 {path} 不存在")
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def _setup_parameters(self):
        # 碰撞参数
        coll = self.config['collision_parameters']
        self.b_imp = float(coll['b'])
        # 默认 rapidity；forward 时若传入 y 则覆盖
        self.default_y = float(coll.get('y', 0.0))

        if 'gamma_1' in coll and 'gamma_2' in coll:
            self.gamma_1 = float(coll['gamma_1'])
            self.gamma_2 = float(coll['gamma_2'])
        else:
            g = float(coll['gamma'])
            self.gamma_1 = g
            self.gamma_2 = g

        # 取向向量（用于 WS 模型）
        orient = self.config.get('orientation_vectors', {})
        self.v_vec_1 = orient.get('v_vec_1', [0.0, 0.0, 1.0])
        self.v_vec_2 = orient.get('v_vec_2', [0.0, 0.0, 1.0])

        # 核结构 — 全局默认值 + 各核独立覆盖
        nuc_global = self.config.get('nuclear_parameters', {})
        Z_global = int(nuc_global.get('Z', 8))
        A_global = int(nuc_global.get('A', 16))

        # 全局 WS 参数（默认值，可被各核覆盖）
        R_ws_p_global = float(nuc_global.get('R_ws_p', 6.62))
        R_ws_n_global = float(nuc_global.get('R_ws_n', 6.62))
        d_ws_p_global = float(nuc_global.get('d_ws_p', 0.546))
        d_ws_n_global = float(nuc_global.get('d_ws_n', 0.546))
        beta2_global = float(nuc_global.get('beta2', 0.0))
        beta3_global = float(nuc_global.get('beta3', 0.0))

        self._nuclear_configs = {}

        for nid in [1, 2]:
            key = f'nuclear_{nid}'
            if key in self.config:
                cfg_src = self.config[key]
            else:
                cfg_src = {}

            # 密度模型
            dm = cfg_src.get('density_model',
                             nuc_global.get('density_model', 'alpha_cluster')).strip().lower()

            cfg = {
                'density_model': dm,
                'Z': int(cfg_src.get('Z', Z_global)),
                'A': int(cfg_src.get('A', A_global)),
                'N': int(cfg_src.get('N',
                                     int(cfg_src.get('A', A_global)) - int(cfg_src.get('Z', Z_global)))),
                # WS 参数（各核可独立覆盖）
                'R_ws_p': float(cfg_src.get('R_ws_p', R_ws_p_global)),
                'R_ws_n': float(cfg_src.get('R_ws_n', R_ws_n_global)),
                'd_ws_p': float(cfg_src.get('d_ws_p', d_ws_p_global)),
                'd_ws_n': float(cfg_src.get('d_ws_n', d_ws_n_global)),
                'beta2': float(cfg_src.get('beta2', beta2_global)),
                'beta3': float(cfg_src.get('beta3', beta3_global)),
                # 取向（WS 模型用；alpha_cluster 时从 cluster_pos 推断）
                'v_vec': cfg_src.get('v_vec', self.v_vec_1 if nid == 1 else self.v_vec_2),
            }

            self._nuclear_configs[nid] = cfg
            setattr(self, f'_density_model_{nid}', dm)
            setattr(self, f'Z_{nid}', cfg['Z'])
            setattr(self, f'A_{nid}', cfg['A'])
            setattr(self, f'N_neut_{nid}', cfg['N'])

        # 向后兼容：self.Z / self.A 指向核 1
        self.Z = self.Z_1
        self.A = self.A_1
        self.N_neut = self.N_neut_1

        # 物理常数
        cst = self.config['physical_constants']
        self.hbar_c = float(cst['hbar_c'])
        self.Mjpsi = float(cst['Mjpsi'])
        self.alpha = float(cst['alpha'])
        self.mass_proton = float(cst.get('mass_proton', 0.938272))

        # 非相干截面相关参数
        inc = self.config.get('incoherent_parameters', {})
        self.bv_Jpsi = float(inc.get('bv_Jpsi', 4.0))
        self.vmphotoncoupling_Jpsi = float(inc.get('vmphotoncoupling_Jpsi', 0.0))
        self.NQ = float(inc.get('NQ', 1.0))

        # ── γp→Vp 弹性截面参数化 ──
        #   'jpsi_powerlaw' (默认, = calculate_Jpsi_spe.cxx:190 的 80.2 幂律拟合, 向后兼容)
        #   'two_component' (= PRC 60 014903 Table II Klein-Nystrand 双分量, 用于 ρ⁰/ω/φ)
        self.gp_parametrization = str(inc.get('gp_parametrization', 'jpsi_powerlaw'))
        self.vm_X = float(inc.get('vm_X', 0.0))                 # μb  (Table II, Pomeron)
        self.vm_epsilon = float(inc.get('vm_epsilon', 0.0))
        self.vm_Y = float(inc.get('vm_Y', 0.0))                 # μb  (Reggeon 项)
        self.vm_eta = float(inc.get('vm_eta', 0.0))
        self.vm_sigma_unit = float(inc.get('vm_sigma_unit', 1.0e-4))  # μb→fm²
        # ── Söding Breit-Wigner (PRC 60 014903 Eq.17) — 仅 two_component (ρ⁰) 有意义 ──
        self.vm_width = float(inc.get('vm_width', 0.0))         # GeV  Γ_V
        self.bw_A = float(inc.get('bw_A', 0.0))                 # μb^0.5
        self.bw_B = float(inc.get('bw_B', 0.0))
        self.bw_ppi0 = float(inc.get('bw_ppi0', 0.358))        # GeV
        self.m_chargedpion = float(inc.get('m_chargedpion', 0.13957))
        self.vm_mass_integration = bool(inc.get('vm_mass_integration', False))
        if self.vm_mass_integration:
            raise NotImplementedError(
                "vm_mass_integration=True 尚未接入 forward（当前一律按极点质量 Mjpsi 成像）。"
                "ρ⁰ 的 π⁺π⁻ 质量积分是下一步；见 configs/upc_forward_rho0_O16O16.yaml 注释。")

        # 默认 sigma_c
        self.default_sigma_c = float(self.config.get('cluster_parameters', {}).get('sigma_c', 1.0))

        # 各核质子能量
        self.Ep_1 = self.mass_proton * self.gamma_1
        self.Ep_2 = self.mass_proton * self.gamma_2

        # 网格参数
        self.grid_params = self.config['grid_parameters']

        # 末端动量轴
        out_settings = self.config.get('output_settings', {})
        self.p_limit_q = float(out_settings.get('p_limit_q', 0.5))
        self.target_size_q = int(out_settings.get('target_size_q', 512))
        self.safe_padding_factor = int(out_settings.get('safe_padding_factor', 8))
        self.b_slope_incoh = float(out_settings.get('b_slope_incoh', 4.0))

        # 非相干 quadrature 网格
        inc_quad = self.config.get('incoherent_quadrature', {})
        self.incoh_r_max = float(inc_quad.get('r_max', 20.0))
        self.incoh_n_r = int(inc_quad.get('n_r', 64))
        self.incoh_n_phi = int(inc_quad.get('n_phi', 64))

    # ------------------- 网格 buffer -------------------

    def _setup_grids(self):
        """
        所有网格都注册成 buffer，自动跟随 device/dtype。

        约定 grid_parameters 至少包含 'WS', 'T_prim', 'output' 三个 key。
        - WS  : 3D（density + form factor，假设两者共网格，与原 numpy 默认一致）
        - T_prim : 2D（厚度函数）
        - output : 2D（最终 amplitudes，FFT padding 在此基础上）
        """
        for grid_name in ['WS', 'T_prim', 'output']:
            if grid_name not in self.grid_params:
                raise KeyError(f"grid_parameters 中缺少 '{grid_name}'")

            params = self.grid_params[grid_name]
            N = int(params['N'])
            L = float(params['L'])
            dx = L / N

            # x_grid：[-L/2, L/2 - dx]，与原 numpy np.linspace(endpoint=False) 对齐
            x_grid = torch.arange(N, dtype=torch.float32) * dx - L / 2.0
            # k_grid：fftfreq(N, dx) * 2π，原 numpy 一致
            k_raw = torch.fft.fftfreq(N, d=dx) * (2 * math.pi)
            # sort 后的 k_grid（用于 grid_sample 插值）
            k_sorted, _ = torch.sort(k_raw)

            self.register_buffer(f'_{grid_name}_x_grid', x_grid)
            self.register_buffer(f'_{grid_name}_k_raw', k_raw)
            self.register_buffer(f'_{grid_name}_k_sorted', k_sorted)
            # Python scalar interpolation bounds are immutable grid metadata.
            # Cache them while grids are still constructed on CPU instead of
            # calling ``Tensor.item()`` inside every CUDA forward chunk.
            setattr(self, f'_{grid_name}_k_min_float', float(k_sorted[0]))
            setattr(self, f'_{grid_name}_k_max_float', float(k_sorted[-1]))
            setattr(self, f'_{grid_name}_N', N)
            setattr(self, f'_{grid_name}_L', L)
            setattr(self, f'_{grid_name}_dx', dx)

        # 2D 网格 (X, Y) for output (impact-parameter integration)
        # 与原 numpy: X_grid, Y_grid = np.meshgrid(x_grid, x_grid, indexing='ij')
        # 即 X_grid[i,j] = x_grid[i], Y_grid[i,j] = x_grid[j]
        x_out = getattr(self, '_output_x_grid')
        X_out, Y_out = _meshgrid_ij(x_out, x_out)
        self.register_buffer('_output_X', X_out)
        self.register_buffer('_output_Y', Y_out)

        # kx, ky for output grid (k-space) - 同样规则
        k_out = getattr(self, '_output_k_raw')
        kx_out, ky_out = _meshgrid_ij(k_out, k_out)
        self.register_buffer('_output_kx', kx_out)
        self.register_buffer('_output_ky', ky_out)

        # 末端动量轴（q 空间）
        target_points = self._output_N * self.safe_padding_factor
        dx_out = self._output_dx
        p_full_q = torch.fft.fftshift(
            torch.fft.fftfreq(target_points, d=dx_out)
        ) * (2 * math.pi * self.hbar_c)
        self.register_buffer('_p_full_q', p_full_q)
        self._p_full_q_min_float = float(p_full_q[0])
        self._p_full_q_max_float = float(p_full_q[-1])
        self._target_points = target_points

        # 用于末端 image_q_total 插值的目标轴 (-p_limit_q, p_limit_q)
        p_axis_q = torch.linspace(-self.p_limit_q, self.p_limit_q, self.target_size_q,
                                  dtype=torch.float32)
        self.register_buffer('_p_axis_q', p_axis_q)
        p_qx, p_qy = _meshgrid_ij(p_axis_q, p_axis_q)
        self.register_buffer(
            '_p_axis_q_interp_coords',
            torch.stack([p_qx, p_qy], dim=-1).reshape(-1, 2),
        )
        self.register_buffer('_p_axis_q_radius2', p_qx * p_qx + p_qy * p_qy)

        # impact parameter offset (b/2 along x): 用作 (B,Q,2) 偏移 broadcast
        # 直接用 float 即可，不需要 buffer

    # ------------------------------------------------------------------
    # 球形 WS 核：rho / F_3d 静态预计算（O(B·N) → O(1)）
    # ------------------------------------------------------------------

    def _is_spherical_ws(self, nuclear_ID: int) -> bool:
        """β₂ = β₃ = 0 的 woods_saxon 密度旋转不变。"""
        cfg = self._nuclear_configs[nuclear_ID]
        return (
            cfg['density_model'] == 'woods_saxon'
            and float(cfg['beta2']) == 0.0
            and float(cfg['beta3']) == 0.0
        )

    def _precompute_static_ws_intermediates(self):
        """
        对每个 β₂=β₃=0 的 woods_saxon 核，把 *与 rapidity 无关* 的中间量
        (rho_p, rho_total, F_3d) 预先算出来 register_buffer。这些张量在训练
        全程都是常量（不依赖 y，不依赖 cluster_pos，不依赖任何取向 R），
        没必要每个 batch 重算。

        ``forward_with_orientation_avg`` 的 α + spherical-WS 快速路径会直接
        消费这些 buffer，把 Pb 端 compute 从 O(B·N) 降到 O(B)；普通 forward
        的 ``_compute_density_ws`` 也会优先走 cache 命中分支。

        预计算保存为：
            self._static_ws_rho_p_<k>        (1, N_grid, N_grid, N_grid)
            self._static_ws_rho_total_<k>    (1, N_grid, N_grid, N_grid)
            self._static_ws_F_3d_<k>         (1, N_grid, N_grid, N_grid) complex
            self._has_static_ws_<k>          (bool, attribute)
        """
        for nid in [1, 2]:
            has_static = self._is_spherical_ws(nid)
            setattr(self, f'_has_static_ws_{nid}', has_static)
            if not has_static:
                continue
            cfg = self._nuclear_configs[nid]
            rho_p, rho_total = self._compute_ws_radial_density(cfg)
            F_3d = self._compute_F_3d(rho_p)
            self.register_buffer(f'_static_ws_rho_p_{nid}', rho_p)
            self.register_buffer(f'_static_ws_rho_total_{nid}', rho_total)
            self.register_buffer(f'_static_ws_F_3d_{nid}', F_3d)

    def _compute_ws_radial_density(self, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        球形 WS 密度（β=0），径向只算一次，不做任何旋转矩阵 einsum。
        返回 (1, N, N, N) tensor，可被 expand 到任意 batch。
        """
        Z_nuc = int(cfg['Z'])
        N_neut = int(cfg['N'])
        R_ws_p = float(cfg['R_ws_p'])
        R_ws_n = float(cfg['R_ws_n'])
        d_ws_p = float(cfg['d_ws_p'])
        d_ws_n = float(cfg['d_ws_n'])

        N = self._WS_N
        dx = self._WS_dx
        x_grid = self._WS_x_grid                          # (N,)
        device = x_grid.device
        dtype = x_grid.dtype

        x = x_grid.view(N, 1, 1)
        y = x_grid.view(1, N, 1)
        z = x_grid.view(1, 1, N)
        r_sq = x * x + y * y + z * z                      # (N, N, N) — 标量场
        r_vals = torch.sqrt(r_sq.clamp_min(1e-20))

        dist_p = torch.clamp((r_vals - R_ws_p) / d_ws_p, -50.0, 50.0)
        ws_p = 1.0 / (1.0 + torch.exp(dist_p))
        integral_p = ws_p.sum() * (dx ** 3)
        rho0_p = Z_nuc / integral_p.clamp_min(1e-20)
        rho_p = (ws_p * rho0_p).unsqueeze(0)              # (1, N, N, N)

        dist_n = torch.clamp((r_vals - R_ws_n) / d_ws_n, -50.0, 50.0)
        ws_n = 1.0 / (1.0 + torch.exp(dist_n))
        integral_n = ws_n.sum() * (dx ** 3)
        rho0_n = N_neut / integral_n.clamp_min(1e-20)
        ws_n_norm = ws_n * rho0_n                          # (N, N, N)

        rho_total = (rho_p.squeeze(0) + ws_n_norm).unsqueeze(0)  # (1, N, N, N)
        return rho_p.to(dtype=dtype), rho_total.to(dtype=dtype)

    # ------------------------------------------------------------------
    # 辅助：取向向量 → 旋转矩阵（与 numpy 版 transfer_v_vec + 旋转矩阵一致）
    # ------------------------------------------------------------------

    def _vvec_to_rotation_matrix(
        self,
        v_vec,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        将取向向量转化为旋转矩阵（体坐标系 → 实验室系）。

        与 numpy 版 ``transfer_v_vec`` + 旋转矩阵构造完全一致。

        Returns
        -------
        R : (3, 3)  rotation matrix
        """
        x, y, z = v_vec
        r = math.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r

        cos_theta = z
        sin_theta = math.sqrt(max(1.0 - z * z, 0.0))
        if sin_theta < 1e-12:
            sin_phi = 0.0
            cos_phi = 1.0
        else:
            sin_phi = y / sin_theta
            cos_phi = x / sin_theta

        Ry = torch.tensor([
            [cos_theta, 0.0, -sin_theta],
            [0.0, 1.0, 0.0],
            [sin_theta, 0.0, cos_theta],
        ], device=device, dtype=dtype)

        Rz1 = torch.tensor([
            [cos_phi, sin_phi, 0.0],
            [-sin_phi, cos_phi, 0.0],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)

        neg_Rz = torch.tensor([
            [cos_phi, -sin_phi, 0.0],
            [sin_phi, cos_phi, 0.0],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)

        R = neg_Rz @ Ry @ Rz1
        return R

    # ------------------------------------------------------------------
    # 物理子模块
    # ------------------------------------------------------------------

    # ---------- Woods-Saxon 密度 ----------

    def _compute_density_ws(
        self,
        nuclear_ID: int,
        R: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Woods-Saxon 形变核密度（含 β₂, β₃ 形变），在实验室系中计算。

        物理上 WS 密度在**体坐标系**中定义（对称轴沿 z 方向）。
        实验室系中的密度通过在体坐标系中计算后得到：

            ρ_lab(r) = ρ_body(R · r)

        其中 R 是由 ``transfer_v_vec(v_vec)`` 在 numpy 版构造、与 numpy 路径
        ``rot_coords = layer_coords @ R^T`` 等价（对行向量作用时 ``R^T`` 右乘
        等同 ``R`` 左乘）的旋转矩阵。下面 einsum 的实现完全镜像 numpy。
        当 ``R=None`` 时，使用配置中的 ``v_vec`` 构造 R。

        Args
        ----
        nuclear_ID : 1 或 2
        R : (B, 3, 3) 旋转矩阵。None 时使用配置中的 v_vec。
        batch_size : 仅在 ``R=None`` 时使用；让返回密度的 batch 维度与外层 forward
                     一致 (否则会硬设为 1，导致下游 qL/y batch broadcast 报错)。
                     默认 1，行为退化为旧版。

        Returns
        -------
        rho_p     : (B, N, N, N)
        rho_total : (B, N, N, N)
        """
        cfg = self._nuclear_configs[nuclear_ID]
        Z_nuc = cfg['Z']
        A_nuc = cfg['A']
        N_neut = cfg['N']

        R_ws_p = cfg['R_ws_p']
        R_ws_n = cfg['R_ws_n']
        d_ws_p = cfg['d_ws_p']
        d_ws_n = cfg['d_ws_n']
        beta2 = cfg['beta2']
        beta3 = cfg['beta3']

        N = self._WS_N
        L = self._WS_L
        dx = self._WS_dx

        # ── 球形快速路径：β=0 时密度旋转不变，直接用 __init__ 预计算的 buffer ──
        # 与下面的旋转代码完全等价（β=0 时 deformation 恒为 1）；省一次 einsum
        # + sigmoid + sum + 归一化。
        if getattr(self, f'_has_static_ws_{nuclear_ID}', False):
            rho_p_one = getattr(self, f'_static_ws_rho_p_{nuclear_ID}')
            rho_total_one = getattr(self, f'_static_ws_rho_total_{nuclear_ID}')
            if R is not None:
                B_out = R.shape[0]
            else:
                B_out = int(batch_size) if batch_size is not None else 1
            # .expand 是 view (零拷贝)；下游若需 contiguous (FFT 等) 会自动 copy 一次
            return (
                rho_p_one.expand(B_out, -1, -1, -1),
                rho_total_one.expand(B_out, -1, -1, -1),
            )

        # 确定 device / dtype / batch
        if R is not None:
            device = R.device
            dtype = R.dtype
            B = R.shape[0]
        else:
            # 使用 v_vec 构造旋转矩阵
            v_vec = cfg['v_vec']
            # 从 register_buffer 推断 device/dtype（确保与模型一致）
            device = self._WS_x_grid.device
            dtype = self._WS_x_grid.dtype
            R_mat = self._vvec_to_rotation_matrix(v_vec, device, dtype)
            # 把固定取向 broadcast 到外层 batch，避免下游 qL/y(B,) 与
            # rho_total(1,N,N,N) reshape 冲突。
            B = int(batch_size) if batch_size is not None else 1
            R = R_mat.unsqueeze(0).expand(B, 3, 3).contiguous()

        x_grid = self._WS_x_grid.to(device=device, dtype=dtype)

        # 构造实验室系 3D 网格
        x = x_grid.view(N, 1, 1)
        y = x_grid.view(1, N, 1)
        z = x_grid.view(1, 1, N)

        coords_lab = torch.stack([
            x.expand(N, N, N),
            y.expand(N, N, N),
            z.expand(N, N, N),
        ], dim=-1).unsqueeze(0).expand(B, N, N, N, 3)  # (B, N, N, N, 3)

        # 实验室系 → 体坐标系：r_body[c] = Σ_m R[c, m] · r_lab[m] = (R · r_lab)[c]
        # 与 numpy `rot_coords = layer_coords @ R^T`（即 R 左乘行向量）严格一致。
        coords_body = torch.einsum('bijkm,bcm->bijkc', coords_lab, R)  # (B, N, N, N, 3)

        rx = coords_body[..., 0]
        ry = coords_body[..., 1]
        rz = coords_body[..., 2]

        r_sq = rx**2 + ry**2 + rz**2
        r_vals = torch.sqrt(r_sq.clamp_min(1e-20))

        # cos(theta) = z / r
        cos_theta = torch.clamp(rz / (r_vals + 1e-12), -1.0, 1.0)

        # 球谐函数 Y_2^0 和 Y_3^0
        Y20 = 0.31539156525252005 * (3.0 * cos_theta**2 - 1.0)
        Y30 = 0.3731763325901154 * cos_theta * (5.0 * cos_theta**2 - 3.0)

        deformation = 1.0 + beta2 * Y20 + beta3 * Y30

        # 质子 WS 分布
        R_def_p = R_ws_p * deformation
        dist_p = (r_vals - R_def_p) / d_ws_p
        dist_p = torch.clamp(dist_p, -50.0, 50.0)
        ws_p = 1.0 / (1.0 + torch.exp(dist_p))

        # 中子 WS 分布
        R_def_n = R_ws_n * deformation
        dist_n = (r_vals - R_def_n) / d_ws_n
        dist_n = torch.clamp(dist_n, -50.0, 50.0)
        ws_n = 1.0 / (1.0 + torch.exp(dist_n))

        # 归一化
        vol = dx ** 3
        integral_p = ws_p.sum(dim=(1, 2, 3), keepdim=True) * vol
        rho0_p = Z_nuc / integral_p.clamp_min(1e-20)
        rho_p = ws_p * rho0_p

        integral_n = ws_n.sum(dim=(1, 2, 3), keepdim=True) * vol
        rho0_n = N_neut / integral_n.clamp_min(1e-20)
        ws_n_norm = ws_n * rho0_n

        rho_total = rho_p + ws_n_norm

        return rho_p, rho_total

    # ---------- 团簇高斯密度 (naive，不含反对称化) ----------

    def _compute_density_gaussian_sum(
        self,
        cluster_pos: torch.Tensor,    # (B, n_clusters, 3)
        sigma_c: torch.Tensor,        # (B,)
        nuclear_ID: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把 cluster 看成 *独立* 高斯波包叠加 → 核子总密度 / 质子密度。
        **不做** Brink-Bloch 反对称化（Pauli 交换修正），适合做参考 baseline。

        每个 cluster 假设包含 A/n_clusters 个核子（其中 Z/n_clusters 个是质子）：
            ρ_total(r) = sum_i N_i · gauss(r - r_i; σ_c)
            ρ_p(r)     = sum_i Z_i · gauss(r - r_i; σ_c)
        其中 N_i = A / n_clusters, Z_i = Z / n_clusters。

        最后对全网格积分归一化，保证 ∑ ρ_p · dx³ = Z_nuc, ∑ ρ_total · dx³ = A_nuc。

        实现：**可分离 Gaussian** —— exp(-(x²+y²+z²)/2σ²) = exp(-x²/2σ²)·…
        把 5D 张量 (B, n_cl, N, N, N) 中间量替换成 3 个 (B, n_cl, N) 1D 因子 +
        一次 einsum。FLOPs 不变（B·n_cl·N³），但中间内存从 ~130 MB 降到几 KB，
        Python 调度开销也消失。生产配置下 (B=32, n_cl=4, N=64) 实测 ~2-3× 加速。

        Args
        ----
        nuclear_ID : 1 或 2，从 ``self._nuclear_configs[nuclear_ID]`` 取 (Z, A)。
                     非对称碰撞时必须显式指定，否则核 2 会被错误归一化为核 1 的 (Z, A)。

        Returns
        -------
        rho_p     : (B, N, N, N)
        rho_total : (B, N, N, N)
        """
        cfg = self._nuclear_configs[nuclear_ID]
        Z_nuc = cfg['Z']
        A_nuc = cfg['A']

        B, n_clusters, _ = cluster_pos.shape
        device = cluster_pos.device
        dtype = cluster_pos.dtype

        N = self._WS_N
        dx = self._WS_dx
        x_grid = self._WS_x_grid.to(device=device, dtype=dtype)  # (N,)

        # 可分离 Gaussian：g_x[b,p,x] = exp(-(x_grid - cx[b,p])² / (2σ²))
        # cluster_pos[:, :, axis] : (B, n_cl)
        inv_2s2 = (1.0 / (2.0 * (sigma_c.to(dtype) ** 2))).view(B, 1, 1)  # (B,1,1)
        # x 方向
        x_diff = x_grid.view(1, 1, N) - cluster_pos[:, :, 0:1]  # (B, n_cl, N)
        gx = torch.exp(-x_diff * x_diff * inv_2s2)
        # y 方向
        y_diff = x_grid.view(1, 1, N) - cluster_pos[:, :, 1:2]
        gy = torch.exp(-y_diff * y_diff * inv_2s2)
        # z 方向
        z_diff = x_grid.view(1, 1, N) - cluster_pos[:, :, 2:3]
        gz = torch.exp(-z_diff * z_diff * inv_2s2)

        # rho_shape[b, x, y, z] = sum_p gx[b,p,x] * gy[b,p,y] * gz[b,p,z]
        # einsum 自动选最优 contraction 顺序，比手写更省内存
        rho_shape = torch.einsum('bpx,bpy,bpz->bxyz', gx, gy, gz)

        # 归一化系数 ( (2πσ²)^(-3/2) ) — 与求和后归一化合并掉
        # 直接以"积分=Z_nuc"重新缩放即可（数值上等价）
        vol = dx ** 3
        integral = rho_shape.sum(dim=(1, 2, 3), keepdim=True) * vol
        integral = integral.clamp_min(1e-20)

        rho_total = rho_shape * (A_nuc / integral)
        rho_p = rho_shape * (Z_nuc / integral)

        return rho_p, rho_total

    # ---------- 团簇密度：Brink-Bloch 反对称化版本 ----------

    def _compute_density_brink(
        self,
        cluster_pos: torch.Tensor,    # (B, n_clusters, 3)
        sigma_c: torch.Tensor,        # (B,)
        nuclear_ID: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Brink-Bloch 反对称化的 α-cluster 单体密度。

        物理推导（Slater 行列式 + 4 重自旋同位旋）：

            ρ_total(r) = 4 · ∑_{ij} M_{ij} · g(r - R_{ij}; b)

        其中
            R_{ij} = (R_i + R_j) / 2                                       (B, n_cl, n_cl, 3)
            S_{ij} = ⟨φ_i|φ_j⟩ = exp(-(R_i - R_j)² / (4b²))                (B, n_cl, n_cl)
            M_{ij} = [S⁻¹]_{ij} · S_{ij}        (Hadamard / element-wise)  (B, n_cl, n_cl)
            g(r; b) = (πb²)^(-3/2) exp(-r²/b²)

        振子长度与密度 σ 的关系：σ_density = b/√2  ⇔  b² = 2σ²。
        外部接口 ``sigma_c`` 沿用密度 σ 约定，与 ``_compute_density_gaussian_sum``
        保持一致。

        - 对角项 M_{ii} 含交换贡献，一般 < 1（被 cluster 间 Pauli blocking 推低）
        - 非对角项 M_{i≠j} 是反对称化带来的 "交换 Gaussian"，中心在 (R_i+R_j)/2
        - cluster 完全远离时 S → I, M → I, 回到 naive Gaussian-sum

        对 16O (n_cl=4 个 α-cluster): A = 4·n_cl = 16, ∫ρ_total · dr = A 闭式成立。

        实现 trick：用 loop-over-pairs 累加，单次只占 O(B·N³) 显存。
        ``n_cl=4`` 时只有 16 次累加，与 FFT 相比可忽略。

        Args
        ----
        nuclear_ID : 1 或 2，从 ``self._nuclear_configs[nuclear_ID]`` 取 (Z, A)。
                     非对称碰撞时必须显式指定，否则核 2 会被错误归一化为核 1 的 (Z, A)。
        """
        cfg = self._nuclear_configs[nuclear_ID]
        Z_nuc = cfg['Z']
        A_nuc = cfg['A']

        B, n_cl, _ = cluster_pos.shape
        device = cluster_pos.device
        dtype = cluster_pos.dtype

        N = self._WS_N
        dx = self._WS_dx
        x_grid = self._WS_x_grid.to(device=device, dtype=dtype)

        sig2 = sigma_c.to(dtype) ** 2
        b2 = 2.0 * sig2
        b2_mat = b2.view(B, 1, 1)
        b2_vol = b2.view(B, 1, 1, 1)

        R_diff = cluster_pos.unsqueeze(2) - cluster_pos.unsqueeze(1)
        R_diff_sq = (R_diff ** 2).sum(dim=-1)
        S = torch.exp(-R_diff_sq / (4.0 * b2_mat))

        # ★ 鲁棒求逆: S 是 α-cluster 单体高斯的重叠 (Gram) 矩阵, 对称半正定。
        #   当两个 α 中心靠得很近时 (dv/dist path-B 可能抽到 |R_i-R_j|→0, 或采样的
        #   6 条距离违反 Cayley-Menger、经 dv_to_cartesian 的 EPS-clamp 退化成近共点),
        #   S 趋于"全 1 矩阵" → 奇异。旧实现 `solve(S + 1e-8·I)` 此时会抛 LinAlgError
        #   (GPU) 或给出爆量级的 S_inv (CPU)。
        #   改用对称特征分解 + 相对特征值下限 (rcond) 求逆:
        #     - 良态 S: 所有 λ > rcond·λ_max, 不触发 floor → 与真逆一致 (误差 ~1e-15);
        #     - 病态/奇异 S: 把过小/为零的特征值抬到 rcond·λ_max, 得到有界的
        #       Tikhonov/截断伪逆, 不再崩溃, M 保持有限。
        #   eigh 比 solve 多一点开销, 但 n_cl=4 极小, 相对 FFT 可忽略。
        S_sym = 0.5 * (S + S.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(S_sym)                       # (B, n_cl), (B, n_cl, n_cl)
        lam_max = evals[..., -1:].clamp_min(1.0e-12)                  # (B, 1) 最大特征值
        lam_floored = torch.maximum(evals, 1.0e-6 * lam_max)         # 相对下限 (rcond=1e-6)
        S_inv = torch.matmul(
            evecs * (1.0 / lam_floored).unsqueeze(-2),
            evecs.transpose(-1, -2),
        )                                                             # Q diag(1/λ) Qᵀ

        M = S_inv * S       # (B, n_cl, n_cl)

        norm = (math.pi * b2_vol) ** (-1.5)
        inv_b2 = (1.0 / b2_vol).view(B, 1, 1)        # (B, 1, 1)

        # ── 可分离 Gaussian 重写 ──
        # 原 nested loop:
        #   rho(x,y,z) = 4·norm · Σ_{ij} M_{ij} · exp(-((x-x_ij)²+(y-y_ij)²+(z-z_ij)²)/b²)
        # 用 e^{-(a²+b²+c²)} = e^{-a²}·e^{-b²}·e^{-c²} 拆开，所有中心 R_ij 一起处理：
        #   centers = 0.5·(R_i + R_j) ∈ (B, P=n_cl², 3)
        #   g_axis[b,p,x] = exp(-(x_grid - centers[b,p,axis])² / b²)
        #   rho[b,x,y,z] = 4·norm · Σ_p M_flat[b,p] · gx[b,p,x]·gy[b,p,y]·gz[b,p,z]
        # einsum 'bp,bpx,bpy,bpz->bxyz' 把 4D 输出写成一次 contraction，
        # 不再开 (B, n_cl², N³) 临时张量；FLOPs 同 loop 但 Python 调度 / 内存碰撞消失。
        P = n_cl * n_cl
        centers = 0.5 * (cluster_pos.unsqueeze(2) + cluster_pos.unsqueeze(1))  # (B, n_cl, n_cl, 3)
        centers = centers.reshape(B, P, 3)
        M_flat = M.reshape(B, P)

        # 1D Gaussian 因子 — exp(-(x_grid - c_axis)² / b²)
        x_diff = x_grid.view(1, 1, N) - centers[:, :, 0:1]   # (B, P, N)
        gx = torch.exp(-x_diff * x_diff * inv_b2)
        y_diff = x_grid.view(1, 1, N) - centers[:, :, 1:2]
        gy = torch.exp(-y_diff * y_diff * inv_b2)
        z_diff = x_grid.view(1, 1, N) - centers[:, :, 2:3]
        gz = torch.exp(-z_diff * z_diff * inv_b2)

        # 把 M 折进 gx (省一次 einsum 维度)
        gx_w = M_flat.unsqueeze(-1) * gx     # (B, P, N)

        rho_total = torch.einsum('bpx,bpy,bpz->bxyz', gx_w, gy, gz)
        rho_total = rho_total * (4.0 * norm.view(B, 1, 1, 1))

        vol = dx ** 3
        integral = rho_total.sum(dim=(1, 2, 3), keepdim=True) * vol
        integral = integral.clamp_min(1e-20)
        rho_total = rho_total * (A_nuc / integral)
        rho_p = rho_total * (Z_nuc / A_nuc)

        return rho_p, rho_total

    # ---------- 厚度函数 T(b) 与含相位的 T'(b) ----------

    def _compute_T_prim_and_thickness(
        self,
        rho_total: torch.Tensor,   # (B, N, N, N)
        qL: torch.Tensor,          # (B,)
        Z_nuc: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 batched 厚度函数 T(b) = ∫ dz ρ(b, z) 以及含相位的
        T'(b) = ∫ dz ρ(b, z) exp(i qL z)。

        归一化：T(b) → 缩放使 ∫ T db² = Z_nuc。

        Args
        ----
        Z_nuc : 该核的质子数。None 时退化为 self.Z_1。

        Returns
        -------
        T_prim_complex : (B, N, N) complex
        T_thickness    : (B, N, N) real
        """
        B, N, _, _ = rho_total.shape
        device = rho_total.device
        dtype = rho_total.dtype

        if Z_nuc is None:
            Z_nuc = self.Z_1

        x_grid = self._WS_x_grid.to(device=device, dtype=dtype)
        dx = self._WS_dx

        complex_dtype = _complex_dtype_for(rho_total)
        phase = torch.exp(1j * qL.view(B, 1, 1, 1).to(complex_dtype) *
                          x_grid.view(1, 1, 1, N).to(complex_dtype))

        rho_c = rho_total.to(complex_dtype)
        T_prim = (rho_c * phase).sum(dim=-1) * dx
        T_thick = rho_total.sum(dim=-1) * dx

        dx_T = self._T_prim_dx
        integral = T_thick.sum(dim=(1, 2), keepdim=True) * dx_T * dx_T
        integral = integral.clamp_min(1e-20)
        scale = Z_nuc / integral

        T_thick = T_thick * scale
        T_prim = T_prim * scale.to(complex_dtype)

        assert self._WS_N == self._T_prim_N and self._WS_L == self._T_prim_L, \
            "torch 版默认 WS 与 T_prim 共网格；若需要不同 N/L，请在此处加 grid_sample 插值。"

        return T_prim, T_thick

    # ---------- 形状因子 F̂(k) ----------

    def _compute_F_3d(self, rho_p: torch.Tensor) -> torch.Tensor:
        """
        Form factor F̂(k) = ∫ dr ρ_p(r) e^{-i k·r}。

        Returns
        -------
        F_sorted : (B, N, N, N) complex，按"k 单调升序"排列（已经做 fftshift）。
                   并对 F(0) 归一化到 1。
        """
        B = rho_p.shape[0]
        N = self._WS_N
        L = self._WS_L

        rho_c = rho_p.to(_complex_dtype_for(rho_p))
        rho_centered = torch.fft.ifftshift(rho_c, dim=(-3, -2, -1))
        F_field = torch.fft.ifftn(rho_centered, dim=(-3, -2, -1)) * (L ** 3)

        F_sorted = torch.fft.fftshift(F_field, dim=(-3, -2, -1))

        c = N // 2
        F0 = F_sorted[:, c, c, c].view(B, 1, 1, 1)
        F0 = torch.where(F0.abs() < 1e-20, torch.ones_like(F0), F0)
        F_sorted = F_sorted / F0

        return F_sorted

    # ---------- 等效光子流 V_x, V_y（在 b-空间） ----------

    def _compute_photon_field(
        self,
        F_3d_sorted: torch.Tensor,    # (B, Nx, Ny, Nz) complex
        gamma_n: float,
        kz_value: torch.Tensor,       # (B,)
        Z_nuc: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        等效光子流的两个分量 V_x(b), V_y(b)。

        与原 numpy `_precompute_photon_matrices` 一致：
            G_x(k_perp) = k_x · F̂(k_perp, kz) / |k|²
            G_y(k_perp) = k_y · F̂(k_perp, kz) / |k|²
            V_x(b) = sqrt(4α γ Z²/kz) · fftshift(fft2(G_x)) / L²
            V_y(b) = sqrt(4α γ Z²/kz) · fftshift(fft2(G_y)) / L²

        Args
        ----
        Z_nuc : 该核的质子数。None 时退化为 self.Z_1。

        Returns
        -------
        Vx, Vy : (B, N_out, N_out) complex
        """
        B = F_3d_sorted.shape[0]
        device = F_3d_sorted.device

        if Z_nuc is None:
            Z_nuc = self.Z_1

        kx_out = self._output_kx.to(device)
        ky_out = self._output_ky.to(device)
        No = self._output_N
        L_out = self._output_L

        kx_min = self._WS_k_min_float
        kx_max = self._WS_k_max_float
        ky_min, ky_max = kx_min, kx_max
        kz_min, kz_max = kx_min, kx_max

        Fk = _interp3d_F_at_kz(
            F_3d_sorted, kz_value.to(device),
            kx_out, ky_out,
            kx_min, kx_max, ky_min, ky_max, kz_min, kz_max,
        )

        k_perp2 = (kx_out ** 2 + ky_out ** 2)[None, :, :]
        kz2 = (kz_value ** 2).view(B, 1, 1)
        k2 = k_perp2 + kz2

        # 当 k2=0 时结果应为 0（与 numpy 版对齐）
        k2_safe = k2.clamp_min(1e-20)

        complex_dtype = Fk.dtype
        kx_b = kx_out[None, :, :].to(complex_dtype)
        ky_b = ky_out[None, :, :].to(complex_dtype)

        common = Fk / k2_safe.to(complex_dtype)
        Gx = kx_b * common
        Gy = ky_b * common

        Vx_real = torch.fft.fftshift(torch.fft.fft2(Gx, dim=(-2, -1)),
                                     dim=(-2, -1)) / (L_out ** 2)
        Vy_real = torch.fft.fftshift(torch.fft.fft2(Gy, dim=(-2, -1)),
                                     dim=(-2, -1)) / (L_out ** 2)

        factor = 4.0 * self.alpha * gamma_n * (Z_nuc ** 2) / kz_value
        sqrt_factor = torch.sqrt(factor.clamp_min(0.0)).to(complex_dtype).view(B, 1, 1)

        Vx = Vx_real * sqrt_factor
        Vy = Vy_real * sqrt_factor
        return Vx, Vy

    # ---------- σ_total^VN ----------

    def _sigma_total_VN(self, Egamma: torch.Tensor, Ep: float) -> torch.Tensor:
        """σ_total^VN(γ N → V N) (fm²)。

        光学定理：σ_tot = sqrt(16π · (f_V²/4π) · ħc² · dσ/dt|0 / α) · NQ。
        dσ/dt|0 由 _dsigma_gp_Vp_overdt 给出（按 gp_parametrization 自动分 J/ψ / ρ⁰）。
        注意：本量是核内吸收强度，ρ⁰ 光学厚时 vmphotoncoupling 不再约去，必须用物理 f_V²/4π。
        """
        dsigma_dt = self._dsigma_gp_Vp_overdt(Egamma, Ep)
        sigma_total = torch.sqrt(
            (16.0 * math.pi * self.vmphotoncoupling_Jpsi *
             self.hbar_c ** 2 * dsigma_dt / max(self.alpha, 1e-20)).clamp_min(0.0)
        ) * self.NQ
        return sigma_total

    # ---------- Gamma 振幅 ----------

    def _gamma_amplitude(
        self,
        T_prim_complex: torch.Tensor,
        Egamma: torch.Tensor,
        Ep_scalar: float,
    ) -> torch.Tensor:
        """
        Γ-amplitude (complex)：
            Γ(b) = sqrt(α / (16π η) · 4π Eγ) · 2/NQ · (1 - exp(-σ T'(b)/2))
        """
        B = T_prim_complex.shape[0]

        complex_dtype = T_prim_complex.dtype
        sigma_tot = self._sigma_total_VN(Egamma, Ep_scalar).to(complex_dtype)
        sigma_tot_b = sigma_tot.view(B, 1, 1)
        # NOTE: 老版 torch (< 1.10) 对 ``1.0 - complex_tensor`` 走的 ``__rsub__``
        # 不支持复数 autograd，会在 recon 训练时报
        # "rsub does not support automatic differentiation for outputs with complex dtype"。
        # 数学等价改写成 ``-(complex_tensor - 1.0)`` (走 ``__sub__``) 规避。
        prof = -2.0 * (torch.exp(-0.5 * sigma_tot_b * T_prim_complex) - 1.0) / self.NQ

        kin = torch.sqrt(
            (self.alpha / (16.0 * math.pi * max(self.vmphotoncoupling_Jpsi, 1e-20))) *
            4.0 * math.pi * Egamma.clamp_min(0.0)
        ).to(complex_dtype).view(B, 1, 1)
        return kin * prof

    # ---------- prob ((1-T/A·crosspp)^A) ----------

    def _prob(self, T_thickness_at_source: torch.Tensor, A_nuc: Optional[int] = None) -> torch.Tensor:
        """prob(b) = (1 - T(b)/A · σ_pp)^A，crosspp=4.2 fm²。"""
        if A_nuc is None:
            A_nuc = self.A_1
        crosspp = 4.2
        return (1.0 - T_thickness_at_source / A_nuc * crosspp).clamp_min(0.0) ** A_nuc

    # ---------- A 振幅 ----------

    def _compute_amplitude_at_grid(
        self,
        nuclear_ID: int,
        T_prim_target: torch.Tensor,
        T_thick_source: torch.Tensor,
        Vx_source: torch.Tensor,
        Vy_source: torch.Tensor,
        Egamma_target: torch.Tensor,
        Ep_target: float,
        A_source: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 output 网格 (X, Y) 上求 A_x, A_y。

        b 偏移：nuclear_ID=1 时 target = r - b/2, source = r + b/2；
                nuclear_ID=2 时 target = r + b/2, source = r - b/2。
        """
        B = T_prim_target.shape[0]
        device = T_prim_target.device

        if A_source is None:
            A_source = self.A_2 if nuclear_ID == 1 else self.A_1

        X_out = self._output_X.to(device)
        Y_out = self._output_Y.to(device)
        No = self._output_N
        coords = torch.stack([X_out, Y_out], dim=-1).view(1, -1, 2)
        coords = coords.expand(B, -1, -1)

        b_half = self.b_imp / 2.0
        if nuclear_ID == 1:
            shift_target = torch.tensor([-b_half, 0.0], device=device).view(1, 1, 2)
            shift_source = torch.tensor([+b_half, 0.0], device=device).view(1, 1, 2)
        else:
            shift_target = torch.tensor([+b_half, 0.0], device=device).view(1, 1, 2)
            shift_source = torch.tensor([-b_half, 0.0], device=device).view(1, 1, 2)

        coords_target = coords + shift_target
        coords_source = coords + shift_source

        Lt = self._T_prim_L
        Nt = self._T_prim_N
        dx_t = self._T_prim_dx
        t_min = -Lt / 2.0
        t_max = Lt / 2.0 - dx_t

        # target 端
        T_prim_at_target = _interp2d_batched(
            T_prim_target, coords_target, t_min, t_max, t_min, t_max
        )
        T_prim_grid = T_prim_at_target.view(B, No, No)
        Gamma_amp = self._gamma_amp_at_points(T_prim_grid, Egamma_target, Ep_target)

        # source 端：T_thickness 和 prob
        T_thick_at_source = _interp2d_batched(
            T_thick_source, coords_source, t_min, t_max, t_min, t_max
        )
        prob_vals = self._prob(T_thick_at_source, A_nuc=A_source).view(B, No, No)

        # source 端：Vx, Vy
        Lo = self._output_L
        dx_o = self._output_dx
        o_min = -Lo / 2.0
        o_max = Lo / 2.0 - dx_o
        Vx_at_source = _interp2d_batched(
            Vx_source, coords_source, o_min, o_max, o_min, o_max
        ).view(B, No, No)
        Vy_at_source = _interp2d_batched(
            Vy_source, coords_source, o_min, o_max, o_min, o_max
        ).view(B, No, No)

        prob_c = prob_vals.to(Gamma_amp.dtype)
        scalar_part = Gamma_amp * prob_c
        Ax = Vx_at_source * scalar_part
        Ay = Vy_at_source * scalar_part
        return Ax, Ay

    def _gamma_amp_at_points(
        self,
        T_prim_complex: torch.Tensor,
        Egamma: torch.Tensor,
        Ep_scalar: float,
    ) -> torch.Tensor:
        """同 _gamma_amplitude，接受任意空间形状。"""
        return self._gamma_amplitude(T_prim_complex, Egamma, Ep_scalar)

    # ---------- 末端：A → image_q_total ----------

    def _coherent_amplitude_q(
        self,
        Ax: torch.Tensor,
        Ay: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """padding + 2D FFT → output-grid complex amplitudes.

        This is the versioned pre-square primitive required by empirical-cloud
        Good--Walker aggregation.  It performs no configuration average, no
        modulus square and no classical impact-parameter quadrature.
        """
        B = Ax.shape[0]
        device = Ax.device

        No = self._output_N
        target_points = self._target_points
        dx_out = self._output_dx

        pad_x = torch.zeros(B, target_points, target_points,
                            dtype=Ax.dtype, device=device)
        pad_y = torch.zeros_like(pad_x)
        start = (target_points - No) // 2
        end = start + No
        pad_x[:, start:end, start:end] = Ax
        pad_y[:, start:end, start:end] = Ay

        Mx_raw = torch.fft.fftshift(torch.fft.fft2(pad_x, dim=(-2, -1)),
                                    dim=(-2, -1))
        My_raw = torch.fft.fftshift(torch.fft.fft2(pad_y, dim=(-2, -1)),
                                    dim=(-2, -1))

        Mx = Mx_raw * (dx_out ** 2) / self.hbar_c
        My = My_raw * (dx_out ** 2) / self.hbar_c

        return Mx, My, self._p_full_q.to(device)

    def _coherent_image_q(
        self,
        Ax: torch.Tensor,
        Ay: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex amplitude → modulus square on the output momentum grid."""
        Mx, My, p_full = self._coherent_amplitude_q(Ax, Ay)
        prob_q = (Mx.abs() ** 2 + My.abs() ** 2) / (2.0 * math.pi)
        if hasattr(self, '_p_full_q_min_float'):
            q_min = self._p_full_q_min_float
            q_max = self._p_full_q_max_float
        else:  # lightweight projection-only test fixtures
            q_min = float(p_full[0])
            q_max = float(p_full[-1])
        p_axis = self._p_axis_q.to(Ax.device)
        T = self.target_size_q
        if hasattr(self, '_p_axis_q_interp_coords'):
            coords_base = self._p_axis_q_interp_coords.to(Ax.device)
        else:  # lightweight projection-only test fixtures
            p_qx, p_qy = _meshgrid_ij(p_axis, p_axis)
            coords_base = torch.stack([p_qx, p_qy], dim=-1).reshape(-1, 2)
        coords_q = coords_base.view(1, -1, 2)
        coords_q = coords_q.expand(Ax.shape[0], -1, -1)
        image_q_total_flat = _interp2d_batched(
            prob_q, coords_q, q_min, q_max, q_min, q_max
        )
        image_q_total = image_q_total_flat.view(Ax.shape[0], T, T)
        return image_q_total, p_axis

    def project_complex_amplitude_intensity_to_output(
        self,
        amplitude_x: torch.Tensor,
        amplitude_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Square full-grid amplitudes and apply the frozen output interpolation.

        ``amplitude_x/y`` must be the tensors returned by
        :meth:`fixed_b_complex_amplitude_multi_y` before their leading
        ``(batch,b,y)`` axes are aggregated.  This helper preserves the legacy
        schema-5 order: square first on the FFT grid, interpolate intensity
        second.  Interpolating amplitudes and then squaring is intentionally
        not allowed because it is a different observable.
        """
        if amplitude_x.shape != amplitude_y.shape or amplitude_x.is_complex() is False:
            raise ValueError("complex amplitude_x/y must have identical shapes")
        if amplitude_x.shape[-2:] != (self._target_points, self._target_points):
            raise ValueError(
                "complex amplitude primitive must retain the full FFT momentum grid"
            )
        prob_q = (amplitude_x.abs() ** 2 + amplitude_y.abs() ** 2) / (2.0 * math.pi)
        return self.project_full_grid_intensity_to_output(prob_q)

    def project_full_grid_intensity_to_output(
        self,
        full_grid_intensity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply only the frozen full-grid -> output-grid intensity map."""
        if full_grid_intensity.is_complex():
            raise ValueError("full_grid_intensity must be real and already squared")
        if full_grid_intensity.shape[-2:] != (self._target_points, self._target_points):
            raise ValueError("full_grid_intensity has the wrong FFT-grid shape")
        leading = full_grid_intensity.shape[:-2]
        prob_q = full_grid_intensity.reshape(
            -1, self._target_points, self._target_points,
        )
        p_full = self._p_full_q.to(full_grid_intensity.device)
        if hasattr(self, '_p_full_q_min_float'):
            q_min = self._p_full_q_min_float
            q_max = self._p_full_q_max_float
        else:  # lightweight projection-only test fixtures
            q_min = float(p_full[0])
            q_max = float(p_full[-1])
        p_axis = self._p_axis_q.to(full_grid_intensity.device)
        if hasattr(self, '_p_axis_q_interp_coords'):
            coords_base = self._p_axis_q_interp_coords.to(full_grid_intensity.device)
        else:  # lightweight projection-only test fixtures
            p_qx, p_qy = _meshgrid_ij(p_axis, p_axis)
            coords_base = torch.stack([p_qx, p_qy], dim=-1).reshape(-1, 2)
        coords = coords_base.view(1, -1, 2).expand(prob_q.shape[0], -1, -1)
        output = _interp2d_batched(prob_q, coords, q_min, q_max, q_min, q_max)
        return output.view(*leading, self.target_size_q, self.target_size_q), p_axis

    # ---------- 非相干贡献 ----------

    def _compute_incoherent_branch_probabilities(
        self,
        cluster_pos: torch.Tensor,
        sigma_c: torch.Tensor,
        y_per_batch: torch.Tensor,
        rho_total: torch.Tensor,
        T_thick_1: torch.Tensor,
        T_thick_2: torch.Tensor,
        Vx_2: torch.Tensor,
        Vy_2: torch.Tensor,
        Vx_1: torch.Tensor,
        Vy_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two legacy empirical-incoherent branch probabilities.

        Branch ``2_to_1`` has nucleus 2 as photon source and nucleus 1 as
        target; ``1_to_2`` uses the opposite assignment.  The historical
        total is their probability-level sum, so exposing the two terms does
        not alter production semantics and lets the Good--Walker gate compare
        like with like without inventing interference between distinct target
        breakup final states.
        """
        B = cluster_pos.shape[0]
        device = cluster_pos.device

        r_grid = torch.linspace(0.0, self.incoh_r_max, self.incoh_n_r, device=device)
        phi_grid = torch.linspace(0.0, 2.0 * math.pi, self.incoh_n_phi + 1, device=device)[:-1]
        Rg, Phig = _meshgrid_ij(r_grid, phi_grid)
        X = (Rg * torch.cos(Phig)).reshape(-1)
        Y = (Rg * torch.sin(Phig)).reshape(-1)

        Q = X.shape[0]
        coords_base = torch.stack([X, Y], dim=-1).unsqueeze(0).expand(B, Q, 2)

        b_half = self.b_imp / 2.0

        shift_src_v1 = torch.tensor([+b_half, 0.0], device=device).view(1, 1, 2)
        shift_tgt_v1 = torch.tensor([-b_half, 0.0], device=device).view(1, 1, 2)
        shift_src_v2 = torch.tensor([-b_half, 0.0], device=device).view(1, 1, 2)
        shift_tgt_v2 = torch.tensor([+b_half, 0.0], device=device).view(1, 1, 2)

        Lo = self._output_L
        dx_o = self._output_dx
        o_min = -Lo / 2.0
        o_max = Lo / 2.0 - dx_o
        Lt = self._T_prim_L
        dx_t = self._T_prim_dx
        t_min = -Lt / 2.0
        t_max = Lt / 2.0 - dx_t

        Egamma_v1 = 0.5 * self.Mjpsi * torch.exp(y_per_batch)
        Egamma_v2 = 0.5 * self.Mjpsi * torch.exp(-y_per_batch)

        sigma_gp_v1 = self._dsigma_gp_Vp_overdt(Egamma_v1, self.Ep_2) / self.bv_Jpsi
        sigma_gp_v2 = self._dsigma_gp_Vp_overdt(Egamma_v2, self.Ep_1) / self.bv_Jpsi

        sigma_incoh_1 = self._sigma_incoherent_total_VN(Egamma_v1, self.Ep_1)
        sigma_incoh_2 = self._sigma_incoherent_total_VN(Egamma_v2, self.Ep_2)

        # V1
        coords_src1 = coords_base + shift_src_v1
        coords_tgt1 = coords_base + shift_tgt_v1
        Vx1_at_src = _interp2d_batched(Vx_2, coords_src1, o_min, o_max, o_min, o_max)
        Vy1_at_src = _interp2d_batched(Vy_2, coords_src1, o_min, o_max, o_min, o_max)
        n_photon_v1 = (Vx1_at_src.abs() + Vy1_at_src.abs())

        T_at_tgt1 = _interp2d_batched(T_thick_1, coords_tgt1, t_min, t_max, t_min, t_max)
        prob_incoh_1 = T_at_tgt1 * torch.exp(-0.5 * sigma_incoh_1.view(B, 1) * T_at_tgt1)
        den_v1 = (Egamma_v1.view(B, 1) * n_photon_v1 * prob_incoh_1 *
                  sigma_gp_v1.view(B, 1))

        # V2
        coords_src2 = coords_base + shift_src_v2
        coords_tgt2 = coords_base + shift_tgt_v2
        Vx2_at_src = _interp2d_batched(Vx_1, coords_src2, o_min, o_max, o_min, o_max)
        Vy2_at_src = _interp2d_batched(Vy_1, coords_src2, o_min, o_max, o_min, o_max)
        n_photon_v2 = (Vx2_at_src.abs() + Vy2_at_src.abs())

        T_at_tgt2 = _interp2d_batched(T_thick_2, coords_tgt2, t_min, t_max, t_min, t_max)
        prob_incoh_2 = T_at_tgt2 * torch.exp(-0.5 * sigma_incoh_2.view(B, 1) * T_at_tgt2)
        den_v2 = (Egamma_v2.view(B, 1) * n_photon_v2 * prob_incoh_2 *
                  sigma_gp_v2.view(B, 1))

        r_w = r_grid.view(1, self.incoh_n_r, 1)
        dr = float(r_grid[1] - r_grid[0]) if self.incoh_n_r > 1 else 1.0
        dphi = 2.0 * math.pi / self.incoh_n_phi
        branch_2_to_1 = (
            den_v1.view(B, self.incoh_n_r, self.incoh_n_phi) * r_w
        ).sum(dim=(1, 2)) * dr * dphi
        branch_1_to_2 = (
            den_v2.view(B, self.incoh_n_r, self.incoh_n_phi) * r_w
        ).sum(dim=(1, 2)) * dr * dphi
        return branch_2_to_1.clamp_min(0.0), branch_1_to_2.clamp_min(0.0)

    def _compute_incoherent_total_prob(
        self,
        cluster_pos: torch.Tensor,
        sigma_c: torch.Tensor,
        y_per_batch: torch.Tensor,
        rho_total: torch.Tensor,
        T_thick_1: torch.Tensor,
        T_thick_2: torch.Tensor,
        Vx_2: torch.Tensor,
        Vy_2: torch.Tensor,
        Vx_1: torch.Tensor,
        Vy_1: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compatible sum of the two empirical branch probabilities."""

        branch_2_to_1, branch_1_to_2 = self._compute_incoherent_branch_probabilities(
            cluster_pos,
            sigma_c,
            y_per_batch,
            rho_total,
            T_thick_1,
            T_thick_2,
            Vx_2,
            Vy_2,
            Vx_1,
            Vy_1,
        )
        return branch_2_to_1 + branch_1_to_2

    def _dsigma_gp_Vp_overdt(self, Egamma: torch.Tensor, Ep: float) -> torch.Tensor:
        """dσ/dt|_{t=0}(γp→Vp) (fm²/GeV²)。"""
        Mp = self.mass_proton
        Wgp2 = (2.0 * Egamma * (Ep + math.sqrt(Ep * Ep - Mp * Mp))) + Mp * Mp
        Wgp = torch.sqrt(Wgp2.clamp_min(0.0))
        if self.gp_parametrization == 'two_component':
            # Klein-Nystrand, PRC 60 014903 Table II (ρ⁰/ω/φ):
            #   σ_int(γp→Vp) = unit·(X·W^ε + Y·W^{-η})  [fm²];  dσ/dt|0 = b_V·σ_int
            #   (与 ref/Incoherent_rho0.cpp:167/190 一致；无 J/ψ 那种近阈 term^1.5 因子)
            Wc = Wgp.clamp_min(1e-6)
            sigma_int = self.vm_sigma_unit * (
                self.vm_X * torch.pow(Wc, self.vm_epsilon)
                + self.vm_Y * torch.pow(Wc, -self.vm_eta)
            )
            return self.bv_Jpsi * sigma_int
        # 默认：J/ψ 幂律拟合 (= ref/calculate_Jpsi_spe.cxx:190)
        W_threshold = Mp + self.Mjpsi
        term = 1.0 - (W_threshold ** 2) / (Wgp ** 2 + 1e-20)
        term = term.clamp_min(0.0)
        return 4.0 * 1.0e-7 * 80.2 * term ** 1.5 * (Wgp ** 2 / 10000.0) ** 0.321

    def _sigma_incoherent_total_VN(self, Egamma: torch.Tensor, Ep: float) -> torch.Tensor:
        s = self._sigma_total_VN(Egamma, Ep)
        return s - s * s / (16.0 * math.pi * self.bv_Jpsi * self.hbar_c ** 2)

    def breit_wigner(self, W: torch.Tensor) -> torch.Tensor:
        """Söding 修正的相对论 Breit-Wigner，π⁺π⁻ 不变质量 W 的线形 |amp|²。

        来源 PRC 60 014903 Eq.(17)（= ref/Incoherent_rho0.cpp:260-283）。仅 ρ⁰
        (two_component + vm_width>0) 有意义；J/ψ 当窄共振、定质量，不用此函数。

        当前 forward 一律按极点质量 self.Mjpsi 成像；本函数供 (可选) π⁺π⁻ 质量
        积分用作权重——开关 vm_mass_integration 暂未接入（置 true 会在 __init__ 报错）。
        """
        mpi = self.m_chargedpion
        m0 = self.Mjpsi  # VM 极点质量 (ρ⁰ = 0.775)
        W = torch.as_tensor(W, dtype=torch.float64)
        ppi = torch.sqrt(((W / 2.0) ** 2 - mpi * mpi).clamp_min(0.0))
        rat = ppi / max(self.bw_ppi0, 1e-20)
        GammaPrim = self.vm_width * (m0 / W.clamp_min(1e-6)) * rat ** 3
        aa = self.bw_A * torch.sqrt((GammaPrim * m0 * W).clamp_min(0.0))
        bb = W * W - m0 * m0
        cc = m0 * GammaPrim
        denom = bb * bb + cc * cc + 1e-30
        re = aa * bb / denom + self.bw_B
        im = aa * cc / denom
        out = re * re + im * im
        return torch.where(W < 2.0 * mpi, torch.zeros_like(out), out)


    # ------------------------------------------------------------------
    # 核心 forward 实现
    # ------------------------------------------------------------------

    def forward(
        self,
        cluster_pos: Optional[torch.Tensor],          # (B, n_clusters, 3)  核1；WS时可为None
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y: Optional[Union[torch.Tensor, float]] = None,
        cluster_pos_2: Optional[torch.Tensor] = None,  # (B, n_clusters_2, 3) 核2
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,         # 'woods_saxon' 或 'alpha_cluster'
        density_model_2: Optional[str] = None,
        R_1: Optional[torch.Tensor] = None,            # (B, 3, 3) 旋转矩阵（WS/旋转平均用）
        R_2: Optional[torch.Tensor] = None,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单一取向的相干前向计算。

        Args
        ----
        cluster_pos   : (B, n_clusters, 3) [fm] — 核1 团簇位置。
                        当 ``density_model_1='woods_saxon'`` 时可传入 None。
        sigma_c       : (B,) 或 scalar [fm]，核1 团簇密度 σ
        y             : (B,) 或 scalar，矢量介子 rapidity
        cluster_pos_2 : (B, n_clusters_2, 3) — 核2 团簇位置。None=复制核1。
        sigma_c_2     : (B,) 或 scalar。None=sigma_c。
        density_model_1/2 : 'woods_saxon' 或 'alpha_cluster'。
                        None 时使用配置中的默认值。
        R_1, R_2      : (B, 3, 3) 旋转矩阵。用于 WS 模型（体坐标系→实验室系）
                        或 alpha_cluster 的旋转平均。None 时 WS 用 v_vec，
                        alpha_cluster 不旋转。
        incoherent    : 是否叠加非相干 Gaussian 包络
        brink_antisym : alpha_cluster 核1 密度模型选项
        brink_antisym_2: alpha_cluster 核2 密度模型选项。None=brink_antisym。

        Returns
        -------
        image_q_total : (B, target_size_q, target_size_q)
        p_axis_q      : (target_size_q,) [GeV]
        """
        # 确定密度模型
        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()
        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "density_model='external_density' requires "
                "forward_multi_y_from_density(); the legacy centre/WS entry point "
                "must not silently substitute a generic Brink density"
            )

        # alpha_cluster 需要 cluster_pos
        if dm_1 == 'alpha_cluster':
            if cluster_pos is None or not isinstance(cluster_pos, torch.Tensor):
                raise ValueError(
                    f"density_model_1='alpha_cluster' 需要 cluster_pos，收到 {cluster_pos}")
            B = cluster_pos.shape[0]
            device = cluster_pos.device
            dtype = cluster_pos.dtype
        elif dm_1 == 'woods_saxon':
            # 优先级：R_1 > cluster_pos_2 (tensor) > R_2 > y (tensor) → 推断 B/device/dtype
            if R_1 is not None:
                B = R_1.shape[0]
                device = R_1.device
                dtype = R_1.dtype
            elif cluster_pos_2 is not None and isinstance(cluster_pos_2, torch.Tensor):
                B = cluster_pos_2.shape[0]
                device = cluster_pos_2.device
                dtype = cluster_pos_2.dtype
            elif R_2 is not None:
                B = R_2.shape[0]
                device = R_2.device
                dtype = R_2.dtype
            elif isinstance(y, torch.Tensor) and y.dim() >= 1:
                B = y.shape[0]
                device = y.device
                # y 经常是 float32/float64，dtype 可以直接复用
                dtype = y.dtype
            else:
                raise ValueError(
                    "WS 模型需要 R_1 / cluster_pos_2 / R_2 / y(tensor) "
                    "中至少一个来推断 batch/device/dtype")
        else:
            raise ValueError(f"不支持的 density_model_1: {dm_1}")

        # 核2 默认复制核1
        if cluster_pos_2 is None and dm_2 == 'alpha_cluster':
            cluster_pos_2 = cluster_pos
        if sigma_c_2 is None:
            sigma_c_2 = sigma_c
        if brink_antisym_2 is None:
            brink_antisym_2 = brink_antisym

        # broadcast scalars — 核1
        if dm_1 == 'alpha_cluster':
            if sigma_c is None:
                sigma_c = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c, (int, float)):
                sigma_c = torch.full((B,), float(sigma_c), device=device, dtype=dtype)
            elif sigma_c.dim() == 0:
                sigma_c = sigma_c.expand(B).contiguous()

        # broadcast scalars — 核2
        if dm_2 == 'alpha_cluster':
            if sigma_c_2 is None:
                sigma_c_2 = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c_2, (int, float)):
                sigma_c_2 = torch.full((B,), float(sigma_c_2), device=device, dtype=dtype)
            elif sigma_c_2.dim() == 0:
                sigma_c_2 = sigma_c_2.expand(B).contiguous()

        if y is None:
            y = torch.full((B,), self.default_y, device=device, dtype=dtype)
        elif isinstance(y, (int, float)):
            y = torch.full((B,), float(y), device=device, dtype=dtype)
        elif y.dim() == 0:
            y = y.expand(B).contiguous()

        # ---- 1) 密度（两套，分别对应核1和核2）----
        # 核1
        if dm_1 == 'woods_saxon':
            rho_p_1, rho_total_1 = self._compute_density_ws(
                nuclear_ID=1, R=R_1, batch_size=B)
        else:  # alpha_cluster
            cp_1 = cluster_pos
            if R_1 is not None and cp_1 is not None:
                cp_1 = torch.einsum('bij,bkj->bki', R_1, cp_1)
            if brink_antisym:
                rho_p_1, rho_total_1 = self._compute_density_brink(cp_1, sigma_c, nuclear_ID=1)
            else:
                rho_p_1, rho_total_1 = self._compute_density_gaussian_sum(cp_1, sigma_c, nuclear_ID=1)

        # 核2
        if dm_2 == 'woods_saxon':
            rho_p_2, rho_total_2 = self._compute_density_ws(
                nuclear_ID=2, R=R_2, batch_size=B)
        else:  # alpha_cluster
            cp_2 = cluster_pos_2
            if R_2 is not None and cp_2 is not None:
                cp_2 = torch.einsum('bij,bkj->bki', R_2, cp_2)
            if brink_antisym_2:
                rho_p_2, rho_total_2 = self._compute_density_brink(cp_2, sigma_c_2, nuclear_ID=2)
            else:
                rho_p_2, rho_total_2 = self._compute_density_gaussian_sum(cp_2, sigma_c_2, nuclear_ID=2)

        # ---- 2) 形状因子（两套）----
        F_3d_sorted_1 = self._compute_F_3d(rho_p_1)
        F_3d_sorted_2 = self._compute_F_3d(rho_p_2)

        # ---- 3) qL & T_prim, T_thick（两套，传入各自的 Z）----
        qL_1 = self.Mjpsi * 0.5 / self.gamma_1 * torch.exp(y)
        qL_2 = self.Mjpsi * 0.5 / self.gamma_2 * torch.exp(-y)

        T_prim_1, T_thick_1 = self._compute_T_prim_and_thickness(
            rho_total_1, qL_1, Z_nuc=self.Z_1)
        T_prim_2, T_thick_2 = self._compute_T_prim_and_thickness(
            rho_total_2, qL_2, Z_nuc=self.Z_2)

        # ---- 4) 光子流 V_x, V_y ----
        Vx_1, Vy_1 = self._compute_photon_field(
            F_3d_sorted_1, self.gamma_1, qL_1, Z_nuc=self.Z_1)
        Vx_2, Vy_2 = self._compute_photon_field(
            F_3d_sorted_2, self.gamma_2, qL_2, Z_nuc=self.Z_2)

        # ---- 5) A 振幅 ----
        Egamma_1 = 0.5 * self.Mjpsi * torch.exp(y)
        Egamma_2 = 0.5 * self.Mjpsi * torch.exp(-y)

        A1_x, A1_y = self._compute_amplitude_at_grid(
            nuclear_ID=1,
            T_prim_target=T_prim_1, T_thick_source=T_thick_2,
            Vx_source=Vx_2, Vy_source=Vy_2,
            Egamma_target=Egamma_2, Ep_target=self.Ep_1,   # ★bugfix(2026-06): σ_VN 用源光子能量 Egamma_2(=V_2/flux 的能量), 对齐学长 C++ (flux 与 dsigma 同 Egamma1); 旧版用 Egamma_1 破坏了 Eγ·flux 抵消 → dσ/dy ∝ exp(2y) 过陡
            A_source=self.A_2,
        )
        A2_x, A2_y = self._compute_amplitude_at_grid(
            nuclear_ID=2,
            T_prim_target=T_prim_2, T_thick_source=T_thick_1,
            Vx_source=Vx_1, Vy_source=Vy_1,
            Egamma_target=Egamma_1, Ep_target=self.Ep_2,   # ★bugfix(2026-06): σ_VN 用源光子能量 Egamma_1(=V_1/flux 的能量), 对齐学长 C++
            A_source=self.A_1,
        )

        # ---- 6) 相干求和 ----
        Ax = A1_x + A2_x
        Ay = A1_y + A2_y

        # ---- 7) FFT → image_q_total ----
        image_q_total, p_axis_q = self._coherent_image_q(Ax, Ay)

        # ---- 8) (可选) 非相干 ----
        if incoherent:
            # 用核1的 cluster_pos 计算（若核1是 WS，则创建 dummy）
            cp_for_incoh = cluster_pos if cluster_pos is not None else torch.zeros(B, 1, 3, device=device)
            sc_for_incoh = sigma_c if sigma_c is not None else torch.full((B,), self.default_sigma_c, device=device)
            total_incoh_prob = self._compute_incoherent_total_prob(
                cp_for_incoh, sc_for_incoh, y, rho_total_1,
                T_thick_1, T_thick_2, Vx_2, Vy_2, Vx_1, Vy_1,
            )
            Pq_X, Pq_Y = _meshgrid_ij(p_axis_q, p_axis_q)
            q2 = (Pq_X ** 2 + Pq_Y ** 2)[None, :, :]
            b_slope = self.b_slope_incoh
            incoh_dist = (total_incoh_prob.view(B, 1, 1) * b_slope / math.pi *
                          torch.exp(-b_slope * q2))
            image_q_total = image_q_total + incoh_dist

        return image_q_total, p_axis_q

    # ------------------------------------------------------------------
    # 高级接口：Euler 角旋转平均
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_uniform_rotations(
        B: int, N: int,
        device: torch.device, dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        在 SO(3) 上均匀采样旋转矩阵（Shoemake 1992 quaternion 法）。

        Returns
        -------
        R : (B, N, 3, 3)  rotation matrices, det = +1
        """
        u = torch.rand(B, N, 3, device=device, dtype=dtype, generator=generator)
        u1 = u[..., 0]
        u2 = u[..., 1]
        u3 = u[..., 2]

        sqrt_1mu1 = torch.sqrt((1.0 - u1).clamp_min(0.0))
        sqrt_u1 = torch.sqrt(u1.clamp_min(0.0))
        ang2 = 2.0 * math.pi * u2
        ang3 = 2.0 * math.pi * u3

        qw = sqrt_u1 * torch.cos(ang3)
        qx = sqrt_1mu1 * torch.sin(ang2)
        qy = sqrt_1mu1 * torch.cos(ang2)
        qz = sqrt_u1 * torch.sin(ang3)

        R = torch.stack([
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ], dim=-1).reshape(B, N, 3, 3)
        return R

    def forward_with_orientation_avg(
        self,
        cluster_pos: Optional[torch.Tensor],          # (B, n_clusters, 3)  核1
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y: Optional[Union[torch.Tensor, float]] = None,
        cluster_pos_2: Optional[torch.Tensor] = None,  # (B, n_clusters_2, 3) 核2
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        n_orientations: int = 20,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SO(3) 旋转平均。

        当两核都是 alpha_cluster 时，走高效的 flatten 路径。
        当有 WS 核时，按取向循环（因为每个取向需要不同的 R）。

        Args
        ----
        cluster_pos    : (B, n_clusters, 3)  核1。WS 模型时可传入 None。
        sigma_c, y     : 同 forward()
        cluster_pos_2  : (B, n_clusters_2, 3) 核2。None=对称。
        sigma_c_2      : 核2 的 sigma_c。None=sigma_c。
        density_model_1/2 : 覆盖配置的密度模型。
        n_orientations : N，每个 batch 采样取向数。
        chunk_size     : flatten 路径下的分块大小。
        generator      : 可选 torch.Generator。

        Returns
        -------
        image_avg : (B, target_size_q, target_size_q)
        p_axis_q  : (target_size_q,) [GeV]
        """
        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()
        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "density_model='external_density' requires "
                "forward_multi_y_from_density(); no centre/WS fallback is allowed"
            )

        # 确定 device/dtype/B
        if dm_1 == 'alpha_cluster':
            if cluster_pos is None:
                raise ValueError("alpha_cluster 模式需要 cluster_pos")
            B, n_cl, _ = cluster_pos.shape
            device = cluster_pos.device
            dtype = cluster_pos.dtype
        else:
            # WS 模式：优先级 cluster_pos_2 > cluster_pos > y(tensor) 推断 batch
            if cluster_pos_2 is not None:
                B = cluster_pos_2.shape[0]
                device = cluster_pos_2.device
                dtype = cluster_pos_2.dtype
                n_cl = 0  # WS 核不需要团簇数
            elif cluster_pos is not None:
                B = cluster_pos.shape[0]
                device = cluster_pos.device
                dtype = cluster_pos.dtype
                n_cl = cluster_pos.shape[1]
            elif isinstance(y, torch.Tensor) and y.dim() >= 1:
                B = y.shape[0]
                device = y.device
                dtype = y.dtype
                n_cl = 0
            else:
                raise ValueError(
                    "WS+WS 模式需要 cluster_pos / cluster_pos_2 / y(tensor) "
                    "至少一个来推断 batch/device/dtype")

        # 核2 默认
        if cluster_pos_2 is None and dm_2 == 'alpha_cluster':
            cluster_pos_2 = cluster_pos
        if sigma_c_2 is None:
            sigma_c_2 = sigma_c
        if brink_antisym_2 is None:
            brink_antisym_2 = brink_antisym

        # ★ 把 α-cluster 的质心移到原点, 再做 SO(3) 旋转 / ±b/2 平移。
        #   zmatrix_to_cartesian / dv_to_cartesian / dist_to_cartesian 反解出的
        #   坐标把第 1 个原子放在原点 (质心偏离 ~2 fm), 若不居中:
        #     (1) 旋转绕原子 1 而非质心 → 取向平均后质心在球面上游走, 密度被抹散;
        #     (2) _compute_A_* 的 ±b/2 网格平移隐含"密度中心在原点", 于是实际
        #         有效冲击参数与 yaml 名义 b 不符。
        #   减去质心后等价于"绕质心旋转 + 真实 b"。对刚体先减质心再绕原点旋转
        #   与先旋转再减质心结果一致, 故放在旋转前最稳妥。WS 核 (cluster_pos=None)
        #   自然跳过; 共享同一对象 (对称碰撞) 时保持 `is` 关系以复用旋转。
        _share_cp = cluster_pos_2 is cluster_pos
        if dm_1 == 'alpha_cluster' and cluster_pos is not None:
            cluster_pos = cluster_pos - cluster_pos.mean(dim=1, keepdim=True)
        if dm_2 == 'alpha_cluster' and cluster_pos_2 is not None:
            cluster_pos_2 = cluster_pos if _share_cp else (
                cluster_pos_2 - cluster_pos_2.mean(dim=1, keepdim=True)
            )

        # broadcast sigma_c, y
        if dm_1 == 'alpha_cluster':
            if sigma_c is None:
                sigma_c = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c, (int, float)):
                sigma_c = torch.full((B,), float(sigma_c), device=device, dtype=dtype)
            elif sigma_c.dim() == 0:
                sigma_c = sigma_c.expand(B).contiguous()

        if dm_2 == 'alpha_cluster':
            if sigma_c_2 is None:
                sigma_c_2 = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c_2, (int, float)):
                sigma_c_2 = torch.full((B,), float(sigma_c_2), device=device, dtype=dtype)
            elif sigma_c_2.dim() == 0:
                sigma_c_2 = sigma_c_2.expand(B).contiguous()

        if y is None:
            y = torch.full((B,), self.default_y, device=device, dtype=dtype)
        elif isinstance(y, (int, float)):
            y = torch.full((B,), float(y), device=device, dtype=dtype)
        elif y.dim() == 0:
            y = y.expand(B).contiguous()

        N = int(n_orientations)

        # ================================================================
        # 路径 A：两核都是 alpha_cluster → 高效 flatten 路径
        # ================================================================
        if dm_1 == 'alpha_cluster' and dm_2 == 'alpha_cluster':
            n_cl_2 = cluster_pos_2.shape[1] if cluster_pos_2 is not None else n_cl

            # 采样旋转
            R_1 = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            r_rotated_1 = torch.einsum('bnij,bkj->bnki', R_1, cluster_pos)

            if cluster_pos_2 is cluster_pos:
                r_rotated_2 = r_rotated_1
                R_2 = R_1
            else:
                R_2 = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
                r_rotated_2 = torch.einsum('bnij,bkj->bnki', R_2, cluster_pos_2)

            # flatten
            r_flat_1 = r_rotated_1.reshape(B * N, n_cl, 3)
            r_flat_2 = r_rotated_2.reshape(B * N, n_cl_2, 3)
            sigma_flat_1 = sigma_c.unsqueeze(1).expand(B, N).reshape(-1)
            sigma_flat_2 = sigma_c_2.unsqueeze(1).expand(B, N).reshape(-1)
            y_flat = y.unsqueeze(1).expand(B, N).reshape(-1)

            total = B * N
            if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
                image_all, p_axis_q = self.forward(
                    r_flat_1, sigma_flat_1, y_flat,
                    cluster_pos_2=r_flat_2, sigma_c_2=sigma_flat_2,
                    density_model_1=dm_1, density_model_2=dm_2,
                    incoherent=incoherent,
                    brink_antisym=brink_antisym, brink_antisym_2=brink_antisym_2,
                )
            else:
                outputs = []
                p_axis_q = None
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    img_chunk, p_axis_q = self.forward(
                        r_flat_1[start:end], sigma_flat_1[start:end], y_flat[start:end],
                        cluster_pos_2=r_flat_2[start:end], sigma_c_2=sigma_flat_2[start:end],
                        density_model_1=dm_1, density_model_2=dm_2,
                        incoherent=incoherent,
                        brink_antisym=brink_antisym, brink_antisym_2=brink_antisym_2,
                    )
                    outputs.append(img_chunk)
                image_all = torch.cat(outputs, dim=0)

            T_size = image_all.shape[-1]
            image_avg = image_all.view(B, N, T_size, T_size).mean(dim=1)
            return image_avg, p_axis_q

        # ================================================================
        # 路径 A2：α-cluster + 球形 WS → Pb 端 compute O(B·N) → O(B)
        # ----------------------------------------------------------------
        # 球形 WS 核 (β=0) 密度旋转不变 → 完全不需要对它采样 SO(3)，且 rho/F_3d
        # 是真常量 (已在 __init__ 缓存为 buffer)。T_prim/V_x/V_y 依赖 y 但不依赖
        # 取向 → 按 batch 唯一的 y 值算 (O(B))，再 expand 到 (B·N) 喂给 α 端
        # 的 amplitude grid 上。这是 plan §4 物理观察"球形原子核没有 orientation
        # 自由度"的工程兑现，整体可再叠加 ~2× 加速 (相对 path B 平铺)。
        # ================================================================
        sym_target_is_2 = (dm_1 == 'alpha_cluster' and dm_2 == 'woods_saxon'
                           and self._is_spherical_ws(2))
        sym_target_is_1 = (dm_2 == 'alpha_cluster' and dm_1 == 'woods_saxon'
                           and self._is_spherical_ws(1))
        if sym_target_is_2 or sym_target_is_1:
            return self._orient_avg_alpha_plus_sym_ws(
                cluster_pos if sym_target_is_2 else cluster_pos_2,
                sigma_c if sym_target_is_2 else sigma_c_2,
                y,
                n_orientations=N,
                alpha_nuclear_ID=(1 if sym_target_is_2 else 2),
                sym_nuclear_ID=(2 if sym_target_is_2 else 1),
                brink_antisym=(brink_antisym if sym_target_is_2 else brink_antisym_2),
                incoherent=incoherent,
                chunk_size=chunk_size,
                generator=generator,
            )

        # ================================================================
        # 路径 B：至少一个核是 WS → 把 N 个取向也平铺进 batch 维（避免 Python 串行）
        # ================================================================
        # 旧实现是 for k in range(N) 串行调 forward(B, ...)；新版本把 (B, N) 平铺
        # 成 B*N 的大 batch，一次性调一次 forward。原始 (双重旋转防御 + 物理正确性)
        # 的语义保持不变：
        #   - alpha_cluster 端：把 R_k 应用到 cluster_pos 后**预旋转**得到 (B*N, n_cl, 3)，
        #                       再传 R_k=None 给 forward，避免内部再旋转一遍。
        #   - woods_saxon 端：cluster_pos=None，把 (B*N, 3, 3) 旋转矩阵 R 透传给
        #                     ``_compute_density_ws`` (它本来就支持 (B, 3, 3) 输入)。
        # 与路径 A 完全同构，只是多了"WS 走 R 透传"这条分支。
        else:
            # 采样 (B, N, 3, 3) 旋转矩阵
            R_1_all = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            if cluster_pos_2 is not None and cluster_pos_2 is not cluster_pos:
                R_2_all = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            else:
                R_2_all = R_1_all

            total = B * N

            # 核 1
            if dm_1 == 'alpha_cluster':
                # 预旋转 (B, N, n_cl, 3) → (B*N, n_cl, 3)
                n_cl_1 = cluster_pos.shape[1]
                r_rot_1 = torch.einsum('bnij,bkj->bnki', R_1_all, cluster_pos)
                r_flat_1 = r_rot_1.reshape(total, n_cl_1, 3)
                sigma_flat_1 = sigma_c.unsqueeze(1).expand(B, N).reshape(-1)
                R_1_pass = None
            else:   # woods_saxon
                r_flat_1 = None
                sigma_flat_1 = None
                R_1_pass = R_1_all.reshape(total, 3, 3)

            # 核 2
            if dm_2 == 'alpha_cluster':
                n_cl_2 = cluster_pos_2.shape[1]
                if cluster_pos_2 is cluster_pos:
                    r_rot_2 = r_rot_1                                    # 共享预旋转结果
                    r_flat_2 = r_rot_2.reshape(total, n_cl_2, 3)
                else:
                    r_rot_2 = torch.einsum('bnij,bkj->bnki', R_2_all, cluster_pos_2)
                    r_flat_2 = r_rot_2.reshape(total, n_cl_2, 3)
                sigma_flat_2 = sigma_c_2.unsqueeze(1).expand(B, N).reshape(-1)
                R_2_pass = None
            else:   # woods_saxon
                r_flat_2 = None
                sigma_flat_2 = None
                R_2_pass = R_2_all.reshape(total, 3, 3)

            y_flat = y.unsqueeze(1).expand(B, N).reshape(-1)

            def _call_forward(start: int, end: int):
                return self.forward(
                    r_flat_1[start:end] if r_flat_1 is not None else None,
                    sigma_flat_1[start:end] if sigma_flat_1 is not None else None,
                    y_flat[start:end],
                    cluster_pos_2=(r_flat_2[start:end] if r_flat_2 is not None else None),
                    sigma_c_2=(sigma_flat_2[start:end] if sigma_flat_2 is not None else None),
                    density_model_1=dm_1, density_model_2=dm_2,
                    R_1=(R_1_pass[start:end] if R_1_pass is not None else None),
                    R_2=(R_2_pass[start:end] if R_2_pass is not None else None),
                    incoherent=incoherent,
                    brink_antisym=brink_antisym, brink_antisym_2=brink_antisym_2,
                )

            if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
                image_all, p_axis_q = _call_forward(0, total)
            else:
                outputs = []
                p_axis_q = None
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    img_chunk, p_axis_q = _call_forward(start, end)
                    outputs.append(img_chunk)
                image_all = torch.cat(outputs, dim=0)

            T_size = image_all.shape[-1]
            image_avg = image_all.view(B, N, T_size, T_size).mean(dim=1)
            return image_avg, p_axis_q

    # ------------------------------------------------------------------
    # 球形 WS + α-cluster 快速路径（plan §"Pb 没必要 orient"）
    # ------------------------------------------------------------------

    def _orient_avg_alpha_plus_sym_ws(
        self,
        alpha_cluster_pos: torch.Tensor,   # (B, n_cl, 3) — α 核的 cluster_pos
        alpha_sigma_c: torch.Tensor,       # (B,)
        y: torch.Tensor,                   # (B,)
        n_orientations: int,
        alpha_nuclear_ID: int,             # 1 或 2 — α 端在配置里的 ID
        sym_nuclear_ID: int,               # 另一边 (球形 WS) 的 ID
        brink_antisym: bool = True,
        incoherent: bool = False,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        α-cluster + 球形 WS 的快速 SO(3) 平均。

        关键加速：
          - 球形 WS 密度 rotation-invariant → 不采样 R₂，直接复用 __init__ 缓存
            的 (rho_p, rho_total, F_3d) buffers
          - WS 端的 T_prim / T_thick / V_x / V_y 仅依赖 y → 按 batch (O(B)) 算，
            然后在 (B·N) 维度上做 view 扩展，避免重复 compute
          - α 端的 density / F / T / V 各异 → O(B·N) 计算 (用 chunk_size 控显存)
          - amplitude 与末端 FFT 仍 O(B·N) (物理上不可避免)

        生产配置 (B=32, N=8, α+Pb)：WS 端 compute 从 O(B·N)=256 → O(B)=32，
        叠加 path B 平铺约再 ~2× 总加速。
        """
        if not self._is_spherical_ws(sym_nuclear_ID):
            raise RuntimeError(
                f"_orient_avg_alpha_plus_sym_ws 被调用但 nuclear {sym_nuclear_ID} "
                f"不是球形 WS"
            )

        B, n_cl, _ = alpha_cluster_pos.shape
        device = alpha_cluster_pos.device
        dtype = alpha_cluster_pos.dtype
        N = int(n_orientations)
        total = B * N

        # ─── 1. α 端旋转 + 平铺到 (B·N) ───
        R_alpha = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
        r_rot = torch.einsum('bnij,bkj->bnki', R_alpha, alpha_cluster_pos)
        r_flat = r_rot.reshape(total, n_cl, 3)
        sigma_flat = alpha_sigma_c.unsqueeze(1).expand(B, N).reshape(-1)
        y_flat = y.unsqueeze(1).expand(B, N).reshape(-1)

        # ─── 2. WS 端：rho/F 从 cache 取，T_prim/V 按 B 唯一 y 算 ───
        # rho / F: buffer (1, ...) → 让下游 expand 到自身 batch
        # T/V 依赖 y → 按 (B,) 算一次
        rho_total_sym_one = getattr(self, f'_static_ws_rho_total_{sym_nuclear_ID}')   # (1, N³)
        F_3d_sym_one = getattr(self, f'_static_ws_F_3d_{sym_nuclear_ID}')              # (1, N³) complex

        gamma_sym = self.gamma_1 if sym_nuclear_ID == 1 else self.gamma_2
        Ep_sym = self.Ep_1 if sym_nuclear_ID == 1 else self.Ep_2
        Z_sym = self.Z_1 if sym_nuclear_ID == 1 else self.Z_2
        A_sym = self.A_1 if sym_nuclear_ID == 1 else self.A_2

        gamma_alpha = self.gamma_1 if alpha_nuclear_ID == 1 else self.gamma_2
        Ep_alpha = self.Ep_1 if alpha_nuclear_ID == 1 else self.Ep_2
        Z_alpha = self.Z_1 if alpha_nuclear_ID == 1 else self.Z_2
        A_alpha = self.A_1 if alpha_nuclear_ID == 1 else self.A_2

        # qL_sym 依赖于 *对面核* 的 gamma：j/ψ rapidity y → 给 WS 端的有效 qL
        # 与 forward() 中的约定一致 (核1 用 +y, 核2 用 -y 通道)
        if sym_nuclear_ID == 1:
            qL_sym_B = self.Mjpsi * 0.5 / gamma_sym * torch.exp(y)
        else:
            qL_sym_B = self.Mjpsi * 0.5 / gamma_sym * torch.exp(-y)

        # T_prim / T_thick: (B,) 唯一
        rho_total_sym_B = rho_total_sym_one.expand(B, -1, -1, -1)        # view, free
        T_prim_sym_B, T_thick_sym_B = self._compute_T_prim_and_thickness(
            rho_total_sym_B, qL_sym_B, Z_nuc=Z_sym
        )
        # photon field: (B,) 唯一
        F_sym_B = F_3d_sym_one.expand(B, -1, -1, -1)                     # view
        Vx_sym_B, Vy_sym_B = self._compute_photon_field(
            F_sym_B, gamma_sym, qL_sym_B, Z_nuc=Z_sym
        )

        # ─── 3. expand WS 中间量到 (B·N)，view 操作 ───
        def _expand_BN(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(B, N, *t.shape[1:]).reshape(total, *t.shape[1:])

        T_prim_sym = _expand_BN(T_prim_sym_B)
        T_thick_sym = _expand_BN(T_thick_sym_B)
        Vx_sym = _expand_BN(Vx_sym_B)
        Vy_sym = _expand_BN(Vy_sym_B)

        # ─── 4. α 端 + amplitudes + FFT  (chunked) ───
        def _process(start: int, end: int) -> torch.Tensor:
            cp = r_flat[start:end]
            sig = sigma_flat[start:end]
            y_c = y_flat[start:end]

            if brink_antisym:
                rho_p_a, rho_total_a = self._compute_density_brink(cp, sig, nuclear_ID=alpha_nuclear_ID)
            else:
                rho_p_a, rho_total_a = self._compute_density_gaussian_sum(cp, sig, nuclear_ID=alpha_nuclear_ID)
            F_a = self._compute_F_3d(rho_p_a)

            if alpha_nuclear_ID == 1:
                qL_a = self.Mjpsi * 0.5 / gamma_alpha * torch.exp(y_c)
            else:
                qL_a = self.Mjpsi * 0.5 / gamma_alpha * torch.exp(-y_c)
            T_prim_a, T_thick_a = self._compute_T_prim_and_thickness(
                rho_total_a, qL_a, Z_nuc=Z_alpha
            )
            Vx_a, Vy_a = self._compute_photon_field(F_a, gamma_alpha, qL_a, Z_nuc=Z_alpha)

            # Egamma：核1 target → +y, 核2 target → -y (与 forward() 严格一致)
            E_alpha = 0.5 * self.Mjpsi * torch.exp(y_c if alpha_nuclear_ID == 1 else -y_c)
            E_sym = 0.5 * self.Mjpsi * torch.exp(y_c if sym_nuclear_ID == 1 else -y_c)

            # α 是 target → 用 α 的 T_prim, sym 的 T_thick/V (source)
            A_alpha_x, A_alpha_y = self._compute_amplitude_at_grid(
                nuclear_ID=alpha_nuclear_ID,
                T_prim_target=T_prim_a,
                T_thick_source=T_thick_sym[start:end],
                Vx_source=Vx_sym[start:end],
                Vy_source=Vy_sym[start:end],
                Egamma_target=E_sym, Ep_target=Ep_alpha,   # ★bugfix(2026-06): σ_VN 用源(sym)光子能量 E_sym, 对齐学长 C++
                A_source=A_sym,
            )
            # sym 是 target → 用 sym 的 T_prim, α 的 T_thick/V (source)
            A_sym_x, A_sym_y = self._compute_amplitude_at_grid(
                nuclear_ID=sym_nuclear_ID,
                T_prim_target=T_prim_sym[start:end],
                T_thick_source=T_thick_a,
                Vx_source=Vx_a, Vy_source=Vy_a,
                Egamma_target=E_alpha, Ep_target=Ep_sym,   # ★bugfix(2026-06): σ_VN 用源(α)光子能量 E_alpha, 对齐学长 C++
                A_source=A_alpha,
            )

            Ax = A_alpha_x + A_sym_x
            Ay = A_alpha_y + A_sym_y
            img, _p_axis = self._coherent_image_q(Ax, Ay)

            if incoherent:
                # incoherent 路径要求两核都有 cluster_pos / V / T_thick
                # 这里 α 端是真实的，sym 端给 dummy cluster_pos (球形物理上等价)
                # 仅在 incoherent=True 时进入这一支，保持与 forward() 一致
                cp_for_incoh = cp
                sc_for_incoh = sig
                if alpha_nuclear_ID == 1:
                    T_thick_1_seg, T_thick_2_seg = T_thick_a, T_thick_sym[start:end]
                    Vx_1_seg, Vy_1_seg = Vx_a, Vy_a
                    Vx_2_seg, Vy_2_seg = Vx_sym[start:end], Vy_sym[start:end]
                    rho_total_1_for_incoh = rho_total_a
                else:
                    T_thick_1_seg, T_thick_2_seg = T_thick_sym[start:end], T_thick_a
                    Vx_1_seg, Vy_1_seg = Vx_sym[start:end], Vy_sym[start:end]
                    Vx_2_seg, Vy_2_seg = Vx_a, Vy_a
                    rho_total_1_for_incoh = rho_total_sym_one.expand(end - start, -1, -1, -1)

                total_incoh_prob = self._compute_incoherent_total_prob(
                    cp_for_incoh, sc_for_incoh, y_c, rho_total_1_for_incoh,
                    T_thick_1_seg, T_thick_2_seg, Vx_2_seg, Vy_2_seg,
                    Vx_1_seg, Vy_1_seg,
                )
                p_axis_q = self._p_axis_q.to(device)
                Pq_X, Pq_Y = _meshgrid_ij(p_axis_q, p_axis_q)
                q2 = (Pq_X ** 2 + Pq_Y ** 2)[None, :, :]
                b_slope = self.b_slope_incoh
                incoh_dist = (total_incoh_prob.view(-1, 1, 1) * b_slope / math.pi *
                              torch.exp(-b_slope * q2))
                img = img + incoh_dist

            return img

        if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
            image_all = _process(0, total)
        else:
            outputs = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                outputs.append(_process(start, end))
            image_all = torch.cat(outputs, dim=0)

        T_size = image_all.shape[-1]
        image_avg = image_all.view(B, N, T_size, T_size).mean(dim=1)
        return image_avg, self._p_axis_q.to(device)

    def forward_from_zmatrix(
        self,
        z: Optional[torch.Tensor],                    # (B, 6) Z-matrix — 核1
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y: Optional[Union[torch.Tensor, float]] = None,
        z_2: Optional[torch.Tensor] = None,           # (B, 6) 核2 的 z-matrix
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        n_orientations: int = 20,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2 主入口：z-matrix → 旋转平均后的 image_q_total。

        当某核 ``density_model='woods_saxon'`` 时，对应的 z 参数应为 None。

        全程可微。
        """
        from .cluster_geometry import zmatrix_to_cartesian

        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()

        # 核1
        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "external continuous densities cannot be reconstructed from a z-matrix; "
                "use forward_multi_y_from_density()"
            )
        if dm_1 == 'alpha_cluster':
            if z is None or z.dim() != 2 or z.shape[1] != 6:
                raise ValueError(f"alpha_cluster 核1 需要 z: (B, 6)，收到 {z}")
            cluster_pos = zmatrix_to_cartesian(z)
        else:
            cluster_pos = None

        # 核2
        if dm_2 == 'alpha_cluster':
            if z_2 is not None:
                if z_2.dim() != 2 or z_2.shape[1] != 6:
                    raise ValueError(f"z_2 期望 (B, 6)，收到 {tuple(z_2.shape)}")
                cluster_pos_2 = zmatrix_to_cartesian(z_2)
            elif z is not None:
                cluster_pos_2 = zmatrix_to_cartesian(z)
            else:
                raise ValueError("alpha_cluster 核2 需要 z_2 或 z")
        else:
            cluster_pos_2 = None

        return self.forward_with_orientation_avg(
            cluster_pos,
            sigma_c=sigma_c, y=y,
            cluster_pos_2=cluster_pos_2, sigma_c_2=sigma_c_2,
            density_model_1=dm_1, density_model_2=dm_2,
            n_orientations=n_orientations,
            incoherent=incoherent,
            brink_antisym=brink_antisym, brink_antisym_2=brink_antisym_2,
            chunk_size=chunk_size,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # multi-rapidity V_y vectorize 路径 (plan §4.3.B.8 做法 B)
    # ------------------------------------------------------------------
    #
    # 物理观察 (与原 forward() 数据流逐项对照)：
    #
    #   阶段                                       是否依赖 y    V_y 间能否共享
    #   ---------------------------------------    -----------   --------------
    #   _compute_density_*  (rho_p, rho_total)     否            ✓
    #   _compute_F_3d       (3D FFT 形状因子)       否            ✓
    #   qL = M_J/ψ · 0.5 / γ · exp(±y)              是            必须 per-y
    #   _compute_T_prim_and_thickness               是            必须 per-y
    #   _compute_photon_field (含 Bessel K_1)       是            必须 per-y
    #   _compute_amplitude_at_grid                  是            必须 per-y
    #   _coherent_image_q                            是            必须 per-y
    #
    # 实现策略：算一次密度 + F_3d (B 维)，把 (B, ...) expand→reshape 到
    # (B·V_y, ...)，y flatten 到 (B·V_y,) → 走原 forward() 的下游子函数。
    # 子函数都按 dim 0 = batch 工作，无需改动。
    #
    # 显存：T_prim/V/amplitude/image 仍是 B·V_y 倍（与 V_y 次循环一致），但
    # density + 3D FFT 只跑 B 次。生产配置 (V_y=5) 下密度占总时间约 30-50%，
    # 实测加速 ~1.5-2.5×。
    # ------------------------------------------------------------------

    def _expand_to_BVY(self, t: torch.Tensor, V_y: int) -> torch.Tensor:
        """
        (B, *) → (B, V_y, *) → (B·V_y, *)。
        expand 是 view（零拷贝），reshape 会强制 contiguous → 实际拷贝一次。
        V_y=1 时直接返回 t（避免无谓拷贝）。
        """
        if V_y == 1:
            return t
        B = t.shape[0]
        return t.unsqueeze(1).expand(B, V_y, *t.shape[1:]).reshape(
            B * V_y, *t.shape[1:]
        )

    def _average_over_impact_parameter(
        self,
        evaluate_at_current_b: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """在同一 ``b`` 节点回调内计算完整图像，再做 hard-window 平均。

        ``evaluate_at_current_b`` 必须返回该 ``self.b_imp`` 下的完整
        ``(image, p_axis)``。对 total 图像而言，coherent 与 incoherent
        必须都在这个回调内，才会共享同一 ``2π b db`` quadrature。
        无论成功还是异常，都恢复调用前的 ``self.b_imp``。
        """
        configured_grid = getattr(self, "_b_integrate_grid", None)
        if configured_grid is None:
            b_values, weights = (float(self.b_imp),), (1.0,)
        else:
            # ``None`` is the only fixed-b sentinel.  An explicitly supplied
            # empty grid is a malformed integration request and must not
            # silently masquerade as a successful fixed-b evaluation.
            b_values, weights = _normalized_impact_parameter_quadrature(
                tuple(float(b) for b in configured_grid)
            )

        original_b = self.b_imp
        image_average = None
        p_axis = None
        try:
            for b_value, weight in zip(b_values, weights):
                self.b_imp = float(b_value)
                image_at_b, p_axis_at_b = evaluate_at_current_b()
                if image_average is None:
                    image_average = image_at_b if weight == 1.0 else image_at_b * weight
                    p_axis = p_axis_at_b
                else:
                    image_average = image_average + image_at_b * weight
        finally:
            self.b_imp = original_b

        if image_average is None or p_axis is None:  # pragma: no cover - 防御性不变量
            raise RuntimeError("impact-parameter quadrature produced no image")
        return image_average, p_axis

    def forward_multi_y(
        self,
        cluster_pos: Optional[torch.Tensor],          # (B, n_clusters, 3)  核1
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y_views: Optional[torch.Tensor] = None,        # ★ (B, V_y) 必传
        cluster_pos_2: Optional[torch.Tensor] = None,  # (B, n_clusters_2, 3) 核2
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        R_1: Optional[torch.Tensor] = None,            # (B, 3, 3) 单取向旋转
        R_2: Optional[torch.Tensor] = None,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
        _return_complex_amplitude: bool = False,
        _fixed_b_values: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        多 rapidity vectorized 单取向 forward (plan §4.3.B.8 做法 B)。

        与 ``forward()`` 数学等价、数值一致 (atol ~1e-5)。区别仅在于：算一次密度
        + 3D FFT 后，对 V_y 个 rapidity 共享中间量、只重算 y-dependent 部分。

        Args
        ----
        cluster_pos   : (B, n_clusters, 3) 核1 团簇位置（WS 时可 None）
        sigma_c       : (B,) 或 scalar；核1 σ
        y_views       : (B, V_y) 必传；每样本 V_y 个 rapidity
        cluster_pos_2 : (B, n_clusters_2, 3) 核2；None=复制核1
        sigma_c_2     : 核2 σ；None=sigma_c
        density_model_1/2 : 同 forward()
        R_1, R_2      : (B, 3, 3) 单取向旋转；高级取向平均请用
                        ``forward_with_orientation_avg_multi_y``
        incoherent / brink_antisym* : 同 forward()

        Returns
        -------
        image_q_total : (B, V_y, target_size_q, target_size_q)
        p_axis_q      : (target_size_q,)
        """
        if y_views is None or not isinstance(y_views, torch.Tensor):
            raise ValueError(
                "forward_multi_y: y_views 必须传入 torch.Tensor (B, V_y)"
            )
        if y_views.dim() != 2:
            raise ValueError(
                f"forward_multi_y: y_views 期望 (B, V_y)，收到 {tuple(y_views.shape)}"
            )

        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()

        # ── batch / device / dtype 推断 (与 forward() 同口径) ──
        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "density_model='external_density' requires "
                "forward_multi_y_from_density(); no generic Brink fallback is allowed"
            )
        if dm_1 == 'alpha_cluster':
            if cluster_pos is None or not isinstance(cluster_pos, torch.Tensor):
                raise ValueError(
                    f"density_model_1='alpha_cluster' 需要 cluster_pos，收到 {cluster_pos}"
                )
            B = cluster_pos.shape[0]
            device = cluster_pos.device
            dtype = cluster_pos.dtype
        else:  # woods_saxon
            if R_1 is not None:
                B = R_1.shape[0]; device = R_1.device; dtype = R_1.dtype
            elif cluster_pos_2 is not None and isinstance(cluster_pos_2, torch.Tensor):
                B = cluster_pos_2.shape[0]; device = cluster_pos_2.device; dtype = cluster_pos_2.dtype
            elif R_2 is not None:
                B = R_2.shape[0]; device = R_2.device; dtype = R_2.dtype
            else:
                B = y_views.shape[0]; device = y_views.device; dtype = y_views.dtype

        V_y = int(y_views.shape[1])
        if y_views.shape[0] != B:
            raise ValueError(
                f"forward_multi_y: y_views.shape[0]={y_views.shape[0]} ≠ B={B}"
            )
        BVY = B * V_y

        # 核2 默认
        if cluster_pos_2 is None and dm_2 == 'alpha_cluster':
            cluster_pos_2 = cluster_pos
        if sigma_c_2 is None:
            sigma_c_2 = sigma_c
        if brink_antisym_2 is None:
            brink_antisym_2 = brink_antisym

        # broadcast scalars (与 forward() 完全一致)
        if dm_1 == 'alpha_cluster':
            if sigma_c is None:
                sigma_c = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c, (int, float)):
                sigma_c = torch.full((B,), float(sigma_c), device=device, dtype=dtype)
            elif sigma_c.dim() == 0:
                sigma_c = sigma_c.expand(B).contiguous()
        if dm_2 == 'alpha_cluster':
            if sigma_c_2 is None:
                sigma_c_2 = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c_2, (int, float)):
                sigma_c_2 = torch.full((B,), float(sigma_c_2), device=device, dtype=dtype)
            elif sigma_c_2.dim() == 0:
                sigma_c_2 = sigma_c_2.expand(B).contiguous()

        # ──────────────────────────────────────────────────────
        # 1) 密度 + F_3d (B 维，与 V_y 无关，只算一次)
        # ──────────────────────────────────────────────────────
        if dm_1 == 'woods_saxon':
            rho_p_1, rho_total_1 = self._compute_density_ws(
                nuclear_ID=1, R=R_1, batch_size=B
            )
        else:
            cp_1 = cluster_pos
            if R_1 is not None and cp_1 is not None:
                cp_1 = torch.einsum('bij,bkj->bki', R_1, cp_1)
            if brink_antisym:
                rho_p_1, rho_total_1 = self._compute_density_brink(cp_1, sigma_c, nuclear_ID=1)
            else:
                rho_p_1, rho_total_1 = self._compute_density_gaussian_sum(cp_1, sigma_c, nuclear_ID=1)

        if dm_2 == 'woods_saxon':
            rho_p_2, rho_total_2 = self._compute_density_ws(
                nuclear_ID=2, R=R_2, batch_size=B
            )
        else:
            cp_2 = cluster_pos_2
            if R_2 is not None and cp_2 is not None:
                cp_2 = torch.einsum('bij,bkj->bki', R_2, cp_2)
            if brink_antisym_2:
                rho_p_2, rho_total_2 = self._compute_density_brink(cp_2, sigma_c_2, nuclear_ID=2)
            else:
                rho_p_2, rho_total_2 = self._compute_density_gaussian_sum(cp_2, sigma_c_2, nuclear_ID=2)

        F_3d_sorted_1 = self._compute_F_3d(rho_p_1)         # (B, N, N, N) complex
        F_3d_sorted_2 = self._compute_F_3d(rho_p_2)

        # ──────────────────────────────────────────────────────
        # 2) expand 到 (B·V_y, ...)；y flatten 到 (B·V_y,)
        # ──────────────────────────────────────────────────────
        rho_total_1_BVY = self._expand_to_BVY(rho_total_1, V_y)
        rho_total_2_BVY = self._expand_to_BVY(rho_total_2, V_y)
        F_3d_sorted_1_BVY = self._expand_to_BVY(F_3d_sorted_1, V_y)
        F_3d_sorted_2_BVY = self._expand_to_BVY(F_3d_sorted_2, V_y)

        y_flat = y_views.to(device=device, dtype=dtype).reshape(BVY)

        # ──────────────────────────────────────────────────────
        # 3) y-dependent: qL / T_prim / V / amplitude / image
        # ──────────────────────────────────────────────────────
        qL_1 = self.Mjpsi * 0.5 / self.gamma_1 * torch.exp(y_flat)
        qL_2 = self.Mjpsi * 0.5 / self.gamma_2 * torch.exp(-y_flat)

        T_prim_1, T_thick_1 = self._compute_T_prim_and_thickness(
            rho_total_1_BVY, qL_1, Z_nuc=self.Z_1
        )
        T_prim_2, T_thick_2 = self._compute_T_prim_and_thickness(
            rho_total_2_BVY, qL_2, Z_nuc=self.Z_2
        )

        Vx_1, Vy_1 = self._compute_photon_field(
            F_3d_sorted_1_BVY, self.gamma_1, qL_1, Z_nuc=self.Z_1
        )
        Vx_2, Vy_2 = self._compute_photon_field(
            F_3d_sorted_2_BVY, self.gamma_2, qL_2, Z_nuc=self.Z_2
        )

        Egamma_1 = 0.5 * self.Mjpsi * torch.exp(y_flat)
        Egamma_2 = 0.5 * self.Mjpsi * torch.exp(-y_flat)

        def _complex_amplitude_at_current_b():
            a1x, a1y = self._compute_amplitude_at_grid(
                nuclear_ID=1, T_prim_target=T_prim_1, T_thick_source=T_thick_2,
                Vx_source=Vx_2, Vy_source=Vy_2,
                Egamma_target=Egamma_2, Ep_target=self.Ep_1, A_source=self.A_2)
            a2x, a2y = self._compute_amplitude_at_grid(
                nuclear_ID=2, T_prim_target=T_prim_2, T_thick_source=T_thick_1,
                Vx_source=Vx_1, Vy_source=Vy_1,
                Egamma_target=Egamma_1, Ep_target=self.Ep_2, A_source=self.A_1)
            return self._coherent_amplitude_q(a1x + a2x, a1y + a2y)

        if _return_complex_amplitude:
            if incoherent:
                raise ValueError(
                    "fixed-b complex amplitude primitive is coherent/pre-square; "
                    "incoherent=True is not defined"
                )
            if _fixed_b_values is None:
                raise ValueError("fixed-b amplitude primitive requires explicit b_values")
            b_values = tuple(float(value) for value in _fixed_b_values)
            if not b_values or any(not math.isfinite(value) or value < 0.0 for value in b_values):
                raise ValueError("fixed-b amplitude nodes must be finite, non-negative and non-empty")
            if len(b_values) > 1 and any(
                right <= left for left, right in zip(b_values[:-1], b_values[1:])
            ):
                raise ValueError("fixed-b amplitude nodes must be strictly increasing")
            old_b = self.b_imp
            mx_by_b = []
            my_by_b = []
            p_full = None
            try:
                for b_value in b_values:
                    self.b_imp = b_value
                    mx, my, p_full = _complex_amplitude_at_current_b()
                    mx_by_b.append(mx)
                    my_by_b.append(my)
            finally:
                self.b_imp = old_b
            assert p_full is not None
            full_size = mx_by_b[0].shape[-1]
            mx = torch.stack(mx_by_b, dim=1).view(B, V_y, len(b_values), full_size, full_size)
            my = torch.stack(my_by_b, dim=1).view(B, V_y, len(b_values), full_size, full_size)
            # Public axis order is (batch,b,y,p_x,p_y).
            return mx.permute(0, 2, 1, 3, 4), my.permute(0, 2, 1, 3, 4), p_full

        # ★ b 内层循环 (2026-07 修正): F_3d/V/T_prim 已算好 (b-无关)，
        #   每个 b 节点内先合并 coherent + incoherent，再用归一化复合梯形
        #   2πb db 权重平均。这是 hard-window average，不是绝对截面。
        #   _b_integrate_grid=None → 单 b (self.b_imp)；显式空 grid 非法。
        #   Egamma_target 用源光子能量 (★bugfix 2026-06, 对齐学长 C++)。
        if incoherent:
            # 与 forward() 一致：cluster_pos / sigma_c 只需 expand 一次。
            if cluster_pos is not None:
                cp_for_incoh = self._expand_to_BVY(cluster_pos, V_y)
            else:
                cp_for_incoh = torch.zeros(BVY, 1, 3, device=device)
            if sigma_c is not None:
                sc_for_incoh = self._expand_to_BVY(sigma_c, V_y)
            else:
                sc_for_incoh = torch.full((BVY,), self.default_sigma_c, device=device)
        else:
            cp_for_incoh = sc_for_incoh = None

        def _total_img_at_current_b():
            amplitude_x, amplitude_y, _ = _complex_amplitude_at_current_b()
            image_at_b, p_axis_at_b = self.project_complex_amplitude_intensity_to_output(
                amplitude_x, amplitude_y,
            )

            if incoherent:
                total_incoh_prob = self._compute_incoherent_total_prob(
                    cp_for_incoh, sc_for_incoh, y_flat, rho_total_1_BVY,
                    T_thick_1, T_thick_2, Vx_2, Vy_2, Vx_1, Vy_1,
                )
                q2 = self._p_axis_q_radius2.to(image_at_b.device)[None, :, :]
                b_slope = self.b_slope_incoh
                incoh_dist = (
                    total_incoh_prob.view(BVY, 1, 1) * b_slope / math.pi
                    * torch.exp(-b_slope * q2)
                )
                image_at_b = image_at_b + incoh_dist
            return image_at_b, p_axis_at_b

        image_q_flat, p_axis_q = self._average_over_impact_parameter(
            _total_img_at_current_b
        )

        # ── 4) reshape (B·V_y, T, T) → (B, V_y, T, T) ──
        T_size = image_q_flat.shape[-1]
        image_q_total = image_q_flat.view(B, V_y, T_size, T_size)
        return image_q_total, p_axis_q

    def _validate_external_density_pair(
        self,
        rho_p: torch.Tensor,
        rho_total: torch.Tensor,
        *,
        nuclear_ID: int,
        normalization_rtol: float,
        negativity_atol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate a caller-supplied continuous one-body density.

        The public UPC path historically constructed either a Woods--Saxon or
        an alpha-centre density internally.  Ground-state 20Ne instead uses the
        hash-pinned analytic ``fden`` oracle, so the density is constructed by
        :mod:`physics.ne20_brink_density` and enters here explicitly.  This
        validator is intentionally fail-closed: it never rescales or clips a
        malformed density behind the caller's back.

        ``rho_p`` and ``rho_total`` must be ``(B,N,N,N)`` on the model WS grid
        and integrate to the configured proton and nucleon numbers.  A tiny
        negative tolerance is allowed for round-off in signed-exponential
        evaluation, but materially negative densities are rejected.
        """
        if not isinstance(rho_p, torch.Tensor) or not isinstance(rho_total, torch.Tensor):
            raise TypeError("external densities must be torch tensors")
        expected_tail = (self._WS_N, self._WS_N, self._WS_N)
        if rho_p.ndim != 4 or tuple(rho_p.shape[1:]) != expected_tail:
            raise ValueError(
                f"rho_p_{nuclear_ID} must have shape (B,{self._WS_N},"
                f"{self._WS_N},{self._WS_N}); got {tuple(rho_p.shape)}"
            )
        if rho_total.shape != rho_p.shape:
            raise ValueError(
                f"rho_total_{nuclear_ID} shape {tuple(rho_total.shape)} does not match "
                f"rho_p_{nuclear_ID} {tuple(rho_p.shape)}"
            )
        if rho_p.device != rho_total.device or rho_p.dtype != rho_total.dtype:
            raise ValueError("rho_p and rho_total must share device and dtype")
        if not rho_p.is_floating_point() or not rho_total.is_floating_point():
            raise TypeError("external densities must use a floating dtype")
        if not torch.isfinite(rho_p).all() or not torch.isfinite(rho_total).all():
            raise ValueError(f"external density for nucleus {nuclear_ID} contains NaN/Inf")
        min_p = float(rho_p.detach().amin().cpu())
        min_total = float(rho_total.detach().amin().cpu())
        if min_p < -abs(float(negativity_atol)) or min_total < -abs(float(negativity_atol)):
            raise ValueError(
                f"external density for nucleus {nuclear_ID} is materially negative: "
                f"min(rho_p)={min_p:.6e}, min(rho_total)={min_total:.6e}"
            )

        vol = float(self._WS_dx ** 3)
        p_integral = rho_p.sum(dim=(1, 2, 3)) * vol
        total_integral = rho_total.sum(dim=(1, 2, 3)) * vol
        cfg = self._nuclear_configs[nuclear_ID]
        expected_p = torch.full_like(p_integral, float(cfg['Z']))
        expected_total = torch.full_like(total_integral, float(cfg['A']))
        if not torch.allclose(p_integral, expected_p, rtol=normalization_rtol, atol=1e-5):
            raise ValueError(
                f"rho_p_{nuclear_ID} normalization mismatch: expected Z={cfg['Z']}, "
                f"got range [{float(p_integral.min()):.8g}, {float(p_integral.max()):.8g}]"
            )
        if not torch.allclose(
            total_integral, expected_total, rtol=normalization_rtol, atol=1e-5
        ):
            raise ValueError(
                f"rho_total_{nuclear_ID} normalization mismatch: expected A={cfg['A']}, "
                f"got range [{float(total_integral.min()):.8g}, "
                f"{float(total_integral.max()):.8g}]"
            )
        return rho_p, rho_total

    def forward_multi_y_from_density(
        self,
        rho_p_1: torch.Tensor,
        rho_total_1: torch.Tensor,
        y_views: torch.Tensor,
        *,
        rho_p_2: Optional[torch.Tensor] = None,
        rho_total_2: Optional[torch.Tensor] = None,
        incoherent: bool = False,
        normalization_rtol: float = 5.0e-4,
        negativity_atol: float = 1.0e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the common UPC calculation from explicit continuous densities.

        This is the canonical entry point for the 20Ne analytic-density
        pipeline.  The two nuclei may carry independent orientations, but they
        must describe the same batch of physical latent rows.  If nucleus 2 is
        omitted, the exact same density tensor is reused; callers performing an
        orientation average should normally provide an independently rotated
        second density.

        The returned image is *one total image*.  At every impact-parameter
        node the coherent intensity and the empirical Gaussian incoherent
        envelope are added first; the shared ``2*pi*b*db`` quadrature is then
        applied.  The incoherent term remains an empirical envelope, not a
        Good--Walker variance.
        """
        if not isinstance(y_views, torch.Tensor) or y_views.ndim != 2:
            raise ValueError(
                "forward_multi_y_from_density: y_views must be a (B,V_y) tensor"
            )
        rho_p_1, rho_total_1 = self._validate_external_density_pair(
            rho_p_1,
            rho_total_1,
            nuclear_ID=1,
            normalization_rtol=float(normalization_rtol),
            negativity_atol=float(negativity_atol),
        )
        if (rho_p_2 is None) != (rho_total_2 is None):
            raise ValueError("rho_p_2 and rho_total_2 must be supplied together")
        if rho_p_2 is None:
            rho_p_2, rho_total_2 = rho_p_1, rho_total_1
        rho_p_2, rho_total_2 = self._validate_external_density_pair(
            rho_p_2,
            rho_total_2,
            nuclear_ID=2,
            normalization_rtol=float(normalization_rtol),
            negativity_atol=float(negativity_atol),
        )

        B = int(rho_p_1.shape[0])
        if rho_p_2.shape[0] != B or y_views.shape[0] != B:
            raise ValueError(
                "external density batch mismatch: "
                f"rho1={rho_p_1.shape[0]}, rho2={rho_p_2.shape[0]}, "
                f"y_views={y_views.shape[0]}"
            )
        if rho_p_2.device != rho_p_1.device or rho_p_2.dtype != rho_p_1.dtype:
            raise ValueError("both nuclei must share density device and dtype")

        device = rho_p_1.device
        dtype = rho_p_1.dtype
        V_y = int(y_views.shape[1])
        BVY = B * V_y

        # Density-level Brink/WS interpolation has already happened in the
        # provider.  From here onward both components pass through the exact
        # same nonlinear amplitude, absorption and image construction.
        F_3d_sorted_1 = self._compute_F_3d(rho_p_1)
        F_3d_sorted_2 = self._compute_F_3d(rho_p_2)
        rho_total_1_BVY = self._expand_to_BVY(rho_total_1, V_y)
        rho_total_2_BVY = self._expand_to_BVY(rho_total_2, V_y)
        F_3d_sorted_1_BVY = self._expand_to_BVY(F_3d_sorted_1, V_y)
        F_3d_sorted_2_BVY = self._expand_to_BVY(F_3d_sorted_2, V_y)
        y_flat = y_views.to(device=device, dtype=dtype).reshape(BVY)

        qL_1 = self.Mjpsi * 0.5 / self.gamma_1 * torch.exp(y_flat)
        qL_2 = self.Mjpsi * 0.5 / self.gamma_2 * torch.exp(-y_flat)
        T_prim_1, T_thick_1 = self._compute_T_prim_and_thickness(
            rho_total_1_BVY, qL_1, Z_nuc=self.Z_1
        )
        T_prim_2, T_thick_2 = self._compute_T_prim_and_thickness(
            rho_total_2_BVY, qL_2, Z_nuc=self.Z_2
        )
        Vx_1, Vy_1 = self._compute_photon_field(
            F_3d_sorted_1_BVY, self.gamma_1, qL_1, Z_nuc=self.Z_1
        )
        Vx_2, Vy_2 = self._compute_photon_field(
            F_3d_sorted_2_BVY, self.gamma_2, qL_2, Z_nuc=self.Z_2
        )
        Egamma_1 = 0.5 * self.Mjpsi * torch.exp(y_flat)
        Egamma_2 = 0.5 * self.Mjpsi * torch.exp(-y_flat)

        def _complex_amplitude_at_current_b():
            a1x, a1y = self._compute_amplitude_at_grid(
                nuclear_ID=1,
                T_prim_target=T_prim_1,
                T_thick_source=T_thick_2,
                Vx_source=Vx_2,
                Vy_source=Vy_2,
                Egamma_target=Egamma_2,
                Ep_target=self.Ep_1,
                A_source=self.A_2,
            )
            a2x, a2y = self._compute_amplitude_at_grid(
                nuclear_ID=2,
                T_prim_target=T_prim_2,
                T_thick_source=T_thick_1,
                Vx_source=Vx_1,
                Vy_source=Vy_1,
                Egamma_target=Egamma_1,
                Ep_target=self.Ep_2,
                A_source=self.A_1,
            )
            return self._coherent_amplitude_q(a1x + a2x, a1y + a2y)

        # _compute_incoherent_total_prob retains two legacy arguments that are
        # not used by its empirical envelope.  Explicit dummies make that
        # non-Good--Walker status visible and avoid inventing alpha centres for
        # the analytic density.
        if incoherent:
            dummy_centres = torch.zeros(BVY, 1, 3, device=device, dtype=dtype)
            dummy_width = torch.ones(BVY, device=device, dtype=dtype)

        def _total_image_at_current_b():
            amplitude_x, amplitude_y, _ = _complex_amplitude_at_current_b()
            image_at_b, p_axis_at_b = self.project_complex_amplitude_intensity_to_output(
                amplitude_x, amplitude_y
            )
            if incoherent:
                incoherent_probability = self._compute_incoherent_total_prob(
                    dummy_centres,
                    dummy_width,
                    y_flat,
                    rho_total_1_BVY,
                    T_thick_1,
                    T_thick_2,
                    Vx_2,
                    Vy_2,
                    Vx_1,
                    Vy_1,
                )
                Pq_X, Pq_Y = _meshgrid_ij(p_axis_at_b, p_axis_at_b)
                q2 = (Pq_X ** 2 + Pq_Y ** 2)[None, :, :]
                incoherent_image = (
                    incoherent_probability.view(BVY, 1, 1)
                    * self.b_slope_incoh
                    / math.pi
                    * torch.exp(-self.b_slope_incoh * q2)
                )
                image_at_b = image_at_b + incoherent_image
            return image_at_b, p_axis_at_b

        image_q_flat, p_axis_q = self._average_over_impact_parameter(
            _total_image_at_current_b
        )
        target_size = int(image_q_flat.shape[-1])
        return image_q_flat.view(B, V_y, target_size, target_size), p_axis_q

    def fixed_b_branch_response_multi_y_from_density(
        self,
        rho_p_1: torch.Tensor,
        rho_total_1: torch.Tensor,
        y_views: torch.Tensor,
        b_values: Sequence[float],
        *,
        rho_p_2: Optional[torch.Tensor] = None,
        rho_total_2: Optional[torch.Tensor] = None,
        normalization_rtol: float = 5.0e-4,
        negativity_atol: float = 1.0e-8,
    ) -> FixedBExternalDensityBranchResponse:
        """Return branch-resolved Ne20 response before any ensemble average.

        This is the external-density counterpart of
        :meth:`fixed_b_complex_amplitude_multi_y`, with the two photon
        directions kept separate.  It additionally returns the two branch
        contributions of the *legacy empirical* incoherent prescription on
        the output momentum grid.  Those real images are a comparison target;
        they are never folded into the complex amplitudes.

        Axis contracts
        --------------
        amplitudes
            ``(batch,n_b,n_y,n_p_full,n_p_full)`` complex.
        empirical branch images
            ``(batch,n_b,n_y,target_size_q,target_size_q)`` real.

        No configuration/orientation average, modulus square of a branch
        mean, Good--Walker variance, or ``2*pi*b*db`` integration occurs here.
        The original ``self.b_imp`` is restored even if a fixed-b evaluation
        raises.
        """

        if not isinstance(y_views, torch.Tensor) or y_views.ndim != 2:
            raise ValueError(
                "fixed_b_branch_response_multi_y_from_density: "
                "y_views must be a (B,V_y) tensor"
            )
        fixed_b = tuple(float(value) for value in b_values)
        if not fixed_b or any(
            not math.isfinite(value) or value < 0.0 for value in fixed_b
        ):
            raise ValueError(
                "fixed-b response nodes must be finite, non-negative and non-empty"
            )
        if len(fixed_b) > 1 and any(
            right <= left for left, right in zip(fixed_b[:-1], fixed_b[1:])
        ):
            raise ValueError("fixed-b response nodes must be strictly increasing")

        rho_p_1, rho_total_1 = self._validate_external_density_pair(
            rho_p_1,
            rho_total_1,
            nuclear_ID=1,
            normalization_rtol=float(normalization_rtol),
            negativity_atol=float(negativity_atol),
        )
        if (rho_p_2 is None) != (rho_total_2 is None):
            raise ValueError("rho_p_2 and rho_total_2 must be supplied together")
        if rho_p_2 is None:
            rho_p_2, rho_total_2 = rho_p_1, rho_total_1
        rho_p_2, rho_total_2 = self._validate_external_density_pair(
            rho_p_2,
            rho_total_2,
            nuclear_ID=2,
            normalization_rtol=float(normalization_rtol),
            negativity_atol=float(negativity_atol),
        )

        B = int(rho_p_1.shape[0])
        if rho_p_2.shape[0] != B or y_views.shape[0] != B:
            raise ValueError(
                "external density batch mismatch: "
                f"rho1={rho_p_1.shape[0]}, rho2={rho_p_2.shape[0]}, "
                f"y_views={y_views.shape[0]}"
            )
        if rho_p_2.device != rho_p_1.device or rho_p_2.dtype != rho_p_1.dtype:
            raise ValueError("both nuclei must share density device and dtype")

        device = rho_p_1.device
        dtype = rho_p_1.dtype
        V_y = int(y_views.shape[1])
        BVY = B * V_y
        F_3d_sorted_1 = self._compute_F_3d(rho_p_1)
        F_3d_sorted_2 = self._compute_F_3d(rho_p_2)
        rho_total_1_BVY = self._expand_to_BVY(rho_total_1, V_y)
        rho_total_2_BVY = self._expand_to_BVY(rho_total_2, V_y)
        F_3d_sorted_1_BVY = self._expand_to_BVY(F_3d_sorted_1, V_y)
        F_3d_sorted_2_BVY = self._expand_to_BVY(F_3d_sorted_2, V_y)
        y_flat = y_views.to(device=device, dtype=dtype).reshape(BVY)

        qL_1 = self.Mjpsi * 0.5 / self.gamma_1 * torch.exp(y_flat)
        qL_2 = self.Mjpsi * 0.5 / self.gamma_2 * torch.exp(-y_flat)
        T_prim_1, T_thick_1 = self._compute_T_prim_and_thickness(
            rho_total_1_BVY, qL_1, Z_nuc=self.Z_1
        )
        T_prim_2, T_thick_2 = self._compute_T_prim_and_thickness(
            rho_total_2_BVY, qL_2, Z_nuc=self.Z_2
        )
        Vx_1, Vy_1 = self._compute_photon_field(
            F_3d_sorted_1_BVY, self.gamma_1, qL_1, Z_nuc=self.Z_1
        )
        Vx_2, Vy_2 = self._compute_photon_field(
            F_3d_sorted_2_BVY, self.gamma_2, qL_2, Z_nuc=self.Z_2
        )
        Egamma_1 = 0.5 * self.Mjpsi * torch.exp(y_flat)
        Egamma_2 = 0.5 * self.Mjpsi * torch.exp(-y_flat)

        dummy_centres = torch.zeros(BVY, 1, 3, device=device, dtype=dtype)
        dummy_width = torch.ones(BVY, device=device, dtype=dtype)
        p_axis = self._p_axis_q.to(device)
        px, py = _meshgrid_ij(p_axis, p_axis)
        q2 = (px * px + py * py)[None, :, :]

        a21x_by_b = []
        a21y_by_b = []
        a12x_by_b = []
        a12y_by_b = []
        empirical_21_by_b = []
        empirical_12_by_b = []
        p_full = None
        original_b = self.b_imp
        try:
            for b_value in fixed_b:
                self.b_imp = b_value
                a21x_grid, a21y_grid = self._compute_amplitude_at_grid(
                    nuclear_ID=1,
                    T_prim_target=T_prim_1,
                    T_thick_source=T_thick_2,
                    Vx_source=Vx_2,
                    Vy_source=Vy_2,
                    Egamma_target=Egamma_2,
                    Ep_target=self.Ep_1,
                    A_source=self.A_2,
                )
                a12x_grid, a12y_grid = self._compute_amplitude_at_grid(
                    nuclear_ID=2,
                    T_prim_target=T_prim_2,
                    T_thick_source=T_thick_1,
                    Vx_source=Vx_1,
                    Vy_source=Vy_1,
                    Egamma_target=Egamma_1,
                    Ep_target=self.Ep_2,
                    A_source=self.A_1,
                )
                a21x, a21y, p_full = self._coherent_amplitude_q(
                    a21x_grid, a21y_grid
                )
                a12x, a12y, _ = self._coherent_amplitude_q(
                    a12x_grid, a12y_grid
                )
                prob_21, prob_12 = self._compute_incoherent_branch_probabilities(
                    dummy_centres,
                    dummy_width,
                    y_flat,
                    rho_total_1_BVY,
                    T_thick_1,
                    T_thick_2,
                    Vx_2,
                    Vy_2,
                    Vx_1,
                    Vy_1,
                )
                envelope = self.b_slope_incoh / math.pi * torch.exp(
                    -self.b_slope_incoh * q2
                )
                a21x_by_b.append(a21x)
                a21y_by_b.append(a21y)
                a12x_by_b.append(a12x)
                a12y_by_b.append(a12y)
                empirical_21_by_b.append(prob_21[:, None, None] * envelope)
                empirical_12_by_b.append(prob_12[:, None, None] * envelope)
        finally:
            self.b_imp = original_b

        assert p_full is not None

        def _stack(values):
            stacked = torch.stack(values, dim=1)
            spatial = stacked.shape[-2:]
            return stacked.view(B, V_y, len(fixed_b), *spatial).permute(
                0, 2, 1, 3, 4
            )

        return FixedBExternalDensityBranchResponse(
            amplitude_2_to_1_x=_stack(a21x_by_b),
            amplitude_2_to_1_y=_stack(a21y_by_b),
            amplitude_1_to_2_x=_stack(a12x_by_b),
            amplitude_1_to_2_y=_stack(a12y_by_b),
            empirical_incoherent_2_to_1=_stack(empirical_21_by_b),
            empirical_incoherent_1_to_2=_stack(empirical_12_by_b),
            p_full_GeV=p_full,
            p_axis_GeV=p_axis,
        )

    def fixed_b_complex_amplitude_multi_y(
        self,
        cluster_pos: Optional[torch.Tensor],
        sigma_c: Optional[Union[torch.Tensor, float]],
        y_views: torch.Tensor,
        b_values: Sequence[float],
        *,
        cluster_pos_2: Optional[torch.Tensor] = None,
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        R_1: Optional[torch.Tensor] = None,
        R_2: Optional[torch.Tensor] = None,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pre-square UPC amplitudes on explicit fixed-b nodes.

        Returns ``(M_x, M_y, p_full)`` with amplitude shape
        ``(batch, n_b, n_y, n_p_full, n_p_full)``.  No impact-parameter
        weights, modulus squares, configuration averages or bin measures are
        applied.  Call :meth:`project_complex_amplitude_intensity_to_output`
        only after forming the required fixed-b complex mean or second moment.
        """
        result = self.forward_multi_y(
            cluster_pos,
            sigma_c=sigma_c,
            y_views=y_views,
            cluster_pos_2=cluster_pos_2,
            sigma_c_2=sigma_c_2,
            density_model_1=density_model_1,
            density_model_2=density_model_2,
            R_1=R_1,
            R_2=R_2,
            incoherent=False,
            brink_antisym=brink_antisym,
            brink_antisym_2=brink_antisym_2,
            _return_complex_amplitude=True,
            _fixed_b_values=b_values,
        )
        return result[0], result[1], result[2]

    def forward_with_orientation_avg_multi_y(
        self,
        cluster_pos: Optional[torch.Tensor],
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y_views: Optional[torch.Tensor] = None,        # ★ (B, V_y)
        cluster_pos_2: Optional[torch.Tensor] = None,
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        n_orientations: int = 20,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
        chunk_size: Optional[int] = None,              # 控制 (B·N) chunk 数
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多 rapidity vectorized + SO(3) 取向平均 (plan §4.3.B.8 做法 B)。

        与 ``forward_with_orientation_avg`` 数学等价、数值一致：
        ``forward_with_orientation_avg_multi_y(... y_views[:, k:k+1] ...)`` 与
        ``forward_with_orientation_avg(... y=y_views[:, k] ...)``（同 generator）
        逐元素一致 (atol ~1e-5)。

        实现：
          - 对 N 个取向 + (B,) batch flatten 到 (B·N) 像 forward_with_orientation_avg
          - 每个 chunk 内用 ``forward_multi_y`` 复用密度 + 3D FFT
          - chunk_size 控制 (B·N) 维分块（与 forward_with_orientation_avg 同义；
            实际单 chunk 的下游开销 ≈ chunk_size·V_y 个 image 卷积）
          - 平均 over N，输出 (B, V_y, H, W)
        """
        if y_views is None or not isinstance(y_views, torch.Tensor) or y_views.dim() != 2:
            raise ValueError(
                f"forward_with_orientation_avg_multi_y: y_views 必须是 (B, V_y) tensor，"
                f"收到 {None if y_views is None else tuple(y_views.shape)}"
            )

        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()

        # ── batch / device / dtype 推断（与 forward_with_orientation_avg 同口径）──
        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "continuous-density orientation averaging is owned by the density provider; "
                "supply rotated tensors to forward_multi_y_from_density()"
            )
        if dm_1 == 'alpha_cluster':
            if cluster_pos is None:
                raise ValueError("alpha_cluster 模式需要 cluster_pos")
            B, n_cl, _ = cluster_pos.shape
            device = cluster_pos.device
            dtype = cluster_pos.dtype
        else:
            if cluster_pos_2 is not None:
                B = cluster_pos_2.shape[0]; device = cluster_pos_2.device; dtype = cluster_pos_2.dtype; n_cl = 0
            elif cluster_pos is not None:
                B = cluster_pos.shape[0]; device = cluster_pos.device; dtype = cluster_pos.dtype; n_cl = cluster_pos.shape[1]
            else:
                B = y_views.shape[0]; device = y_views.device; dtype = y_views.dtype; n_cl = 0

        V_y = int(y_views.shape[1])
        if y_views.shape[0] != B:
            raise ValueError(
                f"y_views.shape[0]={y_views.shape[0]} ≠ batch B={B}"
            )
        N = int(n_orientations)

        # 核2 默认
        if cluster_pos_2 is None and dm_2 == 'alpha_cluster':
            cluster_pos_2 = cluster_pos
        if sigma_c_2 is None:
            sigma_c_2 = sigma_c
        if brink_antisym_2 is None:
            brink_antisym_2 = brink_antisym

        # ★ 把 α-cluster 的质心移到原点 (与 forward_with_orientation_avg 同款修复)。
        #   *_to_cartesian 反解出的坐标第 1 原子在原点 (质心偏离 ~2 fm), 不居中会让
        #   旋转绕原子 1、且 ±b/2 平移给出的有效冲击参数与名义 b 不符。
        _share_cp = cluster_pos_2 is cluster_pos
        if dm_1 == 'alpha_cluster' and cluster_pos is not None:
            cluster_pos = cluster_pos - cluster_pos.mean(dim=1, keepdim=True)
        if dm_2 == 'alpha_cluster' and cluster_pos_2 is not None:
            cluster_pos_2 = cluster_pos if _share_cp else (
                cluster_pos_2 - cluster_pos_2.mean(dim=1, keepdim=True)
            )

        # broadcast sigma_c
        if dm_1 == 'alpha_cluster':
            if sigma_c is None:
                sigma_c = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c, (int, float)):
                sigma_c = torch.full((B,), float(sigma_c), device=device, dtype=dtype)
            elif sigma_c.dim() == 0:
                sigma_c = sigma_c.expand(B).contiguous()
        if dm_2 == 'alpha_cluster':
            if sigma_c_2 is None:
                sigma_c_2 = torch.full((B,), self.default_sigma_c, device=device, dtype=dtype)
            elif isinstance(sigma_c_2, (int, float)):
                sigma_c_2 = torch.full((B,), float(sigma_c_2), device=device, dtype=dtype)
            elif sigma_c_2.dim() == 0:
                sigma_c_2 = sigma_c_2.expand(B).contiguous()

        total = B * N
        T_size = self.target_size_q

        # ─────────────────────────────────────────────
        # 通用路径：α-α / α+WS / WS+WS 都走"flatten BN → forward_multi_y"
        # ─────────────────────────────────────────────
        if dm_1 == 'alpha_cluster' and dm_2 == 'alpha_cluster':
            # α + α：(B, N) 旋转 → (B·N, n_cl, 3)，R_1/R_2=None（已预旋转）
            R_1 = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            r_rot_1 = torch.einsum('bnij,bkj->bnki', R_1, cluster_pos).reshape(total, n_cl, 3)
            if cluster_pos_2 is cluster_pos:
                r_rot_2 = r_rot_1
            else:
                n_cl_2 = cluster_pos_2.shape[1]
                R_2 = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
                r_rot_2 = torch.einsum('bnij,bkj->bnki', R_2, cluster_pos_2).reshape(total, n_cl_2, 3)
            sigma_flat_1 = sigma_c.unsqueeze(1).expand(B, N).reshape(-1)
            sigma_flat_2 = sigma_c_2.unsqueeze(1).expand(B, N).reshape(-1)
            R_1_pass, R_2_pass = None, None
            cp_pass_1, cp_pass_2 = r_rot_1, r_rot_2
        else:
            # 有 WS：cluster_pos=None 时传 None；R_i 透传 (B·N, 3, 3)
            R_1_all = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            if cluster_pos_2 is not None and cluster_pos_2 is not cluster_pos:
                R_2_all = self._sample_uniform_rotations(B, N, device, dtype, generator=generator)
            else:
                R_2_all = R_1_all
            R_1_pass = R_1_all.reshape(total, 3, 3)
            R_2_pass = R_2_all.reshape(total, 3, 3)
            if dm_1 == 'alpha_cluster':
                cp_pass_1 = cluster_pos.unsqueeze(1).expand(
                    B, N, *cluster_pos.shape[1:]
                ).reshape(total, *cluster_pos.shape[1:])
                sigma_flat_1 = sigma_c.unsqueeze(1).expand(B, N).reshape(-1)
            else:
                cp_pass_1 = None
                sigma_flat_1 = None
            if dm_2 == 'alpha_cluster':
                cp_pass_2 = cluster_pos_2.unsqueeze(1).expand(
                    B, N, *cluster_pos_2.shape[1:]
                ).reshape(total, *cluster_pos_2.shape[1:])
                sigma_flat_2 = sigma_c_2.unsqueeze(1).expand(B, N).reshape(-1)
            else:
                cp_pass_2 = None
                sigma_flat_2 = None

        # y_views 在 (B·N) 维度上重复：每个取向共享一组 V_y 个 y
        # → (B, N, V_y) → (B·N, V_y)
        y_views_BN = y_views.unsqueeze(1).expand(B, N, V_y).reshape(total, V_y)

        # ── chunked forward_multi_y over (B·N) 维 ──
        if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
            chunks = [(0, total)]
        else:
            chunks = [(s, min(s + chunk_size, total))
                      for s in range(0, total, chunk_size)]

        outputs = []
        p_axis_q = None
        for s, e in chunks:
            img_chunk, p_axis_q = self.forward_multi_y(
                cluster_pos=(cp_pass_1[s:e] if cp_pass_1 is not None else None),
                sigma_c=(sigma_flat_1[s:e] if sigma_flat_1 is not None else None),
                y_views=y_views_BN[s:e],
                cluster_pos_2=(cp_pass_2[s:e] if cp_pass_2 is not None else None),
                sigma_c_2=(sigma_flat_2[s:e] if sigma_flat_2 is not None else None),
                density_model_1=dm_1, density_model_2=dm_2,
                R_1=(R_1_pass[s:e] if R_1_pass is not None else None),
                R_2=(R_2_pass[s:e] if R_2_pass is not None else None),
                incoherent=incoherent,
                brink_antisym=brink_antisym,
                brink_antisym_2=brink_antisym_2,
            )                                                # (chunk, V_y, T, T)
            outputs.append(img_chunk)
        image_all = torch.cat(outputs, dim=0)                 # (B·N, V_y, T, T)

        # ── 取向平均 over N ──
        image_avg = image_all.view(B, N, V_y, T_size, T_size).mean(dim=1)
        return image_avg, p_axis_q

    def forward_from_zmatrix_multi_y(
        self,
        z: Optional[torch.Tensor],                    # (B, 6) 核1 z-matrix
        sigma_c: Optional[Union[torch.Tensor, float]] = None,
        y_views: Optional[torch.Tensor] = None,        # ★ (B, V_y)
        z_2: Optional[torch.Tensor] = None,
        sigma_c_2: Optional[Union[torch.Tensor, float]] = None,
        density_model_1: Optional[str] = None,
        density_model_2: Optional[str] = None,
        n_orientations: int = 20,
        incoherent: bool = False,
        brink_antisym: bool = True,
        brink_antisym_2: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2 多 rapidity 主入口：z-matrix + (B, V_y) rapidity → 旋转平均后的
        image_q_total (B, V_y, H, W)。与 ``forward_from_zmatrix`` 数学等价。

        全程可微。
        """
        from .cluster_geometry import zmatrix_to_cartesian

        dm_1 = (density_model_1 or self._density_model_1).strip().lower()
        dm_2 = (density_model_2 or self._density_model_2).strip().lower()

        if "external_density" in (dm_1, dm_2):
            raise ValueError(
                "external continuous densities cannot be reconstructed from a z-matrix; "
                "use forward_multi_y_from_density()"
            )
        if dm_1 == 'alpha_cluster':
            if z is None or z.dim() != 2 or z.shape[1] != 6:
                raise ValueError(f"alpha_cluster 核1 需要 z: (B, 6)，收到 {z}")
            cluster_pos = zmatrix_to_cartesian(z)
        else:
            cluster_pos = None

        if dm_2 == 'alpha_cluster':
            if z_2 is not None:
                if z_2.dim() != 2 or z_2.shape[1] != 6:
                    raise ValueError(f"z_2 期望 (B, 6)，收到 {tuple(z_2.shape)}")
                cluster_pos_2 = zmatrix_to_cartesian(z_2)
            elif z is not None:
                cluster_pos_2 = zmatrix_to_cartesian(z)
            else:
                raise ValueError("alpha_cluster 核2 需要 z_2 或 z")
        else:
            cluster_pos_2 = None

        return self.forward_with_orientation_avg_multi_y(
            cluster_pos,
            sigma_c=sigma_c, y_views=y_views,
            cluster_pos_2=cluster_pos_2, sigma_c_2=sigma_c_2,
            density_model_1=dm_1, density_model_2=dm_2,
            n_orientations=n_orientations,
            incoherent=incoherent,
            brink_antisym=brink_antisym, brink_antisym_2=brink_antisym_2,
            chunk_size=chunk_size,
            generator=generator,
        )


# ============================================================
# CLI / smoke test 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/upc_forward.yaml")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--incoherent", action="store_true")
    parser.add_argument("--asymmetric", action="store_true",
                        help="测试非对称碰撞模式（核1=WS, 核2=alpha_cluster）")
    parser.add_argument("--ws-both", action="store_true",
                        help="测试双 WS 模式")
    args = parser.parse_args()

    model = UPCForwardModel(args.config).to(args.device)
    B = args.batch

    if args.ws_both:
        print("=== 测试双 WS 模式 ===")
        y = torch.zeros(B, device=args.device)
        image_q, p_axis = model(
            None, None, y,
            cluster_pos_2=None, sigma_c_2=None,
            density_model_1='woods_saxon',
            density_model_2='woods_saxon',
        )
        print(f"[双WS] image_q_total shape: {image_q.shape}, dtype: {image_q.dtype}")
        print(f"[双WS] p_axis_q range: [{p_axis.min():.3f}, {p_axis.max():.3f}]")
        print(f"[双WS] image_q_total sum (B=0): {image_q[0].sum().item():.4e}")

    elif args.asymmetric:
        print("=== 测试非对称碰撞（核1=WS, 核2=alpha_cluster）===")
        # 核1 使用 WS（不需要 cluster_pos）
        # 核2 使用 alpha_cluster
        cluster_pos_2 = torch.randn(B, 4, 3, device=args.device) * 2.0
        sigma_c_2 = torch.full((B,), 1.36, device=args.device)
        y = torch.zeros(B, device=args.device)

        image_q, p_axis = model(
            None, None, y,
            cluster_pos_2=cluster_pos_2, sigma_c_2=sigma_c_2,
            density_model_1='woods_saxon',
            density_model_2='alpha_cluster',
        )
        print(f"[非对称] image_q_total shape: {image_q.shape}, dtype: {image_q.dtype}")
        print(f"[非对称] p_axis_q range: [{p_axis.min():.3f}, {p_axis.max():.3f}]")
        print(f"[非对称] image_q_total sum (B=0): {image_q[0].sum().item():.4e}")
    else:
        print("=== 测试对称碰撞模式 ===")
        cluster_pos = torch.randn(B, 4, 3, device=args.device) * 2.0
        sigma_c = torch.full((B,), 1.36, device=args.device)
        y = torch.zeros(B, device=args.device)

        image_q, p_axis = model(cluster_pos, sigma_c, y, incoherent=args.incoherent)
        print(f"[对称] image_q_total shape: {image_q.shape}, dtype: {image_q.dtype}")
        print(f"[对称] p_axis_q range: [{p_axis.min():.3f}, {p_axis.max():.3f}]")
        print(f"[对称] image_q_total sum (B=0): {image_q[0].sum().item():.4e}")

    print(f"image_q_total max (B=0): {image_q[0].max().item():.4e}")
    print(f"NaN count: {torch.isnan(image_q).sum().item()}")
