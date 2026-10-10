"""
utils/cluster_geometry.py
=========================

α-团簇核 (任意 N≥2) 的几何工具：

1. **元素 registry** — Be8 / C12 / O16 / Ne20 的 (Z, A, n_clusters,
   cluster_radius_fm) 表，配套 ``reference_geometry`` 给出参考构型。
2. ``canonicalize_clusters(r)`` — 把 (B, N, 3) 笛卡尔团簇坐标按"距质心
   距离从近到远"排序，商掉 S_N 置换对称性（任意 N）。
3. ``cartesian_to_zmatrix(r)`` — 任意 N 的笛卡尔 → Z-matrix invariants。
4. ``zmatrix_to_cartesian(z, n_clusters=None)`` — 反向，输出 *标准 frame*
   下的笛卡尔，全程可微，是 NF inference 路径的关键。``n_clusters`` 缺省时
   按 z 的最后一维自动推断。
5. ``cluster_rms_radius(r)``、``rescale_to_radius(r, R_target)`` —
   "高维球壳" 半径约束的工具（plan 扩展节）。

Z-matrix 编码（与 phase2_6D_NF_plan.md §3.3 对齐，并自然外推到任意 N）：

    atom 1 : (0, 0, 0)                                                    ─ 0 DOF (去平移)
    atom 2 : (d_12, 0, 0)                                                 ─ 1 DOF (d_12)
    atom 3 : (d_13 cos θ_213, d_13 sin θ_213, 0)         y_3 > 0          ─ 2 DOF (d_13, θ_213)
    atom k≥4 : (d_1k cos θ_21k,
               d_1k sin θ_21k · cos φ_321k,
               d_1k sin θ_21k · sin φ_321k)                               ─ 3 DOF / atom

总不变量数：
    N=2  → 1
    N=3  → 3
    N=4  → 6
    N=k≥3 → 3k - 6

Z-matrix 列顺序（与 ``zmatrix_dim`` / ``ZMatrixBijection`` 一致）::

    [d_12,
     d_13, θ_213,                              # if N >= 3
     d_14, θ_214, φ_3214,                      # if N >= 4
     d_15, θ_215, φ_3215, ...]                 # if N >= 5

元素 registry（在每元素自身参考几何下，``cluster_rms_radius == cluster_radius_fm``）：

    Be8   N=2  Z=4   A=8    R_target ≈ 1.60 fm   (α-α 哑铃)
    C12   N=3  Z=6   A=12   R_target ≈ 1.91 fm   (平面正三角形)
    O16   N=4  Z=8   A=16   R_target ≈ 2.02 fm   (正四面体)
    Ne20  N=5  Z=10  A=20   R_target ≈ 2.25 fm   (三角双锥)

向后兼容
--------
- ``cartesian_to_zmatrix(r)`` 接受 (B, 4, 3) 时仍输出 (B, 6) — 与历史调用兼容
- ``zmatrix_to_cartesian(z)`` 接受 (B, 6) 时返回 (B, 4, 3) — 历史调用兼容
  （未传 n_clusters 时按 z.shape[1] 自动推断）

可微性
------
- ``canonicalize_clusters``: argsort 不可微（仅用于数据预处理）。
- ``cartesian_to_zmatrix``: 数学上的 acos/atan2 可微但在退化几何处梯度发散；
  也只用于数据预处理。
- ``zmatrix_to_cartesian``: 全程可微，控制流仅在 ``__init__`` 阶段（按
  ``n_clusters`` 选分支），张量运算无分支。

"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ============================================================
# 元素 registry
# ============================================================

# 每元素的内禀 metadata。``cluster_radius_fm`` = 该元素在参考几何下的
# RMS(|r_i - r_cm|)，单位 fm；``reference_side_fm`` 是相邻 cluster 距离。
ELEMENT_REGISTRY = {
    'Be8':  {
        'n_clusters': 2, 'Z': 4,  'A': 8,
        'cluster_radius_fm': 1.60, 'reference_side_fm': 3.20,
        'reference_shape': 'dumbbell',
    },
    'C12':  {
        'n_clusters': 3, 'Z': 6,  'A': 12,
        'cluster_radius_fm': 1.91, 'reference_side_fm': 3.30,
        'reference_shape': 'equilateral_triangle',
    },
    'O16':  {
        'n_clusters': 4, 'Z': 8,  'A': 16,
        'cluster_radius_fm': 2.02, 'reference_side_fm': 3.30,
        'reference_shape': 'tetrahedron',
    },
    'Ne20': {
        'n_clusters': 5, 'Z': 10, 'A': 20,
        'cluster_radius_fm': 2.25, 'reference_side_fm': 3.30,
        'reference_shape': 'trigonal_bipyramid',
    },
}


def element_info(element: str) -> dict:
    """返回元素 metadata 的 *拷贝*（避免外部改 registry 全局变量）。"""
    if element not in ELEMENT_REGISTRY:
        raise ValueError(
            f"Unknown element '{element}'. 可用: {list(ELEMENT_REGISTRY)}"
        )
    return dict(ELEMENT_REGISTRY[element])


def zmatrix_dim(n_clusters: int) -> int:
    """N → Z-matrix invariant 个数：N=2 → 1, 否则 3N - 6 (N≥3)。"""
    if n_clusters < 2:
        raise ValueError(f"n_clusters 必须 >= 2，收到 {n_clusters}")
    if n_clusters == 2:
        return 1
    return 3 * n_clusters - 6


def infer_n_clusters(z_dim: int) -> int:
    """从 Z-matrix invariant 维数反推 n_clusters。"""
    if z_dim == 1:
        return 2
    if z_dim == 3:
        return 3
    if z_dim >= 6 and (z_dim + 6) % 3 == 0:
        return (z_dim + 6) // 3
    raise ValueError(
        f"Cannot infer n_clusters from z dim={z_dim}. "
        f"合法 dim: 1 (N=2), 3 (N=3), 6 (N=4), 9 (N=5), ..."
    )


# ============================================================
# 参考几何（numpy；rescale 到 cluster_radius_fm）
# ============================================================

def _raw_reference_geometry(element: str) -> np.ndarray:
    """元素的"裸"参考构型，构造时各取自然 side 长度。后面再 rescale。"""
    info = element_info(element)
    side = info['reference_side_fm']
    shape = info['reference_shape']
    if shape == 'dumbbell':
        d = side
        return np.array([
            [-d / 2.0, 0.0, 0.0],
            [+d / 2.0, 0.0, 0.0],
        ], dtype=np.float64)
    if shape == 'equilateral_triangle':
        r = side / math.sqrt(3.0)
        return np.array([
            [r,                          0.0,                       0.0],
            [-r / 2.0, r * math.sqrt(3.0) / 2.0,                    0.0],
            [-r / 2.0, -r * math.sqrt(3.0) / 2.0,                   0.0],
        ], dtype=np.float64)
    if shape == 'tetrahedron':
        # 边长 = side 的正四面体（顶点 (±1, ±1, ±1) 中两两异号的 4 个）
        base = np.array([
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
        ], dtype=np.float64)
        return base * (side / math.sqrt(8.0))
    if shape == 'trigonal_bipyramid':
        # 全等边的三角双锥：赤道 3 顶点 + 轴向 ±z 顶点；所有相邻距离 = side
        r_eq = side / math.sqrt(3.0)
        z_ax = side * math.sqrt(2.0 / 3.0)
        return np.array([
            [r_eq,                          0.0,                       0.0],
            [-r_eq / 2.0, r_eq * math.sqrt(3.0) / 2.0,                 0.0],
            [-r_eq / 2.0, -r_eq * math.sqrt(3.0) / 2.0,                0.0],
            [0.0,                           0.0,                      +z_ax],
            [0.0,                           0.0,                      -z_ax],
        ], dtype=np.float64)
    raise ValueError(f"未知 reference_shape={shape}")


def reference_geometry(element: str) -> np.ndarray:
    """
    返回元素的参考构型 (N, 3) numpy，已 *精确 rescale* 到
    ``cluster_radius_fm``（与 registry 一致）并去掉质心。

    给数据生成 prior 用——围绕这个构型加 Gaussian 噪声即可。
    """
    info = element_info(element)
    positions = _raw_reference_geometry(element)
    com = positions.mean(axis=0, keepdims=True)
    positions = positions - com
    r_current = float(np.sqrt(((positions) ** 2).sum(axis=-1).mean()))
    if r_current > 1.0e-12:
        positions = positions * (info['cluster_radius_fm'] / r_current)
    return positions


# ============================================================
# canonicalize (任意 N，与历史 (B, 4, 3) 完全兼容)
# ============================================================

def canonicalize_clusters(r: torch.Tensor) -> torch.Tensor:
    """
    按"离质心距离从近到远"排序，商掉 S_N 置换对称性。**N >= 4 时**额外
    通过反射破 parity gauge：若第一根二面角 φ_3214 < 0，则把构型整体反射
    通过原子 (0, 1, 2) 决定的平面，使 φ_3214 > 0。

    动机
    ----
    UPC 散射在 parity 下不变（QCD/QED parity 守恒），所以构型 r 与其镜像
    P·r 给出同一张图像。但在原 canonical 实现里两个镜像分别映射到 z-matrix
    的 (..., +φ_3214, ±φ_3215, ...) 与 (..., -φ_3214, ∓φ_3215, ...)，
    成为两个不同的 GT。NF 因此被迫学一个 φ 对称的双峰后验，SBC 在 φ_3214
    维度上必然产生双峰直方图（rank ~25% 与 ~75%）。

    通过把整族 chirality 折到一侧（φ_3214 ∈ (0, π)），训练数据失去 parity
    冗余，NF 只需学单峰 φ 后验，SBC 收敛到 uniform。**配套 patch**：
    ``models/cluster_bijection.py`` 把 φ_3214 维域从 (-π, π) 改成 (0, π)。

    Args
    ----
    r : (B, N, 3) for any N >= 2

    Returns
    -------
    r_canonical : (B, N, 3)
        N >= 4 时已破 chirality（φ_3214 ∈ (0, π)）。
    """
    if r.dim() != 3 or r.shape[2] != 3:
        raise ValueError(
            f"canonicalize_clusters 期望 (B, N, 3)，收到 {tuple(r.shape)}"
        )
    N = r.shape[1]
    if N < 2:
        raise ValueError(f"canonicalize_clusters 需要 N >= 2，收到 N={N}")

    com = r.mean(dim=1, keepdim=True)            # (B, 1, 3)
    dist_sq = ((r - com) ** 2).sum(dim=-1)        # (B, N)
    order = dist_sq.argsort(dim=1)   # (B, N) long
    r_canonical = torch.gather(r, dim=1, index=order.unsqueeze(-1).expand(-1, -1, 3))

    # ★ 破 chirality（N >= 4 时）
    if N >= 4:
        r_canonical = _break_chirality_first_dihedral(r_canonical)

    return r_canonical


def _break_chirality_first_dihedral(r: torch.Tensor) -> torch.Tensor:
    """
    若第一根二面角 φ_3214 < 0，则把构型整体反射通过 (atom 0, 1, 2) 决定
    的平面。原子 0/1/2 在平面内、反射后不变（仅有 1e-15 量级浮点漂移）；
    原子 3+ 被反射到另一侧，所有 φ_321k 同时翻号。

    Args
    ----
    r : (B, N, 3)，N >= 4。已按 COM 距离排序。
    """
    # 第一根 dihedral φ_3214 (atoms 2, 1, 0, 3，注意 _dihedral_from_atoms
    # 签名是 (a, b, c, d) 表示 (a-b)-(b-c)-(c-d) 的二面角，对应
    # cartesian_to_zmatrix 里 `_dihedral_from_atoms(r3, r2, r1, rk)` 的约定)
    phi_3214 = _dihedral_from_atoms(r[:, 2], r[:, 1], r[:, 0], r[:, 3])  # (B,)

    # 平面 (0, 1, 2) 的法向量
    v01 = r[:, 1] - r[:, 0]
    v02 = r[:, 2] - r[:, 0]
    normal = torch.cross(v01, v02, dim=-1)                # (B, 3)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp_min(_EPS)

    # 全部原子做 Householder 反射 r' = r - 2·((r - r_0)·n̂)·n̂
    r1 = r[:, 0:1]                                        # (B, 1, 3)
    offsets = r - r1                                      # (B, N, 3)
    proj = (offsets * normal.unsqueeze(1)).sum(-1, keepdim=True)   # (B, N, 1)
    reflected = r - 2.0 * proj * normal.unsqueeze(1)      # (B, N, 3)

    # 只在 φ_3214 < 0 的样本上取反射构型
    flip = (phi_3214 < 0).reshape(-1, 1, 1)               # (B, 1, 1) bool
    return torch.where(flip, reflected, r)


# ============================================================
# Cartesian → Z-matrix (任意 N)
# ============================================================

_PI = math.pi
_EPS = 1.0e-12


def cartesian_to_zmatrix(r: torch.Tensor) -> torch.Tensor:
    """
    任意 N≥2 的笛卡尔团簇坐标转换成 Z-matrix invariants。

    Args
    ----
    r : (B, N, 3)。**通常应先经过 canonicalize_clusters。**

    Returns
    -------
    z : (B, zmatrix_dim(N))
    """
    if r.dim() != 3 or r.shape[2] != 3:
        raise ValueError(
            f"cartesian_to_zmatrix 期望 (B, N, 3)，收到 {tuple(r.shape)}"
        )
    N = r.shape[1]
    if N < 2:
        raise ValueError(f"cartesian_to_zmatrix 需要 N >= 2，收到 N={N}")

    # ── d_12 (N >= 2) ──
    r1 = r[:, 0]
    r2 = r[:, 1]
    v12 = r2 - r1
    d12 = torch.norm(v12, dim=-1).clamp_min(_EPS)
    invariants = [d12]
    if N == 2:
        return torch.stack(invariants, dim=-1)

    # ── d_13, θ_213 (N >= 3) ──
    r3 = r[:, 2]
    v13 = r3 - r1
    d13 = torch.norm(v13, dim=-1).clamp_min(_EPS)
    cos213 = (v12 * v13).sum(dim=-1) / (d12 * d13)
    cos213 = cos213.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta213 = torch.acos(cos213)
    invariants += [d13, theta213]
    if N == 3:
        return torch.stack(invariants, dim=-1)

    # ── (d_1k, θ_21k, φ_321k) for k = 4, ..., N ──
    for k_idx in range(3, N):                     # 0-indexed; k_idx=3 ↔ 第 4 个 atom
        rk = r[:, k_idx]
        v1k = rk - r1
        d1k = torch.norm(v1k, dim=-1).clamp_min(_EPS)
        cos21k = (v12 * v1k).sum(dim=-1) / (d12 * d1k)
        cos21k = cos21k.clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta21k = torch.acos(cos21k)
        phi321k = _dihedral_from_atoms(r3, r2, r1, rk)
        invariants += [d1k, theta21k, phi321k]

    return torch.stack(invariants, dim=-1)


def _dihedral_from_atoms(a: torch.Tensor,
                         b: torch.Tensor,
                         c: torch.Tensor,
                         d: torch.Tensor) -> torch.Tensor:
    """
    四原子 (a, b, c, d) 的二面角，与 ``zmatrix_to_cartesian`` 的 synthesis 自洽。

    Synthesis 约定 (atom k≥4)::
        r_k = (d_1k cos θ_21k,
               d_1k sin θ_21k · cos φ_321k,
               d_1k sin θ_21k · sin φ_321k)
    ⇒ φ ∈ (-π, π], φ > 0 ↔ z_k > 0
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    b2_norm = b2 / torch.norm(b2, dim=-1, keepdim=True).clamp_min(_EPS)

    norm_n1n2 = (torch.norm(n1, dim=-1) *
                 torch.norm(n2, dim=-1)).clamp_min(_EPS)

    cos_phi = (n1 * n2).sum(dim=-1) / norm_n1n2
    # 取负号以匹配 synthesis 约定（手算 + 单元测试）
    sin_phi = -(torch.cross(n1, n2, dim=-1) * b2_norm).sum(dim=-1) / norm_n1n2

    return torch.atan2(sin_phi, cos_phi)


# ============================================================
# Z-matrix → Cartesian (任意 N，标准 frame)
# ============================================================

def zmatrix_to_cartesian(
    z: torch.Tensor,
    n_clusters: Optional[int] = None,
) -> torch.Tensor:
    """
    任意 N≥2 的 Z-matrix invariants → *标准 frame* 下笛卡尔。全程可微。

    Args
    ----
    z : (B, zmatrix_dim(N))
    n_clusters : 若为 None，按 z.shape[1] 自动推断

    Returns
    -------
    r : (B, N, 3)
        r_1 = (0, 0, 0)
        r_2 = (d_12, 0, 0)
        r_3 = (d_13 cos θ_213, d_13 sin θ_213, 0)
        r_k≥4 = (d_1k cos θ_21k,
                 d_1k sin θ_21k · cos φ_321k,
                 d_1k sin θ_21k · sin φ_321k)
    """
    if z.dim() != 2:
        raise ValueError(
            f"zmatrix_to_cartesian 期望 (B, n_inv)，收到 {tuple(z.shape)}"
        )
    n_inv = z.shape[1]
    if n_clusters is None:
        n_clusters = infer_n_clusters(n_inv)
    expected = zmatrix_dim(n_clusters)
    if n_inv != expected:
        raise ValueError(
            f"z dim={n_inv} 与 n_clusters={n_clusters} 不一致 (期望 {expected})"
        )

    B = z.shape[0]
    device = z.device
    dtype = z.dtype
    zeros = torch.zeros(B, device=device, dtype=dtype)

    # r_1 = origin
    r1 = torch.stack([zeros, zeros, zeros], dim=-1)

    # r_2 = (d_12, 0, 0)
    d12 = z[:, 0]
    r2 = torch.stack([d12, zeros, zeros], dim=-1)
    if n_clusters == 2:
        return torch.stack([r1, r2], dim=1)

    # r_3 = (d_13 cos θ_213, d_13 sin θ_213, 0)
    d13 = z[:, 1]
    theta213 = z[:, 2]
    sin213 = torch.sin(theta213)
    cos213 = torch.cos(theta213)
    r3 = torch.stack([d13 * cos213, d13 * sin213, zeros], dim=-1)
    if n_clusters == 3:
        return torch.stack([r1, r2, r3], dim=1)

    cluster_list = [r1, r2, r3]
    # atom k = 4, 5, ..., n_clusters；每个三元组 (d_1k, θ_21k, φ_321k)
    for k_idx in range(3, n_clusters):     # 0-indexed; k_idx=3 ↔ 第 4 个 atom
        offset = 3 + 3 * (k_idx - 3)        # 该 atom 的三元组起始列
        d1k = z[:, offset]
        theta = z[:, offset + 1]
        phi = z[:, offset + 2]
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin_p = torch.sin(phi)
        cos_p = torch.cos(phi)
        rk = torch.stack([
            d1k * cos_t,
            d1k * sin_t * cos_p,
            d1k * sin_t * sin_p,
        ], dim=-1)
        cluster_list.append(rk)
    return torch.stack(cluster_list, dim=1)


# ============================================================
# 半径"高维球壳"工具
# ============================================================

def cluster_rms_radius(r: torch.Tensor) -> torch.Tensor:
    """
    (B, N, 3) → (B,) 团簇位置的 rms 半径（相对质心）：
        R = sqrt( (1/N) · Σ_i ‖r_i − r_cm‖² )
    """
    if r.dim() != 3 or r.shape[2] != 3:
        raise ValueError(
            f"cluster_rms_radius 期望 (B, N, 3)，收到 {tuple(r.shape)}"
        )
    com = r.mean(dim=1, keepdim=True)
    sq = ((r - com) ** 2).sum(dim=-1).mean(dim=-1)
    return torch.sqrt(sq.clamp_min(0.0))


def rescale_to_radius(
    r: torch.Tensor,
    target_radius,
) -> torch.Tensor:
    """
    线性缩放 (B, N, 3) 使 ``cluster_rms_radius == target_radius``。
    几何形状不变，仅尺度变。**先减质心**，结果 COM 在原点。

    Args
    ----
    target_radius : float / (B,) tensor / (B,) numpy   每 batch 元素的目标半径
    """
    if r.dim() != 3 or r.shape[2] != 3:
        raise ValueError(
            f"rescale_to_radius 期望 (B, N, 3)，收到 {tuple(r.shape)}"
        )
    com = r.mean(dim=1, keepdim=True)
    r_centered = r - com
    cur = cluster_rms_radius(r_centered)                              # (B,)
    if not torch.is_tensor(target_radius):
        target_radius = torch.full_like(cur, float(target_radius))
    else:
        target_radius = target_radius.to(device=cur.device, dtype=cur.dtype).reshape(-1)
    scale = (target_radius / cur.clamp_min(_EPS)).view(-1, 1, 1)
    return r_centered * scale


# ============================================================
# 自检工具
# ============================================================

def round_trip_error(z: torch.Tensor,
                     n_clusters: Optional[int] = None) -> torch.Tensor:
    """
    z → cartesian → z' 的 round-trip 误差。理论上应该 < 1e-5 (float32)。
    单元测试 / sanity 用。
    """
    r = zmatrix_to_cartesian(z, n_clusters=n_clusters)
    z_back = cartesian_to_zmatrix(r)
    return (z - z_back).abs().max(dim=-1).values   # (B,)


# ============================================================
# ★ DV (pairwise-distance + signed-volume) 表征 — 与 Z-matrix 并行存在
# ============================================================
#
# 动机（见对话 + 后续 plan §3.x 备注）：
#   Z-matrix + canonical_ordering 在 O16 正四面体邻域有 S_4 tie 奇点
#   (4 个 d_i 数值上相等，sort 不稳定)，且 chirality 要靠 Householder
#   反射补丁。DV 表征：6 个 pairwise distance d_ij + 1 个 signed
#   volume V，在 S_4 下分量按"原子置换 → 对置换"线性作用 (查表)，
#   在 parity 下 V 翻号；orbit-marginalized NLL 在 tie loci 自动退化。
#
# 表征维度
#   N = 2 → dv_dim = 1 + 1 = 2  (含 V 退化为 0)
#   N = 3 → dv_dim = 3 + 1 = 4  (V 始终为 0，三点共面)
#   N = 4 → dv_dim = 6 + 1 = 7  ★ O16 主用场景
#   N = 5 → dv_dim = 10 + 1 = 11
#   一般 N → dv_dim = C(N, 2) + 1
#
# 落地约定：
#   - 距离按 *固定字典序* 排列，与原子标号 (1, 2, ..., N) 绑定：
#       N=4 → (d12, d13, d14, d23, d24, d34)
#   - V 用前 4 个原子构造 (atoms 1, 2, 3, 4)，前 3 个原子组成参考平面：
#       V = (1/6) det(r2 − r1, r3 − r1, r4 − r1)         (fm³)
#   - 不做 canonical ordering、不破 chirality；S_4 + parity 在
#     loss 端用 orbit log-sum-exp 处理 (见 models/flow_loss.py)。
#
# 反演 (dv → cartesian) 闭式：参考 plan 对话里给出的 4-point MDS：
#   r1 = (0, 0, 0)
#   r2 = (d12, 0, 0)
#   r3 = (x3, y3, 0)            with x3, y3 由 d13, d23 闭式
#   r4 = (x4, y4, z4)            with z4 的符号 = sign(V)
# 全程可微，是 NF 输出 → UPC forward 的链路。

from math import comb as _comb


def dv_dim(n_clusters: int) -> int:
    """N → DV 表征的维数 (n_pairs + 1)。"""
    if n_clusters < 2:
        raise ValueError(f"n_clusters 必须 >= 2，收到 {n_clusters}")
    return _comb(n_clusters, 2) + 1


def _pair_index_list(n_clusters: int) -> list:
    """
    返回 [(i, j) for i < j]，字典序。i, j 是 0-indexed atom label。
    """
    pairs = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            pairs.append((i, j))
    return pairs


def cartesian_to_dv(r: torch.Tensor) -> torch.Tensor:
    """
    (B, N, 3) → (B, dv_dim(N)) DV 不变量。

    输出列顺序：
        [d_{0,1}, d_{0,2}, ..., d_{N-2,N-1}, V]

    其中 V 用前 4 原子 (若 N >= 4) 算 signed volume；N < 4 时 V = 0。
    **不做** canonical_ordering（DV 在 loss 端用 orbit 处理）；也不破 chirality
    （V 的符号即 chirality 信号）。

    可微（atom 坐标 → 距离 / det 都可微），但在两个原子重合处 d → 0 时梯度发散，
    与 cartesian_to_zmatrix 行为一致。
    """
    if r.dim() != 3 or r.shape[2] != 3:
        raise ValueError(
            f"cartesian_to_dv 期望 (B, N, 3)，收到 {tuple(r.shape)}"
        )
    N = r.shape[1]
    if N < 2:
        raise ValueError(f"cartesian_to_dv 需要 N >= 2，收到 N={N}")

    pairs = _pair_index_list(N)
    dists = []
    for (i, j) in pairs:
        v = r[:, j] - r[:, i]
        dists.append(torch.norm(v, dim=-1).clamp_min(_EPS))    # (B,)
    d_vec = torch.stack(dists, dim=-1)                          # (B, n_pairs)

    if N >= 4:
        # signed volume by atoms (0, 1, 2, 3)
        v01 = r[:, 1] - r[:, 0]
        v02 = r[:, 2] - r[:, 0]
        v03 = r[:, 3] - r[:, 0]
        # det([v01, v02, v03]) / 6
        # 用 cross + dot 实现（与 torch.det 等价但更明确，且对小 batch 更快）
        cross_12 = torch.cross(v01, v02, dim=-1)
        V_signed = (cross_12 * v03).sum(dim=-1) / 6.0           # (B,)
    else:
        V_signed = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)

    return torch.cat([d_vec, V_signed.unsqueeze(-1)], dim=-1)   # (B, n_pairs + 1)


def dv_to_cartesian(
    dv: torch.Tensor,
    n_clusters: Optional[int] = None,
) -> torch.Tensor:
    """
    (B, dv_dim(N)) → (B, N, 3) 在 *标准 frame* 下笛卡尔，全程可微。

    标准 frame:
        r_1 = (0, 0, 0)
        r_2 = (d12, 0, 0)
        r_3 = (x3, y3, 0)            y3 >= 0
        r_4 = (x4, y4, z4)            sign(z4) = sign(V)

    这是 plan 对话里给出的 4-point closed-form MDS。当前**只支持 N = 4**
    （O16）；N != 4 触发 NotImplementedError，等 Ne20 上时再扩。

    Args
    ----
    dv : (B, dv_dim(N)) — 列 (d12, d13, d14, d23, d24, d34, V)（N=4 的情形）
    n_clusters : 若 None 按 dv.shape[1] 自动推断；通过解 C(N,2)+1 = dv_dim 反推。

    Notes
    -----
    数值稳定性：
      - y3² = d13² - x3²  在 r3 与 r2 共线时为 0 → 加 ``_EPS`` floor
      - z4² 同理，并把符号取自 V (avoid sign-loss when |V| 极小)
    """
    if dv.dim() != 2:
        raise ValueError(f"dv_to_cartesian 期望 (B, dv_dim)，收到 {tuple(dv.shape)}")
    n_in = dv.shape[1]
    if n_clusters is None:
        n_clusters = _infer_n_clusters_from_dv(n_in)
    expected = dv_dim(n_clusters)
    if n_in != expected:
        raise ValueError(
            f"dv dim={n_in} 与 n_clusters={n_clusters} 不一致 (期望 {expected})"
        )
    if n_clusters != 4:
        raise NotImplementedError(
            f"dv_to_cartesian 当前只实现 N=4 (O16)；收到 N={n_clusters}。"
            "C12 (N=3) 平面构型反演与 Ne20 (N=5) 5-point MDS 待补。"
        )

    B = dv.shape[0]
    device = dv.device
    dtype = dv.dtype
    zeros = torch.zeros(B, device=device, dtype=dtype)

    d12, d13, d14, d23, d24, d34, V = (
        dv[:, 0], dv[:, 1], dv[:, 2], dv[:, 3], dv[:, 4], dv[:, 5], dv[:, 6]
    )

    # r1 = origin
    r1 = torch.stack([zeros, zeros, zeros], dim=-1)
    # r2 = (d12, 0, 0)
    r2 = torch.stack([d12, zeros, zeros], dim=-1)

    # r3 in xy-plane:
    #   x3 = (d12² + d13² - d23²) / (2 d12)
    #   y3 = sqrt(d13² - x3²)            (y3 >= 0 是标号约定)
    d12_safe = d12.clamp_min(_EPS)
    x3 = (d12 ** 2 + d13 ** 2 - d23 ** 2) / (2.0 * d12_safe)
    y3_sq = d13 ** 2 - x3 ** 2
    y3 = torch.sqrt(y3_sq.clamp_min(_EPS))                    # (B,)
    r3 = torch.stack([x3, y3, zeros], dim=-1)

    # r4 in 3D:
    #   x4 = (d12² + d14² - d24²) / (2 d12)
    #   y4 = (d13² + d14² - d34² - 2 x3 x4) / (2 y3)
    #   z4² = d14² - x4² - y4²
    #   sign(z4) = sign(V)
    x4 = (d12 ** 2 + d14 ** 2 - d24 ** 2) / (2.0 * d12_safe)
    y3_safe = y3.clamp_min(_EPS)
    y4 = (d13 ** 2 + d14 ** 2 - d34 ** 2 - 2.0 * x3 * x4) / (2.0 * y3_safe)
    z4_sq = d14 ** 2 - x4 ** 2 - y4 ** 2
    z4_abs = torch.sqrt(z4_sq.clamp_min(_EPS))                # (B,)
    # 用 V 的符号定 z4 chirality.
    #
    # 数学上 z4 = sign(V) · z4_abs. 但 torch.sign 在 V≠0 处梯度 = 0、在 V=0 处
    # 不可导, 反传时 r4 完全失去对 V 的依赖, recon 路径
    # (dv_to_cartesian → forward_model → image MSE) 给不出 V 维的梯度信号.
    #
    # 用 straight-through estimator (Bengio 2013):
    #   forward:  sign(V) (取值 ∈ {-1, 0, +1})
    #   backward: 按 1 处理 → ∂z4/∂V = z4_abs (V 维梯度信号正确导通)
    #
    # 实现技巧:
    #   sign_V_ste = V + (torch.sign(V) - V).detach()
    # forward 上 V 与 (sign(V)-V).detach() 相加 = sign(V);
    # backward 上 (sign(V)-V).detach() 贡献 0, 只剩 V 的梯度 = 1.
    sign_V_ste = V + (torch.sign(V) - V).detach()
    z4 = sign_V_ste * z4_abs
    r4 = torch.stack([x4, y4, z4], dim=-1)

    return torch.stack([r1, r2, r3, r4], dim=1)               # (B, 4, 3)


def _infer_n_clusters_from_dv(dv_size: int) -> int:
    """从 DV 维数 (= C(N, 2) + 1) 反推 N。"""
    # dv_size = N*(N-1)/2 + 1 → 2*(dv_size - 1) = N*(N-1)
    target = 2 * (dv_size - 1)
    for N in range(2, 20):
        if N * (N - 1) == target:
            return N
    raise ValueError(
        f"Cannot infer n_clusters from dv dim={dv_size}. "
        "合法 dim: 2 (N=2), 4 (N=3), 7 (N=4), 11 (N=5), ..."
    )


# ============================================================
# S_4 orbit 查找表（用于 orbit-marginalized NLL）
# ============================================================

import itertools as _itertools


def s_n_orbit_indices(n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    给 N 个全同原子，预计算 S_N 在 DV 表征上的作用查找表。

    返回
    ----
    pair_perm : LongTensor (n_perms, n_pairs)
        pair_perm[g, k] = "在 g 这个原子置换下，第 k 个 (i, j) pair 被映射到 DV
        向量里的哪一个位置"。换句话说，若 DV_old 是原 DV (按字典序)，那么
            DV_new[k] = DV_old[pair_perm[g, k]]
        其中 DV_new 是 *新标号下* 的 DV 表征。
    v_sign : FloatTensor (n_perms,)
        signed volume V 在置换下的乘数 ∈ {-1, +1}。对应 r_g(i) - r_g(0) 三向量
        行列式相对原 r_i - r_0 行列式的符号。

    注意：n_perms = N! ，对 N=4 = 24；这是 LUT，**初始化一次即可常驻**。
    """
    if n_clusters < 2:
        raise ValueError(f"n_clusters 必须 >= 2，收到 {n_clusters}")
    pairs = _pair_index_list(n_clusters)                       # [(i, j), ...]
    pair_to_idx = {p: k for k, p in enumerate(pairs)}
    n_pairs = len(pairs)
    perms = list(_itertools.permutations(range(n_clusters)))
    n_perms = len(perms)

    pair_perm = torch.zeros(n_perms, n_pairs, dtype=torch.long)
    v_sign = torch.ones(n_perms, dtype=torch.float32)

    for g_idx, g in enumerate(perms):
        # 在置换 g 下：新的 (i, j) pair 对应原 (g(i), g(j))
        # 但我们存的是 "DV_new[k] = DV_old[?]"，所以查表：
        #   对每个 k = (i, j) 在新 DV 里的位置，DV_new[k] = |r_g(i) - r_g(j)|
        #     = |r_g(j) - r_g(i)| = DV_old[pair_to_idx[sorted((g(i), g(j)))]]
        for k, (i, j) in enumerate(pairs):
            gi, gj = g[i], g[j]
            key = (gi, gj) if gi < gj else (gj, gi)
            pair_perm[g_idx, k] = pair_to_idx[key]

        # signed volume V 在 g 下变换：
        #   原: V = (1/6) det(r1-r0, r2-r0, r3-r0)    用原子 0..3
        #   新: V' = (1/6) det(r_g(1)-r_g(0), r_g(2)-r_g(0), r_g(3)-r_g(0))
        # 把后者展开成原 r_i 的线性组合，行列式 = det(matrix M) ，M 是从 r_0..r_{N-1}
        # 取 r_g(1) - r_g(0) 等 3 行的系数矩阵。
        # 简单做法：随机一个固定 4-point 构型，前后算 V 比一下符号。这样既正确又
        # 不用手算 sign。我们用一个固定的"不退化"测试构型保证非零体积。
        if n_clusters >= 4:
            # ★ 关键: 任何置换 g ∈ S_N 都可能把后面的原子调到前 4 位, 所以
            # 参考矩阵的 N 行必须线性独立、且任意 4 个的子集都构成非退化四面体.
            # 旧实现把 row 4+ 都放 (k·0.1, 0, 0), 与 row 0 (原点) 几乎共点,
            # 任何把这些行调到前 4 位的置换都会让 v_old·v_new → 0, 误判 v_sign.
            #
            # 新实现: 固定一个种子的随机三维高斯, 再加大量级偏置避免数值 0.
            # 这是 LUT 初始化, 每个 N 只跑一次, 不需要可微.
            _ref_gen = torch.Generator().manual_seed(20251127)
            ref = (torch.randn(n_clusters, 3, generator=_ref_gen, dtype=torch.float64)
                   + torch.arange(n_clusters, dtype=torch.float64).unsqueeze(-1) * 0.7)
            ref_g = ref[list(g)]
            # 原 V: 前 4 atom (在原坐标下)
            v_old = torch.linalg.det(torch.stack([
                ref[1] - ref[0], ref[2] - ref[0], ref[3] - ref[0]
            ], dim=0)).item()
            v_new = torch.linalg.det(torch.stack([
                ref_g[1] - ref_g[0], ref_g[2] - ref_g[0], ref_g[3] - ref_g[0]
            ], dim=0)).item()
            # 防御: 若数值上仍然 ≈ 0, 说明参考矩阵失效, 显式报错而不是默默 sign=+1
            if abs(v_old) < 1e-9 or abs(v_new) < 1e-9:
                raise RuntimeError(
                    f"s_n_orbit_indices: 参考矩阵在 N={n_clusters} 下退化 "
                    f"(|v_old|={abs(v_old):.2e}, |v_new|={abs(v_new):.2e}). "
                    "请换一个随机 seed 或更稠密的参考构型."
                )
            v_sign[g_idx] = 1.0 if v_old * v_new > 0 else -1.0
        else:
            v_sign[g_idx] = 1.0    # N < 4 时 V 恒为 0，符号无意义

    return pair_perm, v_sign


def apply_s_n_orbit(
    dv: torch.Tensor,
    pair_perm: torch.Tensor,
    v_sign: torch.Tensor,
) -> torch.Tensor:
    """
    把 S_N 查找表作用到 DV 向量上。

    Args
    ----
    dv        : (B, dv_dim)            原 DV 表征
    pair_perm : (n_perms, n_pairs)      由 s_n_orbit_indices 得到
    v_sign    : (n_perms,)              同上

    Returns
    -------
    dv_orbit  : (B, n_perms, dv_dim)
        dv_orbit[b, g] = g 作用在 dv[b] 上的结果（距离按 pair_perm 重排、V 乘 ±1）
    """
    if dv.dim() != 2:
        raise ValueError(f"apply_s_n_orbit 期望 dv (B, dv_dim)，收到 {tuple(dv.shape)}")
    B, D = dv.shape
    n_perms, n_pairs = pair_perm.shape
    if n_pairs + 1 != D:
        raise ValueError(
            f"apply_s_n_orbit: pair_perm.shape[1]={n_pairs} 与 dv 维数 {D} 不一致 "
            f"(期望 dv = n_pairs + 1 = {n_pairs + 1})"
        )
    pair_perm = pair_perm.to(device=dv.device)
    v_sign = v_sign.to(device=dv.device, dtype=dv.dtype)

    # 距离部分：fancy indexing 走 expand
    d_part = dv[:, :n_pairs]                                    # (B, n_pairs)
    # gather: result[b, g, k] = d_part[b, pair_perm[g, k]]
    pp_expand = pair_perm.unsqueeze(0).expand(B, n_perms, n_pairs)  # (B, G, P)
    d_orbit = torch.gather(
        d_part.unsqueeze(1).expand(B, n_perms, n_pairs),
        dim=-1,
        index=pp_expand,
    )                                                            # (B, G, n_pairs)

    # V 部分：(B,) × (G,) → (B, G)
    V_part = dv[:, -1].unsqueeze(-1) * v_sign.unsqueeze(0)       # (B, G)

    return torch.cat([d_orbit, V_part.unsqueeze(-1)], dim=-1)    # (B, G, dv_dim)


# ============================================================
# ★ dist (pure pairwise-distance) 表征 — O16 去掉 V 的 6D 版本
# ============================================================
#
# 物理依据 (dv_representation_group_theory.md §6.5 + chirality_forward_check.json):
# UPC forward 在 SO(3) 取向平均后宇称盲 (手征差 ≈ 蒙特卡洛噪声, ratio≈0.48,
# 随 1/√N 衰减趋 0) ⇒ sign(V) 不可识别; 而 |V| 又被 6 条距离经 Cayley-Menger
# 完全决定 (288·V² = CM₄). 因此对 O16 (N=4) 而言 V 不携带任何*可识别*信息:
# 6 条距离本身就是可识别后验的完备且最小坐标, 且对反射 (宇称) 天然不变
# —— 宇称白送, 无需 orbit marginalize, 也没有 v_scale / CM 曲面 / ±V 双峰之累.
#
# 与 DV 的关系: dist = DV 砍掉最后一维 V. dim = C(N,2).
#   - cartesian_to_dist  = cartesian_to_dv[:, :n_pairs]
#   - dist_to_cartesian  = dv_to_cartesian, V 维填正号 sentinel (z4 ≥ 0 约定)
#   - orbit: 只重排距离 (s_n_orbit_indices 的 pair_perm), 不需要 v_sign


def dist_dim(n_clusters: int) -> int:
    """N → dist 表征维数 (= C(N, 2), 纯两两距离, 无 V)。"""
    if n_clusters < 2:
        raise ValueError(f"n_clusters 必须 >= 2，收到 {n_clusters}")
    return _comb(n_clusters, 2)


def _infer_n_clusters_from_dist(dist_size: int) -> int:
    """从 dist 维数 (= C(N,2)) 反推 N。"""
    # dist_size = N*(N-1)/2 → 2*dist_size = N*(N-1)
    target = 2 * dist_size
    for N in range(2, 20):
        if N * (N - 1) == target:
            return N
    raise ValueError(
        f"Cannot infer n_clusters from dist dim={dist_size}. "
        "合法 dim: 1 (N=2), 3 (N=3), 6 (N=4), 10 (N=5), ..."
    )


def cartesian_to_dist(r: torch.Tensor) -> torch.Tensor:
    """
    (B, N, 3) → (B, C(N,2)) 纯两两距离 (字典序 i<j)。

    = cartesian_to_dv 去掉最后的 V 列。对反射 (宇称) 天然不变。
    """
    dv = cartesian_to_dv(r)              # (B, C(N,2)+1)，含 V
    return dv[:, :-1]                     # 去掉 V


def dist_to_cartesian(
    dist: torch.Tensor,
    n_clusters: Optional[int] = None,
) -> torch.Tensor:
    """
    (B, C(N,2)) → (B, N, 3) 在标准 frame 下笛卡尔, 全程可微。

    与 dv_to_cartesian 同一套闭式 MDS, 但因 dist 不含手征信息, 统一取
    ``z4 ≥ 0`` 约定 (右手). 物理上 forward 宇称盲, 取哪个手征重建出的像一致。
    当前只支持 N=4 (与 dv_to_cartesian 相同限制)。
    """
    if dist.dim() != 2:
        raise ValueError(
            f"dist_to_cartesian 期望 (B, dist_dim)，收到 {tuple(dist.shape)}"
        )
    if n_clusters is None:
        n_clusters = _infer_n_clusters_from_dist(dist.shape[1])
    expected = dist_dim(n_clusters)
    if dist.shape[1] != expected:
        raise ValueError(
            f"dist dim={dist.shape[1]} 与 n_clusters={n_clusters} 不一致 (期望 {expected})"
        )
    # 填一个正的 V sentinel → sign(V)=+1 → z4 ≥ 0 约定; 幅值任意 (dv_to_cartesian
    # 只用 sign(V) 定 z4 方向, z4_abs 完全由距离决定, |V| 不进入几何).
    V_pos = torch.ones(dist.shape[0], 1, device=dist.device, dtype=dist.dtype)
    dv = torch.cat([dist, V_pos], dim=-1)
    return dv_to_cartesian(dv, n_clusters=n_clusters)


def apply_s_n_orbit_dist(
    dist: torch.Tensor,
    pair_perm: torch.Tensor,
) -> torch.Tensor:
    """
    把 S_N 查找表作用到 dist 向量上 (纯距离重排, 无 V)。

    Args
    ----
    dist      : (B, C(N,2))           原 dist 表征
    pair_perm : (n_perms, n_pairs)     由 s_n_orbit_indices 得到 (只用 pair_perm)

    Returns
    -------
    dist_orbit : (B, n_perms, C(N,2))
        dist_orbit[b, g] = g 作用在 dist[b] 上 (按 pair_perm 重排距离)。
    """
    if dist.dim() != 2:
        raise ValueError(
            f"apply_s_n_orbit_dist 期望 (B, dist_dim)，收到 {tuple(dist.shape)}"
        )
    B, D = dist.shape
    n_perms, n_pairs = pair_perm.shape
    if n_pairs != D:
        raise ValueError(
            f"apply_s_n_orbit_dist: pair_perm.shape[1]={n_pairs} 与 dist 维数 {D} "
            f"不一致 (dist = n_pairs = {n_pairs})"
        )
    pair_perm = pair_perm.to(device=dist.device)
    pp_expand = pair_perm.unsqueeze(0).expand(B, n_perms, n_pairs)   # (B, G, P)
    dist_orbit = torch.gather(
        dist.unsqueeze(1).expand(B, n_perms, n_pairs),
        dim=-1,
        index=pp_expand,
    )                                                                # (B, G, n_pairs)
    return dist_orbit


__all__ = [
    'ELEMENT_REGISTRY',
    'element_info',
    'zmatrix_dim',
    'infer_n_clusters',
    'reference_geometry',
    'canonicalize_clusters',
    '_break_chirality_first_dihedral',
    'cartesian_to_zmatrix',
    'zmatrix_to_cartesian',
    'cluster_rms_radius',
    'rescale_to_radius',
    'round_trip_error',
    # ★ DV (pairwise-distance + signed-volume) 表征
    'dv_dim',
    'cartesian_to_dv',
    'dv_to_cartesian',
    's_n_orbit_indices',
    'apply_s_n_orbit',
    # ★ dist (pure pairwise-distance) 表征 — O16 去 V 的 6D 版本
    'dist_dim',
    'cartesian_to_dist',
    'dist_to_cartesian',
    'apply_s_n_orbit_dist',
]
