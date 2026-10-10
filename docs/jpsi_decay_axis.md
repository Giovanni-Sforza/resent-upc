# J/psi 衰变轴二维图：计算定义与复现

## 图的目标与范围

用同一套 Torch 产生振幅比较：(a) b=12 fm、已知 b 轴的 J/psi qT；(b) 相同固定 b 下的轻子相对 kT；(c) b=12 fm、以轻子衰变轴对齐的 J/psi qT；(d) 对 |b| 积分、以轻子衰变轴对齐的 J/psi qT。这是诊断性正演比较，不是重训结果或探测器级预测。旧三栏图 a/b/c 的 qT 数据原样保留为 a/c/d。

采用 Python/matplotlib 的四栏定量图。J/psi 图 a/c/d 先在同一个动量圆盘内归一化为单位产额，默认使用共同的对数色标。b 的 kT 范围与归一化不同，使用独立对数色标（覆盖所有案例的完整正值范围），使边界峰值不再压暗内部结构；零概率的运动学区域外显示灰色。保存原始强度、归一化强度、动量轴、极坐标系数、所有参数及 PNG/PDF/SVG。不能逐栏按峰值拉伸来制造视觉相似。

原文提供的碰撞条件：96Zr+96Zr，sqrt(s_NN)=200 GeV，y=0，固定 b=12 fm，显示范围 +/-0.17 GeV。当前仓库未包含原始数据生产配置，原文也未明确列出完整的质子/中子 WS 数值。本次使用明确登记的示例 R_p=R_n=5.02 fm，a_p=a_n=0.46 fm；不声称逐像素复现旧训练集。

三个核结构示例为球形、beta2=0.3 的 tip-tip、beta2=0.3/beta3=0.2 的独立随机核取向平均。随机取向属于已有模型的经典强度平均处方，不是新增的量子转动投影。取向平均与全局 b 方位平均分开。

## 来源

`upc_deformation/vendor/` 从 upc-alpha 的 Torch forward 及其可选几何依赖复制而来，来源提交和文件哈希记录于 `provenance.json`。仅适配本地 import。没有调用原项目，也不依赖 NumPy 版本的物理实现。保留原 Torch 模型包括 thickness 归一化到 Z 在内的既有处方；本次不是对该处方的外部物理标定。

本地 `upc_deformation/forward.py` 另外提供一个明确的单位桥接：源程序的 qL=M/(2 gamma) exp(+-y) 是 GeV，而 exp(i qL z) 中 z 为 fm、光子场 FFT 网格为 fm^-1，故 physical 模式在这两处将 qL 除以 hbar_c。保留源程序的光子能量前因子；本项目只比较归一化形状，尚未标定绝对截面。复制的 vendor 文件不修改。`--longitudinal-units upstream` 可精确重放源版处方；新默认值为 `physical`。不能把该单位修正归因于衰变轴对齐。

## 衰变角公式

在 b 沿 x 的坐标中保留完整复振幅，先在完整 FFT 网格形成

S0=(|Mx|^2+|My|^2)/(2pi)，S1=(|Mx|^2-|My|^2)/(2pi)，S2=2 Re(Mx My*)/(2pi)。

对 J/psi -> l+l-，忽略轻子质量、采用横向偏振/小 q_T/M 近似，在介子静止系以束流为 z 方向：

dP/(d²q dOmega) = 3/(8pi) [S0 - sin²(theta) (S0+S1 cos(2psi)+S2 sin(2psi))/2]。

在完整 cos(theta) 接受度积分后：

dP/(d²q dpsi) = 1/(2pi) [S0 - (S1 cos(2psi)+S2 sin(2psi))/2]。

这来自轻子张量 delta_ij - n_i n_j。角积分归一到原始 S0；没有接入完整事件衰变生成器。复数交叉项被保留，不能把振幅直接换成模长。

设 k_T=(p_T,+ - p_T,-)/2，q_T=p_T,+ + p_T,-，主图 x 轴平行 k_T。令 alpha=phi_q-psi，chi=phi_q-psi_b，在每个 q 上定义

A0=<S0(q,chi)>_chi，C=<S1 cos(2chi)+S2 sin(2chi)>_chi，D=<S1 sin(2chi)-S2 cos(2chi)>_chi。

则主图强度为 I(q,alpha)=A0 - [C cos(2alpha)+D sin(2alpha)]/2。

若改用 x 轴垂直 k_T，上式二阶项反号；这只是将同一幅可观测图转 90 度，不改善分辨率。全局随机 b 方位已经在此积分中消去，保留的是与衰变轴的相对角。此近似只含 0、2 阶方位谐波，无法恢复原图全部高阶条纹。

参考推导：Brandenburg et al., Phys. Rev. D 106, 074008 (2022)，[Eq. (3)-(4)](https://arxiv.org/html/2207.02478)；实验相对角定义见 [STAR J/psi preprint Eq. (2)](https://arxiv.org/html/2512.02865v2)。没有包含软光子辐射、轻子效率/接受度或有限 q_T boost 修正。

## b 积分

默认对 12 <= b <= 60 fm 做 2pi b db 的梯形求积。产生振幅本身已含光子场的 b 依赖；不额外重复乘光子通量。每个 b 先形成强度/密度矩阵，然后积分；不跨不同碰撞 b 相干叠加振幅，也不先把各 b 图归一化后等权平均。

这是明确的 hard-window 诊断样本，没有声称对应已标定的 0n0n/XnXn 中子类别或完整无强子碰撞效率。保留逐 b 结果可检查求积和尾部。

## 新增 kT 分布及角度来源

kT 是轻子相对横动量 `(pT_plus-pT_minus)/2`，与 J/psi 总横动量 qT 不同，也不是光子的横动量。沿用上文的小 qT/M、无质量、全极角接受度近似。先对同一 qT<0.17 GeV 样本积分 Stokes，记 `p1=integral(S1)/integral(S0)`、`p2=integral(S2)/integral(S0)`，`k0=M/2`。

旧 NumPy 代码的 `image_P` 为 `M² Sigma-4 k_i T_ij k_j`。这里仅借用其矩阵元形式作解释图，所有 T_ij 都从本地 Torch 复振幅计算，qT 截断也使用本任务的 0.17 GeV。主偏振沿 b=x 时，该权重约为 `1-kx²/k0²`，所以等值线接近竖线；它并不是周期性干涉条纹，也不等于 kT 的二维概率密度。

对两个 kz 分支求和，`dOmega/d²kT=2/[k0² sqrt(1-kT²/k0²)]`。因此新的 b 图为

`dP/d²kT = 3/[4pi k0² sqrt(1-kT²/k0²)] * [1-((1+p1)kx²+(1-p1)ky²+2p2 kx ky)/(2k0²)]`，仅支持 `kT<k0`。

图中边界的可积增强由有限像素面积积分处理（y 方向解析积分、x 方向 Gauss 求积），没有对边界点取发散值，也没有平滑或裁去极角。b 图的圆边界是两体衰变运动学；c/d 的近圆环是衰变轴对齐后的方位信息损失，两者不能混为同一原因。

角度定义：`psi=phi_k-phi_b`，`alpha=phi_q-phi_k`。完整极角积分给出 `P(psi)=[1-(p1 cos2psi+p2 sin2psi)/2]/(2pi)`。对当前形变平均样本，p1=0.933315、p2=0.010325；其峰谷比约 2.75，仍是很宽的方向分布。取垂直 kT 的轴来估计 b（轴方向按 pi 等价），`<cos 2(delta)> = p1/4 = 0.233329`，偏差在 ±15 度内的概率约 24.09%。这是本理想接受度下的样本相关性，不是实验探测器分辨率。

角度诊断图左栏复现“竖条形”的矩阵元权重，中栏说明几何角度，右栏展示真实的角度边缘分布。c/d 的实际计算继续使用完整 q-k 联合分布；不能将这个边缘 P(psi) 当作与 q 无关的随机模糊核。方向信息并未完全消失，但只能保留有限的二阶调制，无法恢复贯穿图像的暗条纹。

## 运行

```bash
python3 scripts/compare_jpsi_decay_axis.py --config configs/jpsi_decay_axis.yaml --output outputs/jpsi_decay_axis_final
python3 -m pytest -q tests/test_decay_axis.py
```

对已有三栏结果增补 kT 图（只重放固定 b，无需重跑 b 积分）：

```bash
python3 scripts/add_jpsi_kt_panel.py --input outputs/jpsi_decay_axis_final --output outputs/jpsi_decay_axis_four_panel
python3 -m pytest -q tests/test_decay_axis.py tests/test_kt_distribution.py
```

四栏图及角度说明图见 `outputs/jpsi_decay_axis_four_panel/gallery.md`。其 summary.json 单独记录原数据来源、固定 b 的重放误差与 kT 积分检查；原始 a/c/d 图像数组逐元素保持不变。新的完整生产命令也已默认输出四栏图。

最终图在 `outputs/jpsi_decay_axis_final/`（physical、49个 b 节点）。首轮原样 Torch 结果在 `outputs/jpsi_decay_axis/`，单位修正的25节点对照在 `outputs/jpsi_decay_axis_physical/`；以各自保存的 config.yaml 和 summary.json 为准。后续运行建议明确指定新的输出目录，保留旧结果。该差别会影响 b 权重，不能把源版有限 b 窗口作为完整积分的收敛结论。

首次先用 `--quick --output outputs/jpsi_decay_axis_quick` 做小网格端到端检查。正式命令保存配置快照、日志、运行时间、数据及图。实际数值检查与结果在输出目录的 `report.md` / `summary.json`。
