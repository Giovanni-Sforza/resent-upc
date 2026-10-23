# Q10、B_mid、K_tail、mean pT 联合参数扫描：A100 80 GB 运行说明

本次新增独立扫描程序，复用现有 `DiffractionRefiner`；历史 forward、四案例结果和文章文件均不修改。正式扫描由服务器运行。本地微型例子只检查实现，不能用于物理结论。

## 1. 已确认的 300 点参数网格

正式配置为 `configs/shape_scan_a100.yaml`。用户已确认 β₂ 在 0–0.3 之间等距取 10 点、β₃ 在 0–0.2 之间等距取 10 点，即 `linspace(0,0.3,10)` 与 `linspace(0,0.2,10)`。none / skin / halo **每类各一个代表密度**，共 `3 × 10 × 10 = 300` 个案例，其中 3 个是各密度的球形点，其余 297 个为形变点。配置保存完整精度节点，不使用图例中舍入后的两位小数标签。

这是本次明确选定的扫描网格，不声称完整找回了旧文章原始网格或 WS 参数表。三类代表密度均固定 `R_p=5.02 fm`、`a_p=0.46 fm`：none 沿用质子与中子相同的几何轮廓；skin 沿用旧诊断中的 RMS 半径差 `Δr_np=0.20 fm`，固定 `a_n=a_p` 并解 `R_n`；halo 使用同一 `Δr_np=0.20 fm`，固定 `R_n=R_p` 并解 `a_n`。skin 与 halo 的匹配在每个形变点重新求解。

默认每个形变点先算 **2048 对取向**，输出目录为 `outputs/shape_scan/a100_grid300_n2048_extendable`。随后可在同一科学配置、同一目录与同一取向库前缀上延长至 32768 对；目录名中的 `n2048` 记录首轮目标，不是上限。三个球形点各只计算一对确定性取向。

另有 `configs/shape_scan_a100.template.yaml` 供自定义后续扫描。模板保留生产求积、精度和取向库，但三项扫描列表留空，不能直接启动；应复制为另一个配置，填入精确的 `scan.beta2_values`、`scan.beta3_values` 及有独立 `id` 的 `scan.density_variants`，并指定新的输出目录。列表顺序也登记入科学指纹。支持两种密度参数化：

| mode | 必填量 | 形变变化时保持什么 |
|---|---|---|
| `direct_ws` | `R_p,a_p,R_n,a_n` | 四个 WS 参数固定，实际 RMS 差随形变重新计算 |
| `rms_matched` | `R_p,a_p,delta_r_np_fm` | 固定指定的 RMS 差；skin 解 R_n，halo 解 a_n |

`type: skin` 要求 `a_n=a_p, R_n>R_p`；`type: halo` 采用 `R_n=R_p, a_n>a_p`；`type: none` 要求质子、中子的几何轮廓相同。R、a、RMS 差均以 fm 为单位。none 没有非零中子皮强度轴；自定义不同 none 变体时需明确不同的共同 R、a，不能只换名字复制并视为独立物理信息。

正式配置的三种代表密度如下；完整的 10×10 节点列表见该 YAML：

```yaml
scan:
  density_variants:
    - id: none
      type: none
      mode: rms_matched
      R_p: 5.02
      a_p: 0.46
      delta_r_np_fm: 0.0
    - id: skin
      type: skin
      mode: rms_matched
      R_p: 5.02
      a_p: 0.46
      delta_r_np_fm: 0.20
    - id: halo
      type: halo
      mode: rms_matched
      R_p: 5.02
      a_p: 0.46
      delta_r_np_fm: 0.20
```

质子、中子共享 β₂、β₃ 和每个核的取向，两个核的取向独立抽样。沿用旧模型关于 WS 参数原点的 `sqrt(<r²>)` 定义，不额外加入体积保持或质心平移修正。混合 β₂、β₃ 时必须记住这一模型约定。每个案例的实际 R、a、质子/中子 RMS 与 RMS 差都写入结果。

本次正式网格包含 `(0,0)`，可对每种密度作对应的球形比较。自定义网格若没有该点，程序不会自动加入，报告会说明对应比较不可用。

## 2. 固定的观测量定义

末态定义为 `qT=pT_plus+pT_minus`、`kT=(pT_plus-pT_minus)/2`，以 kT 为圆环图参考轴；不需要逐事件知道 b 的方向。本次四个量只使用同一张图的角平均密度 `J(q)=<A0_full(q)>`，不需要球形核的测量比值，也不保留额外的方位信息。阈值量直接使用 J 的峰高；平均 pT 则按径向计数测度 `dN/dq=2πq J(q)` 加权。

先平均原始强度，再找该平均谱自己的离散主峰。在主峰外侧的同一连续下降支上，用相邻节点线性插值定义 `Q50,Q10,Q1`，分别对应自身峰高的 50%、10%、1%。

```
B_mid  = ln(5)  / (Q10² - Q50²)
B_tail = ln(10) / (Q1²  - Q10²)
K_tail = B_tail / B_mid
mean pT = ∫ q² J(q) dq / ∫ q J(q) dq
```

Q10 和 mean pT 的单位为 GeV/c，B_mid 为 (GeV/c)^−2，K_tail 无量纲。mean pT 是该双轻子系统横动量大小 q 的单图均值，代码字段为 `mean_pT_GeV`；积分中的 2π 因子相消。若仅把完整 q 轴拉伸 s 倍，Q10 变成 s 倍，B_mid 变成 1/s²，K_tail 不变；固定有限窗口内的 mean pT 还会受窗口截断影响。这是检查“尺度变化”和“尾部相对形状变化”的出发点，不是保证选择性成立的结论。

达到首次回升或存储边界仍未越过阈值，则明确记为删失；不跨过回升寻找后续交点，不把 qmax 当成 Q1。沿用旧数值定义，只保存 `q<=0.17` 的原有节点；生产网格最后实际节点约为 0.16985545 GeV/c。报告同时列出名义边界和实际末点。

mean pT 使用同一组原有节点的完整已存区间，从 0 积分到实际末点；对分段线性 J 精确计算 q 与 q² 加权积分，不外推到名义 0.17 GeV/c 或更远处。因此它是该有限区间内的均值，不是全 pT 范围的均值。缺少某个阈值交点并不自动删失 mean pT。新增此量只需现有 A0，不改变原始数据存储格式。

full 保留双源相干和；self 去掉两个生产源之间的交叉项，但保留单核内部相干衍射。self 是理论机制对照，不是实验中可单独选择的事件，也不是核破碎的 incoherent 成分。四个量都报告 full/self；主结果使用 full。已有四案例的 K_tail 形变响应在 self 中仍存在，不能先将它命名为“纯干涉观测量”。

## 3. 环境与微型检查

使用服务器已有 CUDA PyTorch 环境，依赖见 `upc_deformation/requirements.txt`。在仓库根目录执行：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_shape_scan_*.py tests/test_diffraction_forward.py
python -u scripts/run_shape_scan.py --config configs/shape_scan_smoke.yaml \
  --device cuda --output outputs/shape_scan/cuda_smoke
python scripts/analyze_shape_scan.py --run outputs/shape_scan/cuda_smoke \
  --output outputs/shape_scan/cuda_smoke_analysis
```

smoke 是三类密度、2×2 形变点、每个非球形点 4 对取向的粗网格，**不是本次 300 点正式扫描**。可以另用原有验证程序检查 CUDA 与 CPU、逐个与批量的同精度闭合：

```bash
python scripts/validate_diffraction_observables.py --device cuda --tiny --pairs 2 \
  --output outputs/shape_scan/cuda_kernel_validation
```

## 4. 正式网格的 A100 批量测试与扫描

先检查正式配置的 300 个注册案例，再测试实际生产网格的 batch：

```bash
python scripts/run_shape_scan.py --config configs/shape_scan_a100.yaml --list-cases
python scripts/benchmark_shape_scan.py --config configs/shape_scan_a100.yaml \
  --device cuda --batches 8 16 32 64 --y-batches 1 5 \
  --max-memory-fraction 0.8 --output outputs/shape_scan/benchmark_a100
```

benchmark 默认选最大绝对 β₂、β₃ 的首个 halo 变体；可用 `--case` 指定注册名称。每个方案先预热，再同步计时，记录 allocated/reserved 显存、吞吐与 OOM。只推荐实际成功且 reserved 不超过设备总显存 80% 的方案；80 GB 容量不等于某个 batch 已经验证。正式配置的 batch=8、y_batch=1 只是起始配置。

将 benchmark 推荐值写入 YAML 的 `runtime.batch_size` 和 `study.y_batch_size`，随后执行：

```bash
python -u scripts/run_shape_scan.py --config configs/shape_scan_a100.yaml
python scripts/analyze_shape_scan.py --run outputs/shape_scan/a100_grid300_n2048_extendable \
  --output outputs/shape_scan/analysis_n2048
```

先查看并回传这一轮结果。需要提高取向统计量时，再在原目录延长：

```bash
python -u scripts/run_shape_scan.py --config configs/shape_scan_a100.yaml \
  --n-orientations 32768
python scripts/analyze_shape_scan.py --run outputs/shape_scan/a100_grid300_n2048_extendable \
  --output outputs/shape_scan/analysis_n32768
```

首条命令使用 YAML 默认的 2048 目标。先做 2048 的作用是检查全参数面的删失、混合效应和明显数值问题，已有取向前缀随后在同一目录原样延长至 32768，不浪费已保存结果。也可以在首次运行时显式指定 `--n-orientations 32768`。球形点只计算一次；其取向 MC 误差为零不意味着物理或离散误差为零。

程序以 case 为外循环，同一时刻仅保留一个 GPU 模型。所有非球形点使用同一对有序取向库；full/self、各 b 子网格也共享样本。生产默认仍为 FP64、b=12–60 fm，0.5/1/2 fm 子网格，|y|<1 的 5 点 Gauss–Legendre。硬 b 窗口不是经校准的中子标记事件选择。

## 5. 断点、存储与分析

原子 checkpoint 默认每 128 对保存一次，与 GPU batch 独立。重新执行同一命令会校验并续跑；中断最多损失尚未提交的一个 checkpoint。`--max-new-pairs` 可限制本次新增总对数，`--case` 可选注册案例，`--prepare-only` 只登记配置与取向库，不执行 forward。

可以延长取向目标，不能减少已登记目标。batch、y_batch、checkpoint 大小可改；改物理参数、求积网格、精度、设备类型、取向生成约定或 forward 源文件需要新输出目录。代码与取向资产的 SHA 在开始和结束都校验。完成的旧 chunk 不覆盖，写锁阻止两个生产进程同时写一个目录。请在生产进程结束后分析。

每对取向只保存三个 b 层级、full/self 两分支的未归一化 A0，生产默认为 455 个径向节点；新增 mean pT 不增加原始数组。2048 对单个形变案例的原始 A0 约 **0.04166 GiB**，297 个形变案例合计约 **12.37 GiB**；延长至 32768 对后单案例约 **0.667 GiB**，合计约 **198 GiB**。此外还有三个确定性球形点、共享取向库与少量元数据，不另存全案例重复数组。以上为压缩前数组估计；并非 A100 显存或耗时测量。分析逐案例读入，不一次加载全扫描；需另留主机内存和输出空间。

误差来自共享计数的取向块 bootstrap：每次先重平均 A0，再重新找峰、阈值和计算 mean pT。默认 32 对为一块、1024 次重采样；四观测量、b 层级、full/self 和同一取向前缀的不同案例保持配对。四个量使用同一批原始强度，报告按 `Q10_GeV, B_mid, K_tail, mean_pT_GeV` 排列的 **4×4 协方差**，不当作四个独立测量。若任一重采样交点缺失，报告删失比例及明确标记的条件统计，不把剩下样本的 SE 冒充无条件误差。小样本使用逐取向重采样；较大 n 必须整除块长。

分析输出包括：

- `metrics.csv`：每案例、b 层级、full/self 的四个量与误差、阈值、删失信息和实际密度。
- 每案例 JSON、bootstrap replicate NPZ：支持后续联合误差与配对比较。
- `radial_means.npz`：各案例的平均谱，方便回传后追溯峰与阈值；不含逐取向大型数组。
- `prefix_stability.csv`：嵌套前缀稳定性；不把前缀之间当作独立样本。
- `paired_response.csv`：与已注册同密度球形点、同 β 的 none 各变体的差值/比值及配对误差。比值是理论响应诊断；单图观测量定义本身不需要该参考。
- 四个量各一张随 β₂、β₃ 的参数图，共四张；每个量跨密度变体保持同一色标，另有分析清单/哈希。

默认必须完成全部注册案例；`--allow-partial` 明确分析已完成子集。`--n` 请求的非球形前缀必须确实存在，绝不静默截短。分析输出不能覆盖非空目录。回传时可以只压缩分析目录：

```bash
tar -czf analysis_n2048_reports.tar.gz -C outputs/shape_scan analysis_n2048
shasum -a 256 analysis_n2048_reports.tar.gz
```

延长至 32768 后，将上述包名和目录名中的 2048 同时替换为 32768。

服务器上保留完整生产目录。报告不包含大型逐取向 checkpoint。MC 误差和 b 子网格比较不包括探测器、事件统计、背景、中子标记、y/空间/FFT/角度/径向网格以及模型系统误差；参数扫描也不会自动证明四个量足以唯一反演所有参数。

## 6. 本地交付检查（2026-10-24）

`tests/test_shape_scan_*.py` 与 `tests/test_diffraction_forward.py` 合计 **75 passed、1 skipped**，跳过的是 CUDA 检查。本地覆盖密度适配、共用取向库、断点续跑/扩展、已提交数据与源码校验、四量联合统计、删失、完整报告与原四案例三量回归。

真实 CPU 粗网格完成三类密度 × 2×2 形变共 12 案例、39 对取向的 forward→分析→4 张 PNG/PDF/SVG 流程；输出 48 个 b/branch 通道，4 维 bootstrap 数组和报告哈希均已核对。另完成粗网格 CPU batch=1/2 的基准测试。300 个正式参数点只做了配置与密度解析核查，没有运行正式 forward，也没有 A100 耗时/显存实测。服务器运行命令及完整生产结果才用于后续物理讨论。
