# 单源衍射参考、外坡与亮区 C2：服务器运行说明

本次交付独立计算代码。正式结果由服务器运行产生；本地小网格检查只能验证实现、存取和分析流程，不能证明生产网格收敛或实验可辨识性。沿用原模型，不修改 Fig. 1–7、神经网络或历史结果。

## 物理定义

对每个核取向对、b 和 y，先获得两路生产振幅 M1、M2，然后在原生 FFT 网格构造：

```
full  = Stokes(M1 + M2)
self  = Stokes(M1) + Stokes(M2)
cross = full - self
```

再执行同一套径向/方位投影、快度与 b 求积、取向平均。分支不分别归一化。self 是去掉两个生产源之间交叉项后的理论对照，仍保留单核内部的相干衍射、非均匀光子场和现有源抑制因子；它不是核破碎的 incoherent 分量，也不是实验可以单独选出的事件类。

保存的系数为 `(A0, C, Dsin)`，衰变角分布 `I(q,alpha)=A0-0.5*(C*cos(2alpha)+Dsin*sin(2alpha))`。本文档中的衍射参考 `D(q)=<A0_self(q)>` 与 Dsin 不同；完整径向密度 `J(q)=<A0_full(q)>`，交叉贡献 `X(q)=J(q)-D(q)`。D、J 是径向密度，不乘 q；`dN/dq=2*pi*q*J(q)` 的峰位不能当成 D 的衍射峰位。D 也不必是 J 的逐点上界。

末态可测变量保持为 `qT=pT_plus+pT_minus`、`kT=(pT_plus-pT_minus)/2`、`alpha=arg(qT)-arg(kT)`。不使用实验未知的 b 方位。C2 仍通过 Cartesian 图像重建、圆周插值和角矩求比计算，保持与原 panel d 定义一致。解析 `-<C>/(4*<A0>)` 仅作为插值诊断另列。q=0 的 alpha 未定义，角关联曲线不报告这个点。

窗口观测量使用 `sum_pixels I*cos(2alpha) / sum_pixels I`，先平均原始分子分母再取比值，不平均各 q 点或各取向的 C2。cross 的原始角矩可以相减，不能将它归一化成一个正概率分布的 C2。分析报告的 `C2_full-C2_self` 是两种计算的差，并不是“交叉项的 C2”。其误差保留同取向 full/self 的协方差。

## 窗口与本次要检验的问题

只用球形 self 参考的主瓣确定窗口，再将物理 q 边界固定应用于所有结构。默认规则如下；这些是预先注册的参考选择，未经灵敏度优化。

| 名称 | 边界 | 用途 |
|---|---|---|
| bright_lobe_inner | 0.03 到 D 外侧半高点 | 有限 q 的亮区内部 C2 |
| outer_slope | D 从峰高 50% 降到 10% 的外侧区间 | 径向产额/形状对中子皮的响应，同时检查形变混入 |
| bright_lobe | 0.03 到 D 外侧 10% 点 | 两个子区合并后的有限 q 亮区 C2 |
| fixed_012_014 | 0.12–0.14 GeV/c | 与既有重点区间对照，无论它是否位于上述亮区 |

50%/10%、0.03 下限和固定窗口均在 YAML 中明确登记。改变分析规则使用新配置和新输出目录；不要在看过响应后悄悄改窗口并称为事先选择。阈值交点用相邻径向节点线性插值，另保留括区；它们的插值精度不是物理误差。报告扫描范围内首个离散极小值是否可见，离散极小值也不等于已确认的物理衍射零点。若 q<=0.17 内没有极小值，标记边界未覆盖，不能把 qmax 当成衍射零点。亮区窗口只是主瓣的一个有限 q 子集。若出现真实 D=0 节点，保留 D/J 原谱，只屏蔽该点未定义的 J/D 比值。

这次首先检验：中子皮是否稳定改变 D 的外坡，并在 J 中留下对应变化；形变在亮区中是否改变 C2，且该变化有多少来自打开双源交叉项。同时检查中子皮对 C2、形变对 D 的响应。代码不预设两个观测量分别“纯测”一个参数。四个端点案例也不足以证明联合参数可辨识性。

## 默认生产配置

`configs/diffraction_observables_a100.yaml` 保留原高精度设置：

- 球形；beta2=0.3；beta3=0.2；球形 skin_fm=0.2。skin_fm 仍是现有模型参数标签，不能直接等同于实验 RMS 中子皮厚度。
- 两种形变各 32768 对取向，两个核独立抽样，beta2/beta3 共用取向库；球形与球形 skin 各计算一次。
- 原 512 对取向矩阵作为前缀，所有振幅按本次精度重算。前缀原本为 FP32，转 FP64 不会消除旧矩阵的舍入残差。
- b=12–60 fm，最细步长 0.5 fm；同节点保留 1、2 fm 子网格。硬 b 窗口仍是诊断样本，不是校准后的中子标记选择。
- |y|<1，5 点 Gauss–Legendre；FP64/complex128，关闭 TF32/AMP；batch_size=1、y_batch_size=1。
- 与原结果相同的密度、空间、padding、投影网格；分析到 0.17 GeV/c。
- 独立目录 `outputs/diffraction_observables/a100_b05_float64`。

## 服务器命令

解压并使用服务器已有的 CUDA 版 PyTorch 环境，依赖见 `upc_deformation/requirements.txt`：

```bash
tar -xzf diffraction_observables_a100.tar.gz
cd diffraction_observables
python scripts/package_diffraction_observables.py --verify ../diffraction_observables_a100.tar.gz
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_diffraction_*.py
```

先做新内核与旧 full 内核、批量与逐个计算、CPU/CUDA 同精度对照。下述 `--tiny` 是接口检查，采用故意粗化的真实 forward 网格，不能评价物理曲线：

```bash
python scripts/validate_diffraction_observables.py \
  --device cuda --tiny --pairs 2 \
  --output outputs/diffraction_validation/cuda_tiny
```

目标网格的少量取向对照可去掉 `--tiny`。默认检查 beta2；可使用 `--case spherical`、`--case beta3`、`--case skin` 和不同输出目录检查其余结构。`validation.json` 同时报告 full 与 self 的 CPU/CUDA 差。CPU 运行只标记 CPU 闭合，不冒充 CUDA 检验。

正式运行：

```bash
python -u scripts/run_diffraction_observables.py \
  --config configs/diffraction_observables_a100.yaml
```

完成后分析并导出图表和轻量报告包：

```bash
python scripts/analyze_diffraction_observables.py \
  --run outputs/diffraction_observables/a100_b05_float64
```

重复生产命令会检查并续算已有原子检查点。若相同目标已全部完成，校验通过后直接返回，保留原文件和校验和。`--max-new-pairs 4` 可限制一次新增对数，按所有案例合计；`--prepare-only` 只登记配置和取向库。可以用 `--n-orientations 64 --output outputs/diffraction_observables/pilot64` 先跑独立小样本，或在同一目录从较小目标追加到 32768。目标数必须为至少 4 的 4 的倍数，不能减少已登记的目标。

batch_size 和 y_batch_size 可调，后者只改变快度分块、不改变节点；生产中断最多丢失未写出的当前批次。改变物理参数、节点数、网格、精度、设备类型或 forward 源代码需要新目录。生产目录有写锁，避免多进程并写；分析也应在生产完成且没有追加作业时执行。运行错误会保留检查点，不会静默降为 CPU。

## 结果、回传与误差范围

生产文件保留每个取向 full/self 的原始系数和 b 子网格结果，以及取向矩阵、配置、代码指纹和校验和。新增分支使磁盘和 CPU 汇总内存高于原计算；按案例处理分析，不复制全体原始系数到报告包。按默认 32768 对、757 个径向点、三个 b 层级估算，检查点与案例汇总的数组合计约 17.7 GiB（压缩前），单案例最终汇总会同时持有约 8.9 GiB 数组及额外开销。建议至少预留 16 GiB 可用主机内存和 25 GiB 磁盘空间；这些是数组量估算，不是 A100 显存或耗时实测。请保留服务器上的整个生产目录，检查点用于续算，案例 NPZ 用于后续改分析。

分析输出在 `analysis_n32768`（实际取向数决定目录名）。首先查看同轴 D/J/C2、full/self 结构响应和窗口指标；同时查看 b 子网格差、嵌套取向前缀与独立半样本。CSV 保留原始均值/角矩及配对误差，NPZ 保留汇总协方差，JSON 登记窗口边界与来源。自动生成的 `analysis_n32768_reports.tar.gz` 可回传讨论，体积不包含大型逐取向生产文件；归档带清单和校验和。

所有 SE 都是取向积分的 Monte Carlo 标准误。球形与球形 skin 的取向 SE 为零，仅因为它们没有取向抽样误差。b 子网格差是数值诊断，不是严格误差上界；快度、密度/振幅网格、padding、角度、径向插值、探测器接受度/效率/背景/分辨率和样本统计仍需单独评价。本次未实现新的 detector 或 neutron-tag 模型。

## 本地交付检查（2026-10-18）

Python 3 / Torch 2.12.0 的 CPU 环境下，四个 `test_diffraction_*.py` 测试文件及原 `test_precision_forward.py`、`test_precision_analysis.py` 共 **40 passed，2 skipped**；跳过项为 CUDA。真实微型 forward 覆盖四个结构案例，经生产、分析、PNG/PDF/SVG、汇总协方差和报告归档完整走通，并检查篡改文件会被拒绝。另运行 CPU `--tiny --pairs 2` 验证脚本，确认 full/self 批量闭合，且新 full 与旧生产采用的 combined FFT 路径闭合。

这些检查没有运行默认 32768 对生产，不包含 A100 实测。服务器应执行上面的 CUDA 检查命令，再运行正式配置。代码包内置逐文件大小和 SHA256 清单，外置 `.sha256`；`--verify` 同时校验 gzip 尾部完整性，便于发现传输截断。
