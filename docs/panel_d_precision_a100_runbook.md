# Panel d 的 A100 / Torch 精度计算

本次增加独立的精度计算路径，不改动原 512 对取向结果。计算定义仍是先获得 Cartesian panel d，再进行圆周插值和像素求和。新路径支持 CUDA、真正从 FP64 网格和缓存开始的双精度计算、取向分批、快度分块、断点续算和取向数扩展。

当前交付是代码与本地数值检查。开发机器没有 CUDA，尚无 A100 的实测速度、显存数据或新高统计量物理结果。服务器运行完成后再判断较高 pT 区间是否稳定地区分形变和中子皮。

## 1. 默认计算内容

| 项目 | 默认值 / 含义 |
|---|---|
| 结构 | 球形；beta2=0.3；beta3=0.2；球形中子皮参数 skin_fm=0.2 |
| 取向 | 两种形变各 8192 对；两个核独立抽样，两种形变共用同一取向库 |
| 旧结果衔接 | 保留原 512 对旋转矩阵作为新取向库前缀，但所有振幅都按新精度重算 |
| b 积分 | 12–60 fm，最细步长 0.5 fm；同一计算保留 1、2 fm 网格结果 |
| 快度 | 物理单位，\|y\|<1，5 点 Gauss–Legendre |
| 数值类型 | float64 / complex128；关闭 TF32，不启用 AMP |
| 计算批次 | batch_size=1、y_batch_size=1；可在基准测试后调大 |
| 分析范围 | 0–0.17 GeV/c；重点查看 0.12–0.17 GeV/c |
| 暂定谱区间 | L=[0.03,0.08)，M=[0.08,0.13)，H=[0.13,0.17] GeV/c |

`skin_fm` 是沿用的模型参数标签，不在此重新解释为实验测得的 RMS 中子皮厚度。b 窗口仍是硬窗口诊断，不代表校准后的中子标记事件类。0.17 GeV/c 以上尚不包含在本配置中。

主要文件：

- `upc_deformation/precision_forward.py`：Torch 双源 FFT、b/y 积分与径向系数；固定投影查询缓存。
- `upc_deformation/precision_campaign.py`、`scripts/run_panel_d_precision.py`：取向库、检查点和生产任务。
- `scripts/validate_panel_d_precision.py`：同精度 CPU/CUDA 对照、批次一致性和计时。
- `scripts/analyze_panel_d_precision.py`：panel d 定义的 C2、S2、分区 C2、H/M、收敛诊断和图件。
- `scripts/make_panel_d_precision_configs.py`、`scripts/compare_panel_d_precision.py`：单项网格变更与配对比较。

## 2. 传到服务器后先做检查

可以直接复制本仓库，也可以使用打包文件。下述命令均在代码根目录执行：

```bash
tar -xzf panel_d_precision_a100.tar.gz
cd panel_d_precision
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_precision_*.py
```

使用服务器现有的 CUDA 版 PyTorch 环境。依赖列表在 `upc_deformation/requirements.txt`。本地检查使用 Python、Torch 2.12.0 CPU、NumPy 2.1.3、SciPy 1.15.3、Matplotlib 3.10.0、PyYAML 6.0.2；这不是服务器版本或 CUDA 兼容性声明。CUDA 不可用时程序明确报错，不会静默转为 CPU 后继续标记为 GPU 计算。

先用原 b 步长做同精度 CPU/CUDA 对照：

```bash
python scripts/validate_panel_d_precision.py \
  --device cuda --dtype both --pairs 2 --b-step 2 \
  --warmup 1 --repeats 1 \
  --output outputs/panel_d_precision_validation/a100_b2
```

再检查目标 b 步长和计划使用的快度分块：

```bash
python scripts/validate_panel_d_precision.py \
  --device cuda --dtype float64 --pairs 2 --b-step 0.5 \
  --y-batch-size 1 --warmup 1 --repeats 3 \
  --output outputs/panel_d_precision_validation/a100_b05
```

上述命令默认验证 beta2。用 `--case beta3`、`--case spherical`、`--case skin` 和不同输出目录分别复核其他结构。

`--pairs` 同时决定验证批次大小。调大到 4 或 8、改变 `--y-batch-size` 后，使用不同输出目录重新计时和核对；先从小批次开始。计时经过 CUDA 同步和预热，包含结果转回 CPU 的时间，不包含模型初始构建。报告保存 GPU 峰值显存；不根据显卡型号预估加速倍数。

同精度 CPU/CUDA 门槛：高 pT 单取向和平均 C2 的最大绝对差小于 1e-6；原始系数和图像相对峰值误差在 FP32 下小于 1e-4、FP64 下小于 1e-8。批量与逐个计算也核对。FP32 与 FP64 的差另列为精度诊断，不要求它们满足同精度门槛。看 `validation.json` 中 `cuda_validation_passed`，不能只看程序产生了图片。

## 3. 先做匹配取向的精度小样本

8192 是生产目标，不是“已经收敛”的条件。先用相同 64 对取向，比较一次只改变一个数值设置的结果：

```bash
python scripts/make_panel_d_precision_configs.py --n-orientations 64
python scripts/run_panel_d_precision.py --config configs/panel_d_precision_pilots/baseline.yaml
python scripts/run_panel_d_precision.py --config configs/panel_d_precision_pilots/y9.yaml
python scripts/compare_panel_d_precision.py \
  --reference outputs/panel_d_precision/pilots_n64/baseline \
  --candidate outputs/panel_d_precision/pilots_n64/y9 \
  --output outputs/panel_d_precision/comparisons_n64/y9
```

生成的变体包括：float32、y9、phi512、radial1513、b025、ws96、output480、padding8、pixels849。它们分别检查浮点精度、快度求积、径向投影角采样、径向节点、b 求积、核密度/厚度网格、振幅空间网格、FFT padding、最终 Cartesian 像素数。各变体与 baseline 使用完全相同的取向；扩大空间点数时保持物理盒子尺寸。

每项变体运行、比较的方式与上面的 y9 相同。每项设置变化都比较球形、形变和中子皮各自的 C2，并比较相对于本次球形基准的结构响应及 H/M。若某项变化显著影响目标区间，应先调整生产配置并复查，再投入 8192 对计算。64 对诊断只适合发现明显网格问题；取向相关的网格差仍需要检查配对 SE 或扩大样本。

`compare_panel_d_precision.py` 检查匹配取向和物理参数，保留两套计算自己的 Cartesian 像素网格，在相同 q 点计算差值。配对误差包含协方差。它不自动宣布物理收敛，不把差值 SE 当作偏差上界。

## 4. 生产、断点续算与追加样本

保留默认网格时：

```bash
python scripts/run_panel_d_precision.py \
  --config configs/panel_d_precision_a100.yaml \
  --batch-size 1 --y-batch-size 1
```

默认输出到 `outputs/panel_d_precision/a100_b05_float64`。球形和球形中子皮各计算一次，两种形变各计算 8192 对。程序按取向保存原子检查点；重复同一命令会验证已有检查点并继续缺失部分。中断后最多重算尚未保存的一批。

可先执行 `--prepare-only` 生成配置、取向库和清单而不运行 forward；也可使用 `--max-new-pairs 4` 限制一次会话新增的取向对数。限制按所有案例合计，球形案例也计入。未完成的生产不能被分析器误作完整结果。

在同一设置下增加到 32768 对：

```bash
python scripts/run_panel_d_precision.py \
  --config configs/panel_d_precision_a100.yaml \
  --n-orientations 32768 --batch-size 1 --y-batch-size 1
```

已有取向前缀和检查点保持不变。batch size、快度分块大小及目标样本数可以改变；快度节点数、b 步长、网格、dtype、设备类型和 forward 源代码变化要求新的输出目录。依赖版本和设备详情按会话记录，检查点绑定会话文件；正式生产尽量固定软件环境。不要同时向同一目录启动两个生产进程，也不要在追加生产的同时分析该目录。

原 512 对矩阵随小文件 `configs/assets/panel_d_orientation_prefix512.npz` 打包，并有 SHA256 校验。保留这些矩阵的原数值，不用它们的旧低精度产额；它们本来是 FP32 旋转，转成 FP64 后仍保留原抽样几何的舍入残差。若需要全新的 FP64 取向库，把 `import_bank` 和 `import_sha256` 都改为 null，并使用新输出目录。

## 5. 分析和绘图

```bash
python scripts/analyze_panel_d_precision.py \
  --run outputs/panel_d_precision/a100_b05_float64
```

8192 对结果写到 `analysis_n8192`；扩展后默认生成 `analysis_n32768`。输出：

- `C2_precision`：全区间 C2 和较高 pT 结构响应。
- `C2_b_grid_comparison`：同取向、同端点的 b=0.5/1/2 fm 步长比较。
- `C2_orientation_precision`：嵌套前缀与两份不重叠半样本。
- 每图有 PNG、PDF、SVG；另有 `report.md`、`summary.json`、原始数组与协方差 `arrays.npz`。
- `curves.csv`、`fixed_pT_diagnostics.csv`、`bin_metrics.csv`、`orientation_convergence.csv`、`b_grid_comparison.csv` 便于复核和后续手动选区间。

定义为先对取向平均原始角矩和产额，再取比值：

\[
C_2(q)=\frac{\langle N_2(q)\rangle_R}{\langle D(q)\rangle_R},\qquad
N_2(q)=\langle I_R(q,\alpha)\cos 2\alpha\rangle_\alpha.
\]

不是先把每个取向的 C2 做算术平均。SE 使用分子/分母协方差；曲线不同 pT 点的协方差一起保存。S2 把 cos 换成 sin，作为对称性和有限取向检查。固定 pT 表精确在指定 q 上评估，不从曲线上再插值；S2 没有被设成自动“通过”判据。相邻点、嵌套前缀有相关性，不能把每个 pT 点当独立实验来数显著性。

后处理将原来的像素重建、圆周双线性插值、求和合成为稀疏线性映射，以降低大量取向的分析成本；每个案例先与原逐像素实现做数值闭合。H/M 与三个区间内 C2 仍直接对 panel d 像素积分。

重点判断：较高 pT 的结构响应是否同时对取向增加、b/y/空间网格加密、FP64 设置稳定；C2 是否提供与 H/M 不同的角向信息。粗细网格差不等于精细网格误差上界，阴影只代表取向积分 SE，尚未加入模型不确定度、探测器响应或实验事件数。

## 6. 回传结果与重新打包

回传验证目录中的报告、JSON，以及生产清单、effective_config.yaml、sessions、四个案例 NPZ 和 analysis 目录即可支持复核。若还需要迁移续算，则一并保留取向库和 checkpoints 目录。

本地重新打包：

```bash
python scripts/package_panel_d_precision.py
```

产生 `dist/panel_d_precision_a100.tar.gz` 和逐文件/压缩包 SHA256 清单。包含代码、测试、配置、旧取向前缀；不包含历史大体积结果或服务器凭据。
