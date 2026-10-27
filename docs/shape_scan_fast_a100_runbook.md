# Shape scan A0 专用内核：A100 验证、旧断点导入与续跑

本流程针对 `configs/shape_scan_a100.yaml` 的 300 点扫描。旧程序、配置和原生产目录保持可独立续跑；新内核在新的生产目录工作。先在实际服务器上验证旧、新内核的逐取向 A0 一致性并测量速度，再把旧计算的已提交前缀复制到新目录继续计算。

用户提供的近期基线为 A100 80 GB、每 128 对取向约 190.4 秒；一次 `nvidia-smi` 显示 GPU 利用率 99%、显存约 5 GB。这只说明采样时 GPU 正在执行任务，不能据此推断哪个算子最慢，也不能把空闲显存直接换算成可获得的加速比例。

交付包 `dist/shape_scan_fast_20261027.tar.gz` 仅含新增内核、运行/验证入口、测试、本文及验收摘要。把包传到服务器后，在原仓库根目录解压；包内不包含需要覆盖的原生产源码或 YAML。包内 `docs/shape_scan_fast_package_manifest.json` 保存新增文件摘要及参考源码摘要。随后按下面步骤验证和导入。原目录在此期间可保留，不能改名或清空。

```bash
tar -xzf /path/to/shape_scan_fast_20261027.tar.gz
```

## 1. 优化范围与保留的数值定义

新内核只计算本次扫描实际保存的 full/self A0：

- 跳过不参与四个单图观测量的 S1、S2、C、D。
- 只投影原有径向网格中 `q <= qmax` 的节点；不重新生成较短的径向网格。
- 缓存径向投影的插值索引与权重，并复用查询涉及的原生 FFT 节点。
- 用 Torch complex128 的分离 CZT 只求原生 FFT 网格上需要的频率矩形，再选取实际参与投影的顶点，避免生成整张随后被丢弃的动量图。

本次生产配置的原生网格为 `1280×1280`，保存的 455 个径向节点仅涉及 24,741 个不同的插值顶点，它们位于 `177×177` 的频率矩形内。CZT 不显式生成原生网格的其余部分；该数量变化不是整体运行时间的加速比例。

原来的空间网格、padding 对应的相位、双源振幅、FP64/complex128、b/y 求积、角度节点、取向库和 full/self 定义保留。CZT 计算的是同一个离散傅里叶和在原网格上的子集；没有截断空间输入，也没有插值复振幅。先在原生频率顶点形成 full/self 强度，再按原双线性权重投影。新内核只支持本轮使用的 FP64，不通过降低精度、减少取向数或稀疏化 b/y 网格提速。

新实现需要与原内核作数值闭合检查。不同运算排列可能产生浮点舍入差别，所以不能声称逐比特一致；验证报告记录误差与判据。闭合通过不等于物理校准或所有离散网格收敛。

独立变换测试使用原来的居中补零完整 FFT 作对照，覆盖非对称频率索引、重复顶点、空间偏移脉冲、近相消双源以及 `320 → 1280` 生产尺寸。它验证 CZT 没有错用频率、索引顺序或补零相位；完整 forward 的数值闭合由下面的服务器 benchmark 另行检查。

## 2. 在服务器上先测量并验证

在原生产终端等一条新的 `s/chunk` 日志后，按一次 `Ctrl+C`，等程序退出并返回命令提示符。已打印的进度对应已提交 checkpoint；尚未提交的部分最多重算一个 checkpoint。benchmark 和导入期间不要让原生产进程继续写原目录，也不要在同一张 GPU 同时运行其他重计算。

从仓库根目录、原 CUDA Python 环境运行以下命令。`--output` 必须是新目录或空目录：

```bash
python -u scripts/benchmark_shape_scan_fast.py \
  --config configs/shape_scan_a100.yaml \
  --batches 8 16 32 \
  --y-batches 1 5 \
  --warmup 1 \
  --repeats 2 \
  --output outputs/shape_scan/benchmark_fast_a100_20261027
```

benchmark 在相同输入上对照原 `DiffractionRefiner` 与新内核，检查保存范围内逐取向、full/self 和各 b 层级的原始 A0；另外覆盖球形点及 none/skin/halo 最大形变代表案例。它记录实际设备、源码摘要、科学配置、数值闭合结果、同步耗时和显存，生产程序使用这份报告核验当前执行条件。

只采用本机报告中通过闭合、成功完成且满足显存约束的方案。较大的 batch 不保证更快，`y_batch_size=5` 也不意味着五倍加速。若全部方案失败，保留旧生产目录，按旧命令继续运行；不要绕过验证要求。

## 3. 保留旧结果，先追加一个 checkpoint

下面 `<B>`、`<Y>` 分别替换为 benchmark 推荐的 `batch_size` 和 `y_batch_size`。验证报告路径以 benchmark 实际输出为准。

```bash
python -u scripts/run_shape_scan_fast.py \
  --config configs/shape_scan_a100.yaml \
  --validation outputs/shape_scan/benchmark_fast_a100_20261027/benchmark.json \
  --import-run outputs/shape_scan/a100_grid300_n2048_extendable \
  --output outputs/shape_scan/a100_grid300_n2048_fast \
  --batch-size <B> \
  --y-batch-size <Y> \
  --max-new-pairs 128
```

导入读取旧目录中已提交、校验通过的连续取向前缀，并在新目录复制、登记来源。原始 A0、取向顺序和已完成对数不会因为切换内核而被重新计算；新目录的元数据明确区分导入的旧内核数据与随后新算的数据。原目录不被改写。

`--max-new-pairs 128` 限制的是此次新增计算，不是导入总量。短试跑结束后，确认新目录中的总完成对数在导入进度上增加，并检查新的每块耗时。首次导入需读取、校验和复制旧数据，导入耗时不应混入 forward 加速比。

新目录需要为复制的旧数据及后续结果预留磁盘空间。不要手工改写 fingerprint、删除锁文件、移动部分 chunk，或把两种来源的文件直接拼进同一旧目录。

## 4. 继续完成新目录

短试跑正常结束后，去掉 `--import-run` 和 `--max-new-pairs`，其余参数保持一致：

```bash
python -u scripts/run_shape_scan_fast.py \
  --config configs/shape_scan_a100.yaml \
  --validation outputs/shape_scan/benchmark_fast_a100_20261027/benchmark.json \
  --output outputs/shape_scan/a100_grid300_n2048_fast \
  --batch-size <B> \
  --y-batch-size <Y>
```

之后仍可用 `Ctrl+C` 停止，并重新执行此命令从新目录已保存的 checkpoint 续算。不要在生产过程中改动源码或报告绑定的数值配置；需要改动时重新验证并遵循程序的来源校验。

如果中断发生在首次导入尚未完成时，保留 `--import-run` 重跑首次命令，程序会校验并继续既定来源快照的迁移。导入完成后的普通续跑不再需要旧目录。`checkpoint_size` 必须是所选 batch 的整数倍；默认 128 与上面的 8/16/32 均相容。

新目录生产完成后，使用原分析程序：

```bash
python scripts/analyze_shape_scan.py \
  --run outputs/shape_scan/a100_grid300_n2048_fast \
  --output outputs/shape_scan/analysis_n2048_fast
```

## 5. 回退与结果解释

旧目录仍可使用原命令续跑：

```bash
python -u scripts/run_shape_scan.py \
  --config configs/shape_scan_a100.yaml
```

旧命令继续处理旧目录；它不会自动吸收新目录后来追加的数据。避免同时运行两条生产命令争抢同一张 GPU。

最终速度结论应来自服务器 benchmark 和连续生产 checkpoint。报告中 CPU 微型网格的闭合结果只验证实现；CUDA、正式网格和实际 A100 吞吐以服务器实测为准。四观测量仍来自同一批 A0，联合统计与不确定度解释沿用原分析流程。

## 6. 本地数值核对记录

最终执行 `python -m pytest -q tests/test_shape_scan_*.py tests/test_diffraction_forward.py`：**115 passed、3 skipped**，三个跳过项均需 CUDA，本机未连接 CUDA GPU。真实 CLI 流程覆盖旧程序 7/39 对 → 新目录导入 7 对并新增 4 对 → 恢复完成 39/39 对 → 原分析程序处理 12 案例、四观测量。旧目录全部文件（含锁文件）SHA 未变，导入数组及数据摘要完全相同。交付摘要为 `docs/shape_scan_fast_validation_20261027.json`。

除微型网格测试外，已对正式配置的 `halo__b2_009__b3_009`、一个非平凡取向对执行完整 `97 b × 5 y` CPU 对照。保存 A0 的逐通道峰值归一最大差为 `1.140e-15`，逐 b/y 节点为 `2.026e-15`；六个 b 层级/分支通道的四观测量最大相对差为 `9.05e-16`，删失状态一致。原径向节点和取向数组逐元素相同。

这项诊断的旧、新内核单次时间分别为 198.20 s、35.79 s。它们来自两个有时间重叠的独立 CPU 进程，没有构成严格的预热性能基准，不能用于承诺 A100 加速比例。原始对照数组和误差报告保存在本地 `outputs/shape_scan/fast_validation_20261027/production_grid_cpu/`；交付摘要不包含这些大数组。
