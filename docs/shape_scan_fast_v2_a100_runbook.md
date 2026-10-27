# Shape scan 第二轮优化：空间支撑裁剪与积分后投影

第二版保留原生产程序和第一版 fast 程序，使用独立的 `run_shape_scan_fast_v2.py` 入口与新输出目录。`--import-run` 支持原始生产目录或第一版 fast 目录：校验、复制已有连续取向前缀，不重新计算已保存的 A0。源目录的数组、元数据和锁文件不被改写。

交付包为 `dist/shape_scan_fast_v2_20261027.tar.gz`，旁边的 `.sha256` 文件记录压缩包摘要。把两者传到服务器仓库根目录，当前计算正常停止后安装：

```bash
sha256sum -c shape_scan_fast_v2_20261027.tar.gz.sha256
tar -xzf shape_scan_fast_v2_20261027.tar.gz
```

包内包含第二版新增文件和它依赖的第一版新增文件，第一版文件内容与原交付包逐字节相同；不包含原生产源码或配置。可用随包 manifest 核对执行文件与交付内容：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('docs/shape_scan_fast_v2_package_manifest.json').read_text())
for group in ('payload_sha256', 'all_execution_source_sha256'):
    for name, expected in manifest[group].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'SHA256 mismatch: {name}')
print('Package and execution sources verified')
PY
```

## 1. 保留的计算定义

原 FP64/complex128、空间采样点、取向库、原生 Fourier 频率、b/y 求积、角度和径向节点均保留。

- 空间支撑来自原 `grid_sample` 零填充双线性插值的有效邻点。目标厚度插值严格为零时，Gamma 为零，因此对应源振幅严格为零。只在可能非零的矩形内计算振幅，不用强度阈值切掉尾部。
- 空间矩形保留它在原输出网格中的整数起点。每个源的傅里叶相位包含原补零起点及自身裁剪起点，随后才合成双源干涉。
- 径向投影与 b/y 加权积分均为线性操作，可以先形成原生频率顶点上的 full/self 强度、完成积分，再做三组 b 层级的径向投影。不能先平均复振幅，也不能在平方之前插值。
- `return_nodes=True` 的诊断路径额外保存逐 b/y 节点投影；生产路径不需要这些中间结果。性能比较必须采用生产的 `return_nodes=False`。
- 复数实/虚分量在每个 y 批次预先打包为通道，跨 b 复用；归一化查询坐标在构造时缓存。普通厚度与其归一化因子跨 y 复用，截面和运动学因子跨 b 复用，遮蔽概率跨 y 复用。离散厚度支撑完全不相交时概率严格为 1。

浮点归约顺序不同，需要重新验证数值一致性。前一版的 benchmark 报告不能作为第二版的执行凭证。

## 2. 测量服务器上的实际收益

先等当前生产程序出现一条新的 checkpoint 日志，按一次 Ctrl+C，等命令提示符返回。使用原 CUDA Python 环境，在原仓库根目录运行：

```bash
python -u scripts/benchmark_shape_scan_fast_v2.py \
  --config configs/shape_scan_a100.yaml \
  --batches 8 16 32 \
  --y-batches 1 5 \
  --warmup 1 --repeats 2 \
  --output outputs/shape_scan/benchmark_fast_v2_a100_20261027
```

输出目录必须不存在或为空。脚本会：

1. 在生产配置上比较第二版与原始内核的逐取向 A0，覆盖各密度代表点、球形与形变、full/self、全部 b 层级。
2. 检查四个单图观测量及删失状态的一致性。
3. 分别测量原始内核、第一版 fast 和第二版的同步耗时，输出 `speedup_vs_reference` 与 `speedup_vs_v1`。
4. 仅在数值检查通过、显存满足限制的实测方案中推荐批大小。

不要在同一卡同时运行生产和 benchmark。报告绑定科学配置、执行源码、设备与已测过的 batch/y_batch，不能拿 CPU 报告代替 CUDA 报告。实际效果以服务器报告及连续生产 checkpoint 为准。

## 3. 迁入旧结果，先追加 128 对

下面将 `<B>`、`<Y>` 替换为报告推荐的批大小；`--import-run` 选择当前进度最多、已经停止运行的那个完整目录。例子使用原始生产目录：

```bash
python -u scripts/run_shape_scan_fast_v2.py \
  --config configs/shape_scan_a100.yaml \
  --validation outputs/shape_scan/benchmark_fast_v2_a100_20261027/benchmark.json \
  --import-run outputs/shape_scan/a100_grid300_n2048_extendable \
  --output outputs/shape_scan/a100_grid300_n2048_fast_v2 \
  --batch-size <B> --y-batch-size <Y> \
  --max-new-pairs 128
```

如果第一版已经算得更多，改用 `--import-run outputs/shape_scan/a100_grid300_n2048_fast`。一次只能导入一个完整来源目录；不要手工拼接不同来源的 chunk。默认 checkpoint 大小 128 应为所选 batch 的整数倍。

`--max-new-pairs` 只计新增 forward，不计导入对数。首次迁移需要读取、校验并复制已有数据，预留相应磁盘空间；这段时间不属于计算内核耗时。源与目标必须为独立、互不嵌套的目录。

如果迁移途中中断，保留原 `--import-run` 命令重跑，继续同一个来源快照；迁移完成后不再依赖源目录。所有原始数组、取向顺序和数据摘要保持不变，新封装明确记录来源内核和原 chunk/session 摘要。

## 4. 续跑与分析

短试跑完成后，去掉首次迁移和新增对数上限：

```bash
python -u scripts/run_shape_scan_fast_v2.py \
  --config configs/shape_scan_a100.yaml \
  --validation outputs/shape_scan/benchmark_fast_v2_a100_20261027/benchmark.json \
  --output outputs/shape_scan/a100_grid300_n2048_fast_v2 \
  --batch-size <B> --y-batch-size <Y>
```

以后重复该命令会校验并续算。生产结束后仍用原分析脚本：

```bash
python scripts/analyze_shape_scan.py \
  --run outputs/shape_scan/a100_grid300_n2048_fast_v2 \
  --output outputs/shape_scan/analysis_n2048_fast_v2
```

原程序及第一版程序仍能在各自原目录续跑；它们不会自动吸收第二版后来新增的数据。运行中的源码和验证报告需要保持不变。实现数值一致性不代表物理校准、取向统计量足够或所有积分网格已收敛。

## 5. 如何看新的耗时日志

每个 checkpoint 的日志同时显示三项时间：原有第一个 `s/chunk` 仍只累计 forward；`wall` 从该块开始到数据及进度文件保存完毕；`nonforward` 是二者之差，包含数据检查、压缩、fsync 和 manifest 等开销。首次导入、case 构造和进程启动校验不计入该块的 wall。

若内核已经明显变快、`nonforward` 却接近或超过 forward，应检查共享文件系统或 checkpoint 设置，不要仅按 GPU 利用率判断。batch/y_batch 的推荐需要来自本轮服务器实测。

## 6. 本地验证记录（2026-10-27）

- 回归测试：`python -m pytest -q tests/test_shape_scan_*.py tests/test_diffraction_forward.py`，178 通过、5 项 CUDA 测试因无 CUDA 设备跳过。
- 生产网格核对：halo、β2=0.3、β3=0.2、一对非平凡取向，97 个 b 节点、5 个 y 节点、455 个保存径向点。积分后 A0 与原始实现的最大误差/各通道峰值为 `1.52e-15`，逐 b/y 节点为 `1.52e-15`；观测量检查通过。原始参考来自此前归档，并核对了数组文件及执行源码 SHA256。
- 同一生产案例、CPU 单线程、batch=1、y_batch=1、`return_nodes=False`：各预热一次、交错计时两次，第一版中位 `25.9115 s`，第二版 `2.9996 s`，第二版相对第一版约 `8.64×`。这是单个 CPU 案例，不能用来换算 A100 的 `190 s/chunk`，也不能当作整场扫描的加速比。
- 空间计算由每源 `320×320` 点缩至 `48×48` 或 `49×48` 的严格有效支撑；仍保留原原生 Fourier 频率。
- 真实 CLI 验证了原始来源 `7 → 导入后追加至 11 → 续跑至 39`，以及含原始祖先数据的第一版来源 `11 → 15 → 39`。均由原分析器完成 12 案例、4 个单图观测量；来源目录所有文件不变，导入数组及摘要一致，第一版祖先档案完整保留。

详细记录见 `docs/shape_scan_fast_v2_validation_20261027.json`。本地证据仅覆盖实现一致性与迁移流程；A100 性能和数值检查仍需执行第 2 节的服务器 benchmark。
