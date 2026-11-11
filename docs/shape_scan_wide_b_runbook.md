# J/ψ 形状扫描的碰撞参数上限

从已有 fast_v2 工程根目录运行。`run_shape_scan_wide_b.py` 读取 `shape_scan_a100.yaml` 中的核、快度、取向库、动量网格和近区步长。宽窗上限由 `--cutoffs` 指定，默认依次为 60、120、240、480、1000、10000 fm。积分下限沿用 12 fm。各窗口分别保留 full 和 self 两个未归一化径向谱。

## A100 复算

将 `shape_scan_wide_b_20261111.tar.gz` 放在装有 fast_v2 的服务器仓库根目录，解包后做球形基准与一个最大形变取向的试算。

```bash
tar -xzf shape_scan_wide_b_20261111.tar.gz
python -u scripts/run_shape_scan_wide_b.py \
  --config configs/shape_scan_a100.yaml --device cuda \
  --case none__b2_000__b3_000 --case none__b2_009__b3_009 \
  --n-orientations 1 --batch-size 1 \
  --output outputs/shape_scan/wide_b_a100_pilot
```

在 `cutoff_metrics.csv` 中比较两种分支在每个上限的 `Q10_GeV`、`K_tail` 和有限保存动量域内的 `qdq_yield`。`*_over_first_cutoff` 以同一案例、同一分支的第一积分上限为分母。可以对照本地 CPU 基准复核 A100 输出，两边使用相同的有序取向。完整扫描另开目录。

```bash
python -u scripts/run_shape_scan_wide_b.py \
  --config configs/shape_scan_a100.yaml --device cuda \
  --all-cases --n-orientations 2048 --batch-size 1 --checkpoint-size 32 \
  --output outputs/shape_scan/wide_b_a100_grid300_n2048
```

`--batch-size 1` 是保守起点。可在小规模有形变的案例中实测吞吐量后调整。球形案例每种密度只需一对取向。变形案例以成对取向的未归一化谱求均值，再从均值谱提取阈值。程序每提交一块报告进度；中断后原命令续跑。每个案例的 `chunk_*.npz` 保存逐取向结果，`mean_A0.npz` 保存各截断的平均谱。汇总表在所选案例全部完成后写出。可用 `--step-scale 0.5` 在独立目录加密 (b) 求积。CPU 版本可用 `--device cpu --threads 2`。

## 如何比较新旧结果

旧 fast_v2 将光子场存储在边长 160 fm 的横向 FFT 盒子里。新算法将相同 Woods–Saxon 质子密度投影到横向平面，用自由空间的屏蔽库仑场核求两分量光子场，在靶核附近求振幅。质子电荷的纵向相位、反应厚度、遮蔽概率和快度权重沿用项目模型。双源相对平移相位在径向投影所用的实际动量与方位角上计算。

默认求积在 12–60、60–120、120–240、240–480、480–1000 fm 上的步长依次为 0.5、1、2、4、10 fm，在 1000–10000 fm 间使用 20 个对数间隔。每个截断单独使用 \(2\pi b\,db\) 梯形权重，共享边界只在对应区间接受半端点权重。相同求解器的 60 和 10000 fm 结果给出扩窗效应。旧 fast_v2 和新算法的 60 fm 结果给出同窗算法差异，两组比较使用相同案例、分支与取向。

`qdq_yield` 是保存域内的 \(\int q\,dq\,A_0(q)\)，在相邻节点间线性插值后精确积分，用于比较同一模型的相对谱权重。新运行直接写 `cutoff_metrics.csv`，其中的多档上限与旧 `analyze_shape_scan.py` 的步长层级含义不同。
