# Fig. 9 球形中子皮厚度扫描

在服务器上的项目根目录、原形变扫描所用的 PyTorch/CUDA 环境中运行

```sh
git pull --ff-only
bash scripts/run_fig9_skin_scan.sh
```

默认设备为 CUDA。脚本以 UTC 时间和任务名生成 `run_id`，每个点完成后写出半径、谱和观测量，并在终端打印进度。可用 `FIG9_RUN_ID` 指定新运行名，用 `FIG9_PYTHON` 指定 Python 解释器。显式设置 `FIG9_DEVICE=cpu` 可在本地复算。

## 计算与作图

- 左列取球形 `96Zr`，固定 `R_p = 5.02 fm`、`a_n = a_p = 0.46 fm`。对每个目标皮厚求解中子半密度半径 `R_n`，横轴为中子与质子均方根半径之差。复用 0 和 0.20 fm 两个端点，新算 0.04、0.08、0.12、0.16 fm 四个点。球形点取一对单位旋转矩阵。
- 右列用原形变网格的无皮结果，保持质子和中子几何相同。横轴为 0 至 0.30 的十个 β₂ 节点，四条曲线取 β₃ = 0、1/15、2/15、0.20。除球形端点外，每点沿用原扫描的 2048 对取向。
- 两列使用原 fast_v2 全相干谱的相同积分和取点约定。碰撞参数窗口为 12 至 60 fm，步长 0.5 fm。快度取 |y| ≤ 1 的 5 点 Gauss–Legendre 积分，窗口内存活概率取 1。程序在同一条峰后下降支上对谱强度作线性插值，求峰高的 50%、10% 和 1% 交点。上排画 10% 交点 `p_T,10`（数据列 `Q10_GeV`），下排画 `K_tail = B_tail / B_mid`。
- 按原文 Fig. 9 排版，(a)、(b) 每组主图下方各有一排 Ratio。两列都以同一个球形无皮点（Δrₙₚ = β₂ = β₃ = 0）的相应观测量归一化，虚线标出 Ratio = 1。
- `configs/fig9/reference_grid.csv` 随代码提供。其 41 行取自旧图归档 `outputs/20261112_095608_pt10_notation_response/source_grid.csv` 的 `full` 分支。40 行用于右列，左列的无皮端点与右列共用，另 1 行为球形 0.20 fm 端点。

## 下载与重画

运行结束时，终端打印下载包的完整服务器路径 `outputs/_downloads/<run_id>.tar.gz`。把该文件下载到本机后，在本地项目根目录解压

```sh
mkdir -p outputs
tar -xzf /path/to/<run_id>.tar.gz -C outputs
```

`outputs/<run_id>/` 内的主要文件如下。

| 文件 | 内容 |
|---|---|
| `fig9_skin_deformation.pdf`、`.svg`、`.png` | 2×2 定稿图和预览 |
| `fig9_plot_data.csv` | 左列 6 点和右列 40 点的绝对值、两组 Ratio 及来源 |
| `new_skin_points.csv` | 4 个新点的目标和实际皮厚、WS 半径、均方根半径、交点和斜率 |
| `spectra/skin_*.npz`、配套 `.json` | 每点的动量网格 `q_GeV`、完整相干径向谱 `A0_full`、解出的密度参数和交点插值区间 |
| `reference_grid.csv`、`settings.json`、`environment.json` | 旧图输入表、计算参数和运行环境 |
| `run_info.txt`、`run.log` | 实际命令、Git 版本、运行状态和进度日志 |

遇到计算或绘图错误时，脚本以非零状态退出，归档已有的结果和日志，并在 `run_info.txt` 标为 `failed`。仅重画已下载的完整结果时，在本地项目根目录执行

```sh
run_id=实际运行名
python scripts/plot_fig9_skin_scan.py \
  --reference "outputs/$run_id/reference_grid.csv" \
  --new-points "outputs/$run_id/new_skin_points.csv" \
  --output "outputs/$run_id"
```
