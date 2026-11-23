# Fig. 9 球形中子皮厚度扫描

在服务器上的项目根目录、原形变扫描所用的 PyTorch/CUDA 环境中运行

```sh
git pull --ff-only
bash scripts/run_fig9_skin_scan.sh
```

默认设备为 CUDA。脚本以 UTC 时间生成 `<YYYYMMDD_HHMMSS>_fig9_skin_profiles` 运行名，在 `outputs/<run_id>/` 逐点写出半径、谱和观测量，并打印进度。运行只计算左列缺少的 9 个球形谱。可用 `FIG9_RUN_ID` 指定新运行名，或用 `FIG9_PYTHON` 指定 Python 解释器。设置 `FIG9_DEVICE=cpu` 可在本地复算。

## 计算与作图

- 左列取球形 `96Zr`，固定 `R_p = 5.02 fm`、`a_p = 0.46 fm`。横轴为中子与质子均方根半径之差 Δrₙₚ，依次取 0、0.04、0.08、0.12、0.16 和 0.20 fm。每个正皮厚取下表三组 Woods–Saxon 参数，零皮厚共用同一密度。球形计算只需一对单位旋转矩阵。

| 配对 | 固定量和求解量 | 本次计算 |
|---|---|---|
| 半径型 `radius` | 固定 aₙ = aₚ，解 Rₙ | 复用已有六个皮厚点 |
| 弥散型 `diffuseness` | 固定 Rₙ = Rₚ，解 aₙ | 新算 0.04、0.08、0.12、0.16 fm，复用 0.20 fm |
| 混合型 `mixed` | aₙ 取同一皮厚下前两组弥散度的算术中点，再解 Rₙ | 新算全部五个正皮厚点 |

- 左列实心点和连线为混合型的观测量，竖线覆盖这三组配对在同一皮厚下的最小值至最大值。
- 右列用原形变网格的无皮结果，保持质子和中子几何相同。横轴为 0 至 0.30 的十个 β₂ 节点，四条曲线取 β₃ = 0、1/15、2/15、0.20。竖线为取向平均的 ±1 个标准误差。原扫描对每个形变点的 2048 对取向按每 32 对分块，做了 1024 次 bootstrap 重采样，每次先平均谱，再提取观测量。球形点取向标准误差为零。
- 两列使用原 fast_v2 全相干谱的相同积分和取点约定。碰撞参数窗口为 12 至 60 fm，步长 0.5 fm。快度取 |y| ≤ 1 的 5 点 Gauss–Legendre 积分，窗口内存活概率取 1。程序在同一条峰后下降支上对谱强度作线性插值，求峰高的 50%、10% 和 1% 交点。上排画 10% 交点 `p_T,10`（数据列 `Q10_GeV`），下排画 `K_tail = B_tail / B_mid`。
- 按原文 Fig. 9 排版，(a)、(b) 每组主图下方各有一排 Ratio。主图的中心和竖线两端都除以球形无皮点的同一观测量，虚线标出 Ratio = 1。
- 两张输入表随代码提供。`configs/fig9/existing_skin_points.csv` 收录旧扫描的球形端点和前次固定弥散度计算的四个中间点。`configs/fig9/reference_grid.csv` 保留旧图 `outputs/20261112_095608_pt10_notation_response/source_grid.csv` 的中心值，取向标准误差来自 `outputs/shape_scan/review_n2048_fast_v2_20261027/analysis_n2048_fast_v2/metrics.csv` 的 `full`、0.5 fm 分支。

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
| `fig9_plot_data.csv` | 左列 6 点和右列 40 点的中心值、竖线种类、上下限及相应 Ratio |
| `fig9_skin_profiles.csv` | 全部 16 组不同的球形密度及其两个观测量，包括共用的零皮厚点 |
| `new_skin_points.csv` | 9 个新点的配对、目标和实际皮厚、WS 参数、均方根半径、交点和斜率 |
| `spectra/*.npz`、配套 `.json` | 每个新点的动量网格 `q_GeV`、完整相干径向谱 `A0_full`、解出的密度参数和交点插值区间 |
| `existing_skin_points.csv`、`reference_grid.csv`、`settings.json`、`environment.json` | 实际使用的两张输入表、计算参数和运行环境 |
| `run_info.txt`、`run.log` | 实际命令、Git 版本、运行状态和进度日志 |

遇到计算或绘图错误时，脚本以非零状态退出，归档已有的结果和日志，并在 `run_info.txt` 标为 `failed`。仅重画已下载的完整结果时，在本地项目根目录执行

```sh
run_id=实际运行名
python scripts/plot_fig9_skin_scan.py \
  --reference "outputs/$run_id/reference_grid.csv" \
  --existing-skin "outputs/$run_id/existing_skin_points.csv" \
  --new-points "outputs/$run_id/new_skin_points.csv" \
  --output "outputs/$run_id"
```
