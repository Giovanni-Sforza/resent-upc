# Fig. 7 机制到可测投影：本轮计算约定

用户在 2026-10-07 指定以 `upc_arXiv_v1.pdf` 为论文版本，跳过新旧前向基准追溯，直接实施机制桥接及候选检验。正文、Fig. 1–7 和原网络保持不动。

后续按用户更正，固定b与积分下限均改为12 fm，最新输出为 `b12_20261007`。此前20–60 fm结果保留为历史对照。原“工字型”暗条纹包含高pT上下横臂；暗区位于外围不是淘汰理由，不能仅由半径将其归因为衍射尾部。

## 图的证据逻辑

核心问题：Fig. 7 的暗区填充和半高亮区尺度，经过轻子参考轴及 b、y 平均后，是否仍能由固定窗口产额和径向动量尺度追踪？不预设两者完全解耦或实验可达。

Python/matplotlib，图像与定量曲线结合，PNG/PDF/SVG 三种导出。每个样本保存原始线性强度、逐 b/y 节点系数、取向样本及统计量。颜色不作为计算输入。新图像指标是当前诊断前向下的重新计算，不宣称数值复现 Fig. 7。

1. baseline_bridge：球形分布按真实 b 轴、轻子轴、b 平均、b/y 平均依次显示，归一化范围相同。
2. structure_bridge_ratios：独立 β2、β3、skin 对照；逐行统一比值范围，低基准产额区标灰。
3. mechanism_radial：完整与去交叉项分布；归一化谱与原始 full/self 分开，后者诊断相消程度。
4. feature_translation：固定暗区平均亮度与 Fdip、半高亮区面积平方根与 mean qT 的关系，区分完整与去交叉项结果。
5. response_screening：两条物理量及辅助 C2 的轨迹响应、数值误差与条件事件数。数值不显著的结构差不估计3σ事件需求。
6. dark_region_audit：二维暗区轮廓与原始强度交叉项K=(full−self)/self并列。保留高pT横臂的方向信息，以交叉项判断相消/相长，不能以半径代替机制判据。各阶段独立选取的暗mask是分阶段诊断，不是同一像素区域的严格输运。

## 物理、平均与机制对照

- 96Zr+96Zr，200 GeV，沿用 physical 单位模式；本轮不比较旧稿生产链。
- 主样本 b=12–60 fm，25节点（步长2 fm），固定b=12 fm；2πb db，|y|<1 的 Gauss–Legendre 求积。光子场已经在产生振幅中，不重复乘通量。y 积分节点不表示事件在 y 中均匀。
- 这是登记的有限 b 硬区间诊断样本。无中子类别、平滑强子生存概率或探测器接受度；不冒充完整 inclusive 预测。
- 两核取向独立；逐取向先形成 Stokes 双线性量，随后按原始产额平均，最后归一化。
- full=Stokes(M21+M12)，self=Stokes(M21)+Stokes(M12)。self 仅去掉两源交叉项，保留单源核相干衍射，不是 incoherent 核破裂。
- 归一化后 full/self 的指标差为机制干预对照，并非非线性指标的加法分解。原始强度 full-self 才是交叉项的加法贡献。
- Born level、横向偏振、无质量轻子、小 qT/M 近似、完整极角接受度，x 平行 kT。有限 y 本身可能填暗区，所有结构比较共用相同 y 权重。
- 原经验非相干高斯模板作为单独背景检验，并在衰变轴中假设各向同性。

## 指标与预先登记的参数

统一分析圆盘 qT<0.17 GeV/c。主干涉候选 Fdip=N(qT<0.02)/N(qT<0.17)，q0=0.01、0.03、0.05 只作窗口依赖，不根据保留验证点重新选窗。主衍射候选 mean qT；C2=<cos2alpha> 为辅助诊断。对 C2 报绝对差。

半高指标 A1/2 是图像超过半峰值的面积。fringe_low5 是圆盘内最暗5%像素的平均归一化强度。dark_fixed_mean 使用球形完整图像预先定义的5%暗区，并在同一阶段所有参数与机制分支中冻结。圆盘约定与原稿可能不同，故标为 Fig.7-inspired diagnostics。

探索点：球形；β2=0.15、0.30；β3=0.10、0.20；skin=0.10、0.20fm。另以 β2=0.15、β3=0.10、skin=0.10fm 为预登记混合验证点。皮厚用三维形变密度的实际 RMS 差求解中子 WS 半径，不以半密度半径差代替。

## 响应与误差

轻子横动量采用 kT=(M/2)sqrt(1-u²)、p+ = k+q/2、p- = -k+q/2，联合角密度保留偏振条件相关。每条轨迹施加独立相对动量 Gaussian 误差后，由同一轻子对重新计算 q、k、alpha。0、0.5%、1% 是未校准 toy 场景；不包含材料辐射、效率或完整重建，不能称为解决实验探测器响应。

使用真值支持至约0.283 GeV/c以包含 qmax 外迁入，截断显式登记。重要性抽样事件是数值求积点，不能等同于实验样本数。

输出分开保存有限取向误差、响应数值MC误差和给定选中事件数 N 的物理统计误差。对每个取向的原始角密度在同一响应样本上重新加权，传播取向积分误差。共同随机数用于结构差，保持数值误差相关。

条件3σ事件数为单一样本对精确模型模板的估算，不含理论、背景分离、参数简并与实验系统误差。当结构差不超过总数值误差3倍时，不报告有限的事件需求。

本轮首先保证代码和代数不变量闭合，并给出 b 网格、取向积分等数值精度诊断。图像结构敏感性、机制联系、数值可分辨程度和实验可行性是不同结论。

## 代码与复现入口

- `configs/jpsi_observables_revision.yaml`：冻结物理、求积、参数点与窗口。
- `scripts/analyze_jpsi_observables.py`：按样本缓存的前向计算、提取、响应与成图入口。
- `upc_deformation/source_amplitudes.py`：两源 Stokes 及去交叉项对照。
- `upc_deformation/observables.py`：密度RMS、几何/取向积分及候选提取。
- `upc_deformation/observable_b_extension.py`：复用已验证20–60 fm缓存、补算12–20 fm并逐取向拼接原始积分；b=20端点两侧各给半权重。
- `upc_deformation/observable_response.py`：相关轻子响应、条件统计与数值MC误差。
- `upc_deformation/observable_diagnostics.py`：机制交互、暗区位置、窗口与混合点检查。
- `upc_deformation/observable_dark_regions.py`：固定球形暗区中的原始自项/交叉项、同分母结构差与配对取向误差。
- `upc_deformation/observable_figures.py`：Python/matplotlib 六组图。
- `scripts/verify_jpsi_observables.py`：已保存数据的代数与数值精度核查。
- `scripts/check_observable_convergence.py`：四个匹配取向的定向加密检查。

```bash
cd /Users/qiuyu/resent-upc
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_jpsi_observables.py --output outputs/jpsi_observables_revision/b12_20261007 --extend-from outputs/jpsi_observables_revision/main_20261007
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_jpsi_observables.py --render-only --output outputs/jpsi_observables_revision/b12_20261007
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/verify_jpsi_observables.py outputs/jpsi_observables_revision/b12_20261007
```

首次命令会运行较长的CPU前向，完成的样本可重用。物理代码哈希或冻结配置不同会拒绝混用缓存；新物理设置应使用新目录。`--render-only` 使用完整冻结配置，仅重做分析/响应与图片，不声称重新运行旧缓存的物理。形变样本的32组取向只属于首轮采样规模，不能据此直接声称收敛。

为核查窗口选择，还从相同缓存报告原稿的q0=0.005 GeV/c，单独标为“原稿窗口参照”；它没有替代预登记主窗口。固定b也改为12 fm并不等于核对了原稿整套生成链。

旧b20样本的四取向快度/空间加密检查只覆盖旧区间，不能认证新增12–20 fm。新缓存保留b=20重叠节点闭合证据，另用球形生产网格直接重算25节点核查增量积分。新增区域表采用同一原始强度归一化，记录净交叉项、局部绝对交叉项，以及自项/交叉项结构响应；自项响应非零本身不是否定干涉贡献的依据。
