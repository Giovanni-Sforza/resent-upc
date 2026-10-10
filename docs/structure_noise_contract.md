# d 图模板：核结构与各向同性非相干背景

目标：直接观察形变与中子皮在 b 积分、衰变轴参考系中的形状响应，以及各向同性非相干背景如何改变这些响应。此图检验直观归因，不预设两种核结构只影响各自专属区域。

Python/matplotlib quantitative grid，主图两行五列。每行三张强度图依次为球形、取向平均的大形变、球形中子皮；右侧为形变/球形、中子皮/球形。第一行 coherent，第二行 coherent+isotropic incoherent。强度图先在同一 qT<0.17 GeV 圆盘中归一化，两行三列统一对数色标；比值图以 1 为中性色，每列上下行统一色标。基准低于其峰值 0.1% 或位于归一化圆盘外的比值像素留空，不加 epsilon。

采用 y=0、20<=b<=60 fm、2pi b db 积分、x 平行轻子 kT 的 d 图条件。形变核的两核独立取向，逐取向先形成振幅双线性量/强度，再分别完成衰变角与 b 积分，最后平均原始强度后归一化；不平均密度或复振幅，不平均各自归一化后的图。

新增非相干项采用 vendor 现有经验总概率及 B=4 GeV^-2 高斯径向分布，在衰变轴坐标中也假设各向同性。这是用户指定的噪声模型，不加入新的衰变生成器或 Good-Walker 方差模型。对所有 b 和核取向分别计算背景强度，与相干项相加后再归一化。保存窗口内非相干占比，不人为指定固定噪声百分比。

非相干横向求积域需覆盖 b_max/2 加核支持范围，因此本次显式扩大其 r_max，不能沿用固定 b 用途的默认 r_max=20 fm。检查粗细求积和两通道加和；检查形变取向样本前半/全部的差异。原始相干/非相干/总强度、逐取向图、比值有效掩码、配置、随机种子与取向矩阵一并保存。

中子皮定义为球形密度的 RMS 半径差 rn-rp，不把 WS 半径参数差直接当作皮厚；固定质子参数、通过数值径向积分解出中子 WS 半径。大形变保留既有 R、a，归一粒子数；不额外匹配球形 RMS 半径。

本次示例设置为 beta2=0.3、beta3=0.2；skin type 采用 ap=an=0.46 fm、Rp=5.02 fm、Rn=5.30084413 fm，对应 RMS 皮厚 0.20 fm。形变核采用 64 对独立 SO(3) 取向；两个随机流分别为 seed=20260928 与 20260929。纯相干及加背景两行共用同一组取向。保存全部逐取向相干图，取向图册使用统一的平均产额作为分母，确保图册各图的算术平均仍等于主图。

非相干求积域为半径 44 fm，256×512 个径向/方位节点；在每种核结构的首个取向批次（球形、皮厚各 1 对，形变 2 对）上对所有 b 加密到 512×1024 节点，比较总概率。取向与 b 收敛诊断分别比较前 32/全部 64 对，以及 21/41 个 b 节点。比值列各自采用以 1 为中心的对数对称色标，上下两行共用该列色标；所有有效像素均落在色标范围内，没有截断极值。

## 复现

```bash
python3 scripts/compare_structure_noise.py --output outputs/jpsi_structure_noise
python3 scripts/compare_structure_noise.py --render-only --output outputs/jpsi_structure_noise
python3 scripts/verify_structure_noise.py --output outputs/jpsi_structure_noise
python3 -m pytest -q tests/test_structure_noise.py tests/test_decay_axis.py tests/test_kt_distribution.py
```

配置为 `configs/jpsi_structure_noise.yaml`。小网格端到端检查使用 `--quick --output outputs/jpsi_structure_noise_smoke`，该结果不用于形变大小的物理解读。输出包括两行五列对数主图、64 个逐取向图及平均图的图册、径向强度与 v2(qT) 曲线，均为 PNG/PDF/SVG。原始数值和比值掩码保存在 npz，参数和核 RMS 半径等保存在 json。

导出 PNG/PDF/SVG，主图约 18×8.7 inch；另存逐取向图册及径向分布/v2 诊断。所有图为 Born level、小 qT/M、全极角接受度诊断，不是探测器级结果。

默认主图为 structure_noise_main.png/pdf/svg（与 *_log 内容一致），左侧六图全部对数着色，右侧比值的既有着色保持。旧 *_linear 文件为历史版本。

当前 J/psi 的两个动量分量均计算并显示至 ±0.2 GeV/c（425×425 像素，757 个径向节点覆盖方形角点），其余计算网格及物理参数沿用原设置。归一化和比值有效圆盘仍为 qT<0.17 GeV/c。b 下限改为 20 fm 后的结果保存于 outputs/jpsi_structure_noise_b20。
