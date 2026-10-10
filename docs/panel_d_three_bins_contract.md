# 由中子皮角向换向确定三个动量区间

目的：直接统计原 panel d 在中子皮角向图两条径向白线划分的三个区间中的 C2，比较球形参照、四极形变、八极形变和中子皮。

分界规则：由已有线性密度的完整圆周条件矩计算 Delta C2(q)=C2_skin(q)-C2_sphere(q)，取 q>0.01 GeV/c 的两次换号点；排除原点的数值零，但第一区间仍从零开始。分界约为 0.11418、0.15919 GeV/c；上限为原图 0.17 GeV/c。所有样本共用中子皮定义的区间，不能分别选边界。白线是角向相对变化为零，不等同 C2 本身为零。本轮为图启发的探索性划分。

统计：主结果从原 Cartesian normalized_images[3,0] 像素求和，C2=sum I cos(2alpha)/sum I，S2 类似；原点 harmonic 取零。每个环带按强度加权，不平均比值像素、不等权平均 C2(q)、不使用低亮度显示掩膜。给出 C2、相对球形参照的 Delta C2 和区间产额份额。极坐标 q dq 积分仅作离散交叉核对。有限取向误差由已有32对原始取向的同一 panel-d 图表达重新投影到同一像素网格后估算，不运行前向模型，不换用64取向结果。球形和中子皮的零取向误差不代表无系统误差。

来源：outputs/jpsi_observables_revision/b12_20261007，以及 outputs/panel_d_ratio_decomposition/b12_original32_20261009/decomposition.npz。原 b=12–60 fm、步长2 fm、|y|<1，完整相干信号，x 沿轻子参考方向。

图件合同：Python/matplotlib；image plate + quant。顶部中子皮条件角向热图标出三段区间，下面两幅点图分别显示 C2 和 Delta C2；颜色按结构固定。取向误差棒明确标为1标准误，与实验统计误差区分。讨论宽度约215 mm，PNG/PDF/SVG，可编辑文字；CSV/NPZ、计算说明、输入哈希和检查结果可复现。尾部事件份额单独列出，不能只凭相对色差判断测量灵敏度。旧数据、旧图及正文不修改。

输出：outputs/panel_d_three_bins/b12_original32_skin_boundaries_20261009。
