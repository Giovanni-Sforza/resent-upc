# 有限pT角关联：计算与成图约定

用户于2026-10-08同意将原暗条纹亮度推进为同一动量环带内的角向明暗对比，并要求实现代码与绘图。沿用Python前向与matplotlib，保留b=12 fm及12–60 fm、|y|<1设置，不改论文正文或训练网络。

核心检验：原形变干涉信息在可重建轻子参考轴中是否仍表现为有限qT的角向反差，并且相对于数值误差、轨迹响应和背景具有可解释的响应。qT即J/ψ对横动量，kT=(pT,+−pT,−)/2，alpha=phi_q−phi_k。C2=<cos2alpha>是轻子轴角关联，不是集体流，也不假定beta3对应三阶谐波。

固定环带为0.03–0.06、0.06–0.10、0.10–0.14、0.14–0.17 GeV/c。它们来自此前讨论的探索窗口，本轮不根据显著性调窗，不称为盲验证。分析窗口内的归一化范围仍为qT<0.17。

明暗对比Aphi=(N_perp−N_parallel)/(N_perp+N_parallel)，平行/垂直方向各含两块半宽pi/4的对称扇区。理想当前二阶衰变模型给出Aphi=−4C2/pi；它们是同一角向信息的两种表达。完整径迹响应可引入更高谐波，重建层不强迫该等式成立。S2=<sin2alpha>记录为对称性及取向积分诊断。

原始径向系数保留q dq积分测度，先积分强度后取比值。full/self的结构差为机制干预对照，不能解释为归一化角系数的唯一加法分解。self仍保留单源相干衍射及偏振各向异性；外围暗区的机制由交叉项核查，不能用半径判定。

基础八点复用b12缓存；beta2=0.30和beta3=0.20各新增32对独立核取向，合并为64对，种子20261018/20261019，与原20261007/20261008分离。其他形变点保持32对，球形及纯skin点为确定性1点。取向前缀及独立两半比较检验采样稳定性；64点不自动意味着收敛。保存跨环带、跨full/self的配对协方差。

响应采用相同轻子对重建q和k，保留相关误差与环带迁移。轨迹相对动量误差0、0.5%、1%以及0.5%配合1 mrad方位误差均为未标定toy场景，无接受度、材料辐射或完整探测器。有限真值支持约0.283 GeV/c之外的迁入未建模。数值MC、取向误差与条件事件统计分别保存。

每例使用1,000,000个共同随机求积点。主响应曲线使用真值控制变量：解析真值+MC响应值−MC零响应值。结构差的MC协方差由四项共同随机影响函数计算，取向协方差由解析真值取向影响加响应增量计算。零响应MC方差严格为零，但取向误差保留。固定系数的真值锚定不保证降低非零响应结构差的MC方差，部分场景会增大；两套SE均保留。真实事件涨落并未降低，仍使用raw分支的事件协方差。有限样本比值偏差及二阶混合误差没有因此消失。raw与truth_anchored两套结果均导出，不使用“探测器校准”称呼该方法。

五组图为定量图组：角关联机制及产额；四环带结构响应；原图暗纹/宽度与角关联/平均动量的联系；联合径迹响应；取向采样精度。图间颜色统一，每幅明确误差和分母。PNG供讨论、PDF/SVG保留可编辑文字，目标宽度约210–230mm；投稿前另按栏宽排版。所有图由线性数值绘制，不从颜色反演数据，不把趋势线当机制证明。

## 入口与产物

- 配置：[jpsi_angular_observables.yaml](../configs/jpsi_angular_observables.yaml)。
- 独立取向补算：[refine_angular_orientations.py](../scripts/refine_angular_orientations.py)。
- 分析及成图：[analyze_jpsi_angular.py](../scripts/analyze_jpsi_angular.py)。
- 保存数据校验：[verify_angular_observables.py](../scripts/verify_angular_observables.py)。
- 球形/纯skin的b网格加密：[check_angular_b_quadrature.py](../scripts/check_angular_b_quadrature.py)。只检验这两例，不认证形变端点。
- 当前结果目录：`outputs/jpsi_angular_revision/b12_20261008`；旧b12缓存不覆盖。

```bash
cd /Users/qiuyu/resent-upc
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/refine_angular_orientations.py
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/check_angular_b_quadrature.py --threads 2
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_jpsi_angular.py
/opt/anaconda3/bin/python3 scripts/verify_angular_observables.py outputs/jpsi_angular_revision/b12_20261008
```

纯成图使用分析入口的`--render-only`，只读已保存摘要。新配置或MC样本数应写到新目录；`--base-only`可复用原32取向作开发检查，不能冒充新增64端点结果。

数值校验把代数闭合与求积精度分开：不变量失败记为失败；取向未分辨、粗b变化及实验响应缺项单独说明。协方差保留C2与Aphi的理想线性冗余，不用奇异信息矩阵虚增约束。2026-10-08全套72项代码测试通过；最终保存数据和成图QA另见结果目录。
