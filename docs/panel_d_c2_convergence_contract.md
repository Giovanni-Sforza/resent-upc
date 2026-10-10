# C2 的512对取向积分与收敛检查

目的：提高原panel d的取向积分精度，重新比较四极形变、八极形变和球形中子皮的C2(pT)及相对球形差值。结果允许显示512仍未充分收敛，不预先判定C2有效或无效。

物理范围保持原设置：b=12:2:60 fm，|y|<1的5点Gauss积分，原物理单位桥接、64³密度网格、320²振幅网格及4倍FFT填充、757径向点和256产生方位点。形变参数分别beta2=.3与beta3=.2；球形与skin=.2 fm继续使用确定性缓存。无网络训练、响应模拟或实验误差估计。

取向样本：按顺序保留原32对、已核验独立扩展32对，再添加14批各32对独立SO(3)取向，共512对。两个核分别生成取向；相同取向银行用于两类形变。保存全部旋转矩阵，检查重复，不依赖改变随机数组形状时的隐式前缀行为。新样本每对保存检查点，可恢复运行；旧数据不修改。

计算优化复用与b无关的密度、厚度和光子场，省略本任务不使用的经验背景与source-self分支。使用相同底层物理函数；总振幅FFT的线性合并须对照存档取向，证明差异远小于拟估计的C2误差后方可用于生产。另经实际取向回归验证，CZT可直接计算原1280² FFT网格后续所需的297²频率点；不改变频率、积分节点或插值权重，仍先形成Stokes再插值。此前完成的78个完整FFT检查点保留原哈希与来源，其余818个新取向用验证后的CZT实现。前向代码、输入配置、实现来源和取向银行均记录哈希。

主观测量严格沿用Cartesian panel d→完整圆周双线性取样→强度加权C2，不用解析谐波比替换原离散投影。先平均原始强度，再求角矩比值。图用原341个q点和720个圆周角点，排除方向未定义的原点。误差是考虑分子分母协方差的比值delta-method点态1SE；不是事件涨落带或同时置信带。

收敛证据：画实际嵌套32/64/128/256/512前缀中心值与SE；比较两组互不重叠的256对取向；额外保存4个128块和固定L/M/H区间的C2。独立半样本差可按sqrt(SE1²+SE2²)标准化，嵌套前缀不能这样处理。相邻q点相关，不把最大点态z值称为整体显著性检验。对照SE32/4仅为Monte Carlo理想缩放参考。

图件采用已有Python/matplotlib风格：更新的三条C2曲线及512误差带；三结构分面DeltaC2及原32中心线；两种形变的前缀、误差缩小和独立半样本对照。讨论宽度约210–230mm，PNG/PDF/SVG保留可编辑文本，导出CSV、NPZ、配置、报告及数值/图件QA。

最终输出独立保存于outputs/panel_d_c2_convergence/b12_512_czt_20261010；早期完整FFT检查点保存在outputs/panel_d_c2_convergence/b12_512_20261010。取向收敛与原b求积、网格、前向模型和实验误差分开报告。

运行入口（保留存档缓存及验证文件）：

```bash
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/refine_panel_d_512_czt.py --workers 4
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_c2_convergence.py --config configs/panel_d_c2_convergence_czt.yaml
```

`--workers`仅覆盖并行进程数，实际数量记录于生产manifest；单次前向计算的2线程设置和已验证内核保持不变，不改变取向银行及科学配置身份。

CZT回归验证可独立复现，包含原八极取向0、球形取向0，以及新四极取向64对完整FFT检查点的比较；验证原Cartesian投影后的C2、图像和系数，并运行CZT单元测试。为保留生产门槛文件，复现输出指定到独立目录：

```bash
OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/validate_orientation_czt.py --output outputs/panel_d_c2_convergence/czt_validation_replay
```

验证器默认指向本次CZT结果目录；若该目录仍在生产，则拒绝重写其中的验证文件。独立输出不修改原缓存、内核或检查点。
