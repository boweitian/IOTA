# Router 方向与题目分布

绑定 Qwen3.5-0.8B，block 15，1024 维；包括 GSM8K 1300 题、BBH 三子任务 540 题、IFEval 500 题。这三个 benchmark 已有冻结 Router；Navigator 原有五 benchmark 图谱保留。

默认横轴严格等于训练最佳固定配置的有符号单位方向。也可切换为选中题的最佳代表或 Router 方向；所有题目随之在同一坐标系重投影。先减去训练均值，移除横轴分量，再仅在训练题上拟合两个正交 PCA 轴。标签不参与剩余 PCA 的拟合，题目状态和方向向量来自同一层。默认参照方向由训练集确定；选中题的最佳方向模式使用了该题的评分，属于事后诊断。页面支持各轴适配与真实等单位两种显示比例，坐标不变；向量角度比较应使用真实等单位。

各子目录中：
- data.json：逐题选择、成功集合和结果；
- aligned_projection.npz：默认均值、坐标基、坐标、全部原生 steering 向量及其投影；
- all_direction_axes.npz：各方向坐标系、配置到坐标系的 mapping 和 signs、题目坐标；对配置 j，其有向基第一行为 axes[mapping[j],0] * signs[j]，其余行保持原向；none 没有方向；
- direction_matrix.npz：题目 × 配置的评分，-1 未测、0 失败、1 成功；
- questions.csv：全部题目、最佳代表和新 Router 的选择；
- train/test_distribution.png 和 SVG：四方法对照，默认最佳固定方向对齐。

二元评分存在并列最优。代表方向按训练支持数选择，完整成功集合全部保留；代表频次会偏向常见方向，因此页面也提供全部成功集合计数。109 配置均失败时不编造成功方向，未测全时显式标记。

测试集完整扫描属于答案知情的事后 oracle 诊断；当前模型和 Router 选择冻结，不用这轮结果重新训练。这轮匹配批次评分不能替代先前的独立盲测报告。箭头是注入向量的投影，不是生成轨迹；显示放大倍率在页面标明。

完整原始输出与验证：network_pilot/runs/router-direction-atlas-v1。冻结 Router 和真实题目状态：runs/state-geometry-router-v1。
