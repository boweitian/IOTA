# IOTA：从概念地图到任务调度

检索与实验日期：2026-10-11 至 2026-10-12。这里只采用原论文、作者仓库和官方模型页面。

## 先明确这次要交付什么

核心产品是可操作的概念地图。用户看到一个问题在地图中的位置，看到调度器选择哪条高维方向及强度，再看到真实输出与评测结果。地图不是为 benchmark 临时画出的装饰：用于调度的特征就是题目 activation 对概念方向的投影；界面坐标仅是同一空间的有损 PCA 视图。

创作是主展示：不同作品可能需要探索、意象、共情、表达勇气，也可能需要简洁或严格遵循。推理题提供另一种可明确判分的任务，以及干预是否损害普通能力的检查。它们不会代替创作质量。

## 相关方法，以及本次学到的东西

| 原始来源 | 方法要点 | MVP 的采用 / 取舍 |
|---|---|---|
| [Activation Engineering / ActAdd](https://arxiv.org/abs/2308.10248) | 用 activation 差得到推理时控制方向。 | 使用 residual 干预；冻结模型权重。 |
| [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) | 对照回答 activation 的均值差提供概念方向。 | 独立的人工作答对照提取方向；作为静态基线，不假定所有输入都有效。 |
| [Function Vectors](https://arxiv.org/abs/2310.15213) | 某些功能可用内部向量触发，作者使用因果分析定位。 | 高层功能方向值得探索，但本次均值差不是该论文的头部定位算法，不能继承其泛化结论。 |
| [SADI](https://arxiv.org/abs/2410.12299), [作者代码](https://github.com/weixuan-wang123/SADI) | 对照差识别元素，按输入 activation 动态缩放。 | 输入语义应影响干预；本次选择方向和强度，并非复现 SADI 的元素掩码。 |
| [Adaptive Activation Steering / ACT](https://arxiv.org/abs/2406.00034) | 按不同幻觉类别调整方向与强度。 | 保留强度选择和 no-op；这套算法不假定一条“更好”方向适合全部任务。 |
| [Steering Vector Fields](https://arxiv.org/html/2602.01654v1) | 固定向量可能与局部有效方向不一致；概念评分函数的局部梯度定义场。 | 第一版先实现可检查的局部 KNN 效用策略；后续可升级非线性向量场。本次不是该论文方法的完整复现。 |
| [Anthropic：Evaluating feature steering](https://www.anthropic.com/research/evaluating-feature-steering) | 特征激活语境不一定等于干预效果，需要量化目标效果和能力损失。 | 同时报告变好、变差、长度与截断；不凭方向名称宣称效果。 |
| [SteerScope，2026-10-06](https://arxiv.org/abs/2610.07722) | 多维比较目标效果、副作用与泛化；报告提示词基线仍很强。 | 包含提示词增强、随机方向、无几何调度；强度不能只挑漂亮例子。 |

这些工作支持“建立候选方向库并验证其作用”的路线；它们没有证明这九条方向是全部概念、全局不变量，或适用于所有模型的统一坐标。

## 算法 brainstorm 与本次选择

1. **开发集校准的局部效用调度。** 测量每个开发题在不同方向 / 强度下的输出；用概念投影空间中的邻居估计干预的净收益，选择一个方向、一个强度或 no-op。第一版已实现，成本低，选择轨迹可解释。
2. **上下文 bandit。** 不必扫描全部动作；以探索预算采样方向，学习上下文奖励。适合下一版 Walker 主动探索，需同时记录探索分配、奖励方差与停止规则。本次是完整动作扫描，不称作已实现 bandit。
3. **非线性概念向量场。** 学习不同语境下的局部边界，用梯度定义方向；更贴合非线性 geometry，但需要更大且独立的概念数据与稳定性验证。
4. **生成过程中的反馈控制。** 每隔若干 token 更新内部状态，改变方向混合 / 强度，约束总扰动范数。可处理作品写到一半开始重复等问题，但需要不读取标准答案的中间质量信号。
5. **另一个 LLM 做调度 Agent。** 读任务、地图和证据，选择动作。方便交互解释，但会引入额外模型能力、成本和提示词影响。第一版用轻量算法，便于归因；创作 judge 只提供开发奖励与测试评分，不参与测试时路由。

本次模型选择不是全局最佳性能保证。它是一个有限动作空间中的局部经验策略，开发奖励有噪声，创作 judge 尤其如此。

## 评测来源与选择

| Benchmark | 官方来源 | 本次使用 |
|---|---|---|
| EQ-Bench Creative Writing v3 | [作者仓库](https://github.com/EQ-bench/creative-writing-bench) | 32 个公开 brief，12 开发 / 20 测试；改成 180–220 词短篇、贪心生成、本地 4B judge。不是官方 1000 词、采样多轮、Sonnet judge 和 Elo 协议。 |
| WritingBench | [作者仓库](https://github.com/X-PLUG/WritingBench), [论文](https://arxiv.org/abs/2503.05244) | 学习逐题 rubric 与多写作领域的评测设计；本次不冒充运行了其完整数据或作者的 critic。 |
| GSM8K | [OpenAI 原始数据与代码](https://github.com/openai/grade-school-math) | 官方 train 抽 16 开发、test 抽 32 测试，数字最终答案判分。 |
| BIG-Bench Hard | [作者仓库](https://github.com/suzgunmirac/BIG-Bench-Hard) | 布尔表达式、五对象逻辑、多步算术各 16 开发 / 32 测试，分割不重叠。 |
| IFEval | [Google 原始实现](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | 参考可验证约束的思路；本次只报告自己的短篇词数检查，不称作官方 IFEval。 |
| lm-evaluation-harness | [官方仓库](https://github.com/EleutherAI/lm-evaluation-harness) | 学习样本、提示和判分协议的记录方式；本次需要 generation hook 和创作奖励，使用自己的小型可复核 runner，不称 harness 官方成绩。 |

创作评分分成 faithfulness、coherence、originality、craft 四维，保留理由。开发只优化这些维度的平均值；测试另做 A/B 与 B/A 顺序交换偏好检查。单个同族 4B judge 会有文风和同族偏好，仍需跨 judge 与人类盲评验证。界面提供人工盲评入口，未输入的人工评分不会伪装为实测结果。

## 模型选择

[官方模型列表](https://huggingface.co/Qwen/models) 已有较新的 Qwen3.8 大模型；这不意味着存在同系列的 0.8B 通用生成模型。MVP 选择 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 这个近期小型开放权重模型，便于完整方向扫描；judge 使用已经缓存的 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。不把 Qwen3.5 说成整个 Qwen 家族的最新版本。

权重 revision、源数据 commit、文件 hash、chat template hash、层位、dtype、完整对照文本、方向向量、动作扫描、冻结 policy 和原输出均写入实验目录。

## 与 Map · Walker · Soul 的关系

Map 保存有证据的概念方向、关系和语境范围；Walker 扫描方向作用并按当前任务选择局部动作；Soul 暂时是这个模型在测试范围中的行为响应画像。后续可以接入 Monad Theory 的身份一致性检验，但本次没有用球面投影证明全局语义身份，也没有把稳定 probe cosine 解释为意识或灵魂。
