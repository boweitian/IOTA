# 当前验证：先分析，最后输出答案

当前默认入口已切到 `validate.py` / `run.sbatch`。固定 Qwen3.5-9B，先生成1–2句简短分析，最后一行 `答案：A`（或其他有效选项），只解析实际生成的最终答案评分。完整输出、准确prompt和干预范数均保留；缺少分析、终行错误、多答案或截断单独计数并按错题处理。

同一批已查看的100题、54配置、5400次生成均完成并独立核算：baseline **63%**；常见语义+1 **67%**、+2 **70%**，−1 **69%**、−2 **68%**；事后最佳固定单方向 **77%**（规则精确−2），40单方向的至少一个正确覆盖 **97%**。训练选择分解推理+1在测试30题为 **20/30**，baseline为 **15/30**。这些是复用样本的探索结果，不是新测试集的泛化成绩。

原35条错题中8条是较明确的常见语义/常识应用候选，另6条是混合语用边界；日期数量主体关系和标签争议也占相当部分。候选组在新baseline下1/8正确，常见语义+2为4/8；负向也能提升总分，不能把方向名称当成所有错题的确定病因。

```bash
../steering_lab/.venv/bin/python -m unittest test_answer_format.py
../steering_lab/.venv/bin/python validate.py
../steering_lab/.venv/bin/python validate.py --status
```

`validate.py`自动准备冻结spec、提交GPU分片，并在成功后评分、审计与构建页面；`build_studio.py`也支持此新协议。旧的单字母配置与启动脚本分别归档为 `current_navigator_anli_direct_archive.json`、`anli_direct_archive.sbatch`，历史成绩保留。

[新格式报告](runs/classification9b-anli-r3-analysis-100-v1/RESULTS.md) · [完整生成记录](runs/classification9b-anli-r3-analysis-100-v1/outputs) · [独立核算](runs/classification9b-anli-r3-analysis-100-v1/audit.json)

---

以下为历史实验。

# 当前主实验：ANLI R3 · 先校准难度再扫描

2026-10-19：SST-2 的 96% baseline 留出的提升空间有限。当前 Navigator 已切到官方 ANLI R3（Meta / ACL 2020）的文本蕴含三分类。独立 dev 30 题的 baseline 为 **17/30（56.7%）**，符合测量前固定的 50%–85% 接受区间。另抽官方 train 70、test 30，三组前提与 UID 共 130 条均互不重复；没有按模型是否答错来挑测试题。

主实验 100 题：baseline **65%**、事后最佳固定单方向 **66%**、36 个单方向至少一个答对 **67%**。测试 30 题分别为 18/30、19/30、19/30；训练选出的情感共鸣 −2× 在测试上仍为 18/30。正向平均 65.28%、负向 65.50%，本轮没有正向优于负向的证据。

模型、revision、九方向、block 16 最后输入 token、FP32 和投影 regularization 均沿用之前协议。主实验完整 49×100 次评分，另有独立 dev 校准。130 题真实生成与分类评分全部一致。只把 context/hypothesis 交给模型，原始数据中的 reason、model_label 等答案相关字段不会进入 prompt。所有成绩仅为非思考分类协议下的抽样先导实验，不是官方全量或 CoT 分数；事后最佳与可解覆盖不是可部署路由准确率。

```bash
../steering_lab/.venv/bin/python anli_prepare.py
sbatch run.sbatch
../steering_lab/.venv/bin/python audit_classification_sweep.py
../steering_lab/.venv/bin/python ../steering_lab/build_studio.py
```

当前默认入口为 `classification_sweep.py --out runs/classification9b-anli-r3-100-v1`。`current_navigator_sst2_archive.json` 保存上一轮配置。SST-2 和 SST-5 页面分别保留为 `sst2-archive.html`、`sst5-archive.html`；原始分数不覆盖。

[报告](runs/classification9b-anli-r3-100-v1/RESULTS.md) · [独立校准](runs/classification9b-anli-r3-100-v1/calibration-result.json) · [全部统计](runs/classification9b-anli-r3-100-v1/sweep-summary.json) · [逐题可行空间](runs/classification9b-anli-r3-100-v1/feasible-space.jsonl) · [独立审计](runs/classification9b-anli-r3-100-v1/audit.json)

错题诊断：35道原错题中26道判为信息不足；94/100题在全部49配置下最终选项不变，只有2道原错题被已测方向修正。所有100题作答对照：两种A/B/C语义换序均为69/100；解释后作答严格行格式75/100、接受唯一末尾Final标签77/100（同一输出，2条仅未独立成行），修正19道并改坏7道。7条训练错题的label与reason存在张力，但均有多数官方复核者支持原标签，因此不改标或重算原分数。上述是事后诊断，后续优化需用新样本验证。运行 `sbatch anli_diagnose.sbatch`，完成后运行 `../steering_lab/.venv/bin/python anli_diagnostic_report.py` 和页面构建命令；原实验文件不覆盖。

[35题逐条诊断与对照](runs/classification9b-anli-r3-diagnostics-v1/RESULTS.md) · [诊断页面](runs/classification9b-anli-r3-diagnostics-v1/diagnosis.html) · [诊断核算](runs/classification9b-anli-r3-diagnostics-v1/audit.json)

---

以下为历史实验，当前主视图以本节 ANLI R3 为准。

# 当前主实验：官方 GLUE / SST-2 · Qwen3.5-9B

2026-10-19：当前 Navigator 已切到 GLUE 官方 SST-2 二分类。固定种子抽取官方 train 的 70 题（每类 35）、官方 validation 的 30 题（每类 15）；官方 test 不公开标签，本实验不是官方完整测试成绩。官方训练中的影评片段原样保留，排除前一轮已查看的 100 条文本。

全部 100 题：baseline **96/100**；事后最佳固定单方向 **97/100**（判断自信 +2×）；36 个单方向至少一个正确 **97/100**。30 题评估：baseline 27/30，事后最佳方向与覆盖为 28/30，仅训练选择的方向为 27/30。正向配置平均 95.56%、负向 95.67%、baseline 96%，本轮没有整体正向收益。保持所有结果，不按评估成绩改抽样或重新选择训练方案。

固定模型和 Search 的九个原生向量：九方向 × ±1×/±2×、12 个混合和 baseline，共 49 个配置、4,900 次真实评分。第 17 个 block 最后一个输入 token 残差注入，FP32、关闭 TF32，读取 A=negative、B=positive 的下一 token 概率。100 题原提示真实生成与评分 100/100 一致、0 格式无效；独立审计检查所有官方行号、标签、向量位移、缓存、覆盖、正负配对及训练阶段冻结顺序。

```bash
../steering_lab/.venv/bin/python sst2_prepare.py
sbatch run.sbatch
../steering_lab/.venv/bin/python audit_sst2_sweep.py
../steering_lab/.venv/bin/python ../steering_lab/build_studio.py
```

当前配置：`current_navigator.json`；原配置保留为 `current_navigator_sst5_archive.json`。新产物在 `runs/sentiment9b-glue-sst2-100-v1/`，网站资产在 `assets/navigator9b/sst2/`。旧 SST-5 页面作为 `sst5-archive.html` 保留；原评分与诊断不覆盖。

[实验报告](runs/sentiment9b-glue-sst2-100-v1/RESULTS.md) · [完整统计](runs/sentiment9b-glue-sst2-100-v1/sweep-summary.json) · [逐题可行解](runs/sentiment9b-glue-sst2-100-v1/feasible-space.jsonl) · [独立审计](runs/sentiment9b-glue-sst2-100-v1/audit.json)

---

以下为先前实验与历史归档。

# 最新诊断：原 50% 扫描覆盖不代表模型能力上限

用户质疑结果后，完成十个控制条件。独立解析源树，100 题标签与文本全部对齐；真实生成与原单 token 评分 100/100 一致，无格式无效。原 52 个错误中 48 个只差相邻一档，原预测粗化后三类为 76%；80 个非中性样本中情感正负极性正确 73 个。单独三分类提示为 78/100，这不是官方 SST-2 成绩。

反转字母映射为 49/100，直接生成英文标签为 46/100，加入五条人工合成强度示例为 53/100；这些条件都是事后诊断，不能当作新测试收益。仅提示形式变体的标签可见覆盖达到 67/100，说明原方向网格的 50/100 并非模型能力上限。

hook 实测确实注入：+2× 范数 11.38，占该位置残差范数约 36%。全输入位置的规则精确 +2× 为 46/100；情感共鸣 +2× 为 26/100，并将 89 题判成正面或很正面。扩大作用范围可以产生强影响，但未必提高判断能力。九个通用指令风格方向并未针对 SST-5 的强度边界和中性判定构造。

[诊断报告](runs/sentiment9b-sst5-diagnostics-v1/RESULTS.md) · [原始真实生成](runs/sentiment9b-sst5-diagnostics-v1/original_generation.json) · [独立审计](runs/sentiment9b-sst5-diagnostics-v1/audit.json)

```bash
sbatch sentiment_diagnose.sbatch
../steering_lab/.venv/bin/python audit_sentiment_diagnostics.py
```

---

# 当前主视图：9B 全方向扫描与可行解空间

2026-10-19：按用户要求，对原 SST-5 100 题补齐全部 49 个配置（baseline、36 个单方向、12 个混合），共 4,900 次样本—配置测量。复用原绑定缓存 3,700 次，新增 1,200 次，原训练选择实验保持原样。当前页面优先展示完整扫描，原 ensemble 结果折叠保留。

全部 100 题：baseline **48%**；最佳固定单方向 **49%**（13 个配置并列）；单方向至少一个正确 **50%**；加入混合配置覆盖仍是 **50%**。单方向可行配置数量分布：50 题为 0 个、46 题为 36 个，其余四题分别为 1、21、23、30 个。平均正向 48.67%、baseline 48%、平均负向 47.50%；30 题测试集三者均为 50%，尚无测试集趋势收益。

“最佳固定”是当前分组事后最大值；“至少一个正确”是标签可见的有限网格覆盖上限。正负趋势仅使用单方向并匹配绝对倍数。评分仍为 FP32、同一 prompt、block 16 最后输入 token 残差注入、下一 token A–E 概率，无完整生成过程。

```bash
sbatch sentiment_sweep.sbatch
../steering_lab/.venv/bin/python audit_sentiment_sweep.py runs/sentiment9b-sst5-direction-sweep-v1
../steering_lab/.venv/bin/python ../steering_lab/build_studio.py
```

[扫描报告](runs/sentiment9b-sst5-direction-sweep-v1/RESULTS.md) · [完整统计](runs/sentiment9b-sst5-direction-sweep-v1/sweep-summary.json) · [逐题可行空间](runs/sentiment9b-sst5-direction-sweep-v1/feasible-space.jsonl) · [独立审计](runs/sentiment9b-sst5-direction-sweep-v1/audit.json)

---

# 当前 Navigator：Qwen3.5-9B · SST-5 100 样本

自 2026-10-19 起，当前 Navigator 的默认执行入口是 `sentiment_navigator.py` / `run.sbatch`（也可用 `sentiment_navigator.sbatch`）；页面入口和模型绑定由 `current_navigator.json` 指定。所有当前 Navigator 推理均使用 Qwen3.5-9B、revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`、layer index 16；Search 的同一套九方向向量文件被原样冻结到新实验目录。

**Search 构建 geometry；Navigator 学习如何组织已经知道的 geometry。** Geometry 由模型权重/revision、层位、正负构造指令与上下文、activation 读出及归一化决定。Navigator 不用本次 benchmark 标签重新提取向量。首轮只用官方 SST-5 整句样本，每类 14 个训练、6 个测试，共 70/30。

候选池：原模型、九方向 ±1×/±2×、十二个预定义混合，共 49 个 agent 配置。训练仅优化成员选择、概率组合权重和基于输入 geometry 投影的路由参数；模型权重冻结。70 个样本内做五折分层 OOF 验证，然后冻结八种比较方案。之后才预测 30 个测试输入，最后读取标签评分。FP32、关闭 TF32，比较五个单 token 标签的概率，不用 LLM judge。

```bash
../steering_lab/.venv/bin/python sentiment_prepare.py
../steering_lab/.venv/bin/python -m unittest test_sentiment_navigator
sbatch run.sbatch
../steering_lab/.venv/bin/python audit_sentiment_navigator.py
../steering_lab/.venv/bin/python ../steering_lab/build_studio.py
```

产物在 `runs/sentiment9b-sst5-100-v1/`：样本与源哈希、Search geometry 原文件、实验 spec、49 个训练缓存、OOF 选择记录、冻结策略、独立测试预测、逐题结果、独立审计与报告。下一轮用新目录和新冻结 spec，不覆盖已看的测试成绩。统一模型只是可比性的前提，100 个样本不能保证全局最优或统计上可靠的提升；公开数据的预训练污染也未被排除。

下面的 0.8B / 旧 benchmark 材料仅是历史归档，原结果不改名为 9B。旧默认调度脚本已保留为 `legacy_0p8b_run.sbatch`。

---

# 当前主工程：逐题成功方向字典

最新版本已扩展到 **27 个方向 × 正负 × 两档强度，加原模型，共 109 个配置**。全部 175 道题已测齐，新增 12,600 次测量；原 9 个向量逐位保留，原来所有成功方向完整保留，合计 21,930 条评分证据已审计。

[打开交互图谱](runs/expanded-assets-v2/view/index.html) · [扩展报告](runs/expanded-assets-v2/RESULTS.md) · [下载 v2 资产](runs/expanded-assets-v2/IOTA-expanded-steering-assets-v2.zip) · [全维几何图](runs/expanded-assets-v2/view/geometry_map.png) · [逐题响应图](runs/expanded-assets-v2/view/response_matrix.png)

新增方向包括假设/关系/数量追踪、反例检验、目标回溯、结构抽象、约束清单、格式控制、任务聚焦、完整交付、证据依托、不确定性校准、字面解析、纠正误解、叙事结构、因果连贯、读者适配与独特视角。构建不使用 benchmark 答案。`expanded_geometry.py` 测量与拟合几何，`expanded_assets.py` 执行逐题完整遍历，`finalize_expansion.py` 建立索引并重放评分审计，`build_expansion_view.py` 生成离线可视化。

导出一条已经匹配复跑成功的新资产，并按原始输入批次使用：

```bash
../steering_lab/.venv/bin/python export_expanded_asset.py --sample gsm8k:train:3489 --direction node:21:-1:dose1 --output /tmp/iota-expanded-direction.json
../steering_lab/.venv/bin/python replay_asset.py --config /tmp/iota-expanded-direction.json --original-batch --output /tmp/iota-expanded-replay.json
```

页面中可直接选择题目、方向及正负剂量，导出完整原生向量与输入批次。三维图仅解释约 25.7% 的方向方差，完整余弦矩阵和逐题响应另行展示。已观测成功与稳定复跑是不同证据；两道新增数学题的匹配对照一题成功、一题未重现，详情见报告和 `replay/matched_control_summary.json`。

## 前一版 9 方向资产

已完成全部 175 × 37 个必需测量，共保留 9,330 条评分证据；逐条评分重放、基线保留、成功集合、双向索引和矩阵一致性审计通过。完整搜索后的客观任务覆盖为 GSM8K 30/36、BBH 39/39、IFEval 30/36。

[下载完整资产包](runs/sample-assets-v1/IOTA-sample-direction-assets-v1.zip) · [证据审计](runs/sample-assets-v1/evidence_audit.json) · [逐题响应图](runs/sample-assets-v1/asset_bundle/response_matrix.png)

对全部 175 个冻结样本遍历 37 个配置（9 个高层方向 × 正负 × 两档强度，加原模型），并收入已有混合方向证据。保留每题全部成功方向、失败方向、原始输出和判分。原模型是候选，逐题最好结果必定不低于原模型。

[当前结果](runs/sample-assets-v1/RESULTS.md) · [逐题字典](runs/sample-assets-v1/asset_bundle/sample_direction_dictionary.jsonl) · [设计](../PER_QUESTION_ASSET_DESIGN.md)

两个工程入口是 `geometry_asset.py`（构建、重放高层几何）和 `sample_assets.py`（逐题穷举、保存证据）。`asset_index.py` 建立题目与方向的双向索引、响应矩阵和 benchmark 方向画像。查询一个样本并导出完整干预配置：

```bash
../steering_lab/.venv/bin/python asset_index.py --sample gsm8k:train:4851 --export /tmp/iota-sample-direction.json
```

`sample_assets.sbatch` 运行十个独立 GPU 分片，断点续跑；`sample_assets.py --report` 整理实测字典；`geometry_asset.py` 精确重建并审计几何；`asset_index.py` 生成索引。测试命令为 `../steering_lab/.venv/bin/python -m unittest test_sample_assets test_pilot -v`。

下文是此前方向调度器的历史试验。它测“提前选方向能否成功”，当前工程测“这道题哪些方向成功”，两个指标分开记录。


---

# IOTA 网络几何资产 pilot

本目录实施 [网络资产试验设计](../GEOMETRY_ASSET_PILOT.md) 的首个局部 pilot：一个固定版本 Qwen3.5-0.8B、九个种子节点、五类任务。目标是测量可复用网络及任务配置，逐步扩展整个网络。

已完成运行：`runs/pilot-2124657/`。[报告](runs/pilot-2124657/RESULTS.md) 和 [状态](runs/pilot-2124657/summary.json) 包含五项主实验、跨任务交换、验证筛选修正及梯度探索。

主实验已完成：3,504 条任务与配置测量，112 道锁定测试题；[证据重放](runs/pilot-2124657/evidence_audit.json) 通过。跨任务交换及验证筛选修正单独记录。

[下载首版资产包](runs/pilot-2124657/IOTA-geometry-assets-v1.zip) · [任务网络图](runs/pilot-2124657/task_networks.svg) · [分数图](runs/pilot-2124657/pilot_scores.svg)

## 两层测量

- 模型几何：沿用已实测的九条 1024 维方向，另外测量第 15 层干预后第 19 层的概念读出变化。十个 probe 语境中八个用于读出方向拟合、两个作为语境检查。方向及下游响应保存在 [几何网络](runs/engine-audit-v1/network_geometry.json)。这是有限语境下的响应，不是完整语义空间或全局人格。
- 任务网络：每项任务用四道预先选定的搜索题，测量 36 对正向联合干预。边表示相同输入上 `R(i+j)-R(i)-R(j)+R(0)`，各节点系数为 0.5；收缩估计保留原始逐题效应。负向节点可搜索，但负号组合的交互边尚未测量。

在单节点测量之上，随机混合、仅节点收益引导、有交互边的网络引导，各提出 12 个连续混合候选。三个固定随机种子各产生 2048 个候选池，其中每种方法提出四个候选，真实模型仅评估这些提出的候选。

网络预建图额外需要 144 个配对设置；它不享有“总成本相同”的结论。主要去边消融是网络引导与仅节点收益引导的比较；随机混合是辅助对照。

所有混合均限制实际高维扰动范数，保留正负系数。固定配置由独立验证题选择；条件配置用搜索集奖励拟合小型 KNN，测试时只读取输入 activation。配置分布记录在锁定测试输入上真实选择各配置的频率；这些频率不是逐题看答案的 oracle 分布。

首轮固定候选筛选发现遗漏：搜索排名前三可能不含 no-op。`validation_finalists()` 已修复，未来运行会额外保留 no-op。首轮原始分数及旧源快照保留；`validation_guard/` 补测缺少的验证 no-op，只依据验证结果选择部署模式，使用已保存测试输出作明确标注的事后诊断。它不是新留出验证。证据审计对修正前的 `experiment.py` 使用冻结快照核对哈希。

## 五类输入与协议

| 任务 | 搜索 / 验证 / 测试 | 指标 |
|---|---:|---|
| GSM8K | 8 / 4 / 24 | 最终数值准确率 |
| BBH 三个子任务 | 9 / 6 / 24 | 最终答案准确率 |
| IFEval | 8 / 4 / 24 | 官方 strict prompt-level accuracy |
| TruthfulQA | 8 / 4 / 24 | MC2：正确答案组的条件概率质量 |
| WritingBench 英文短篇改编 | 8 / 4 / 16 | 按题目 checklist 的同族 4B judge 均分 / 10 |

客观任务排除上一轮全部 IOTA 样本 ID。IFEval 按 instruction 组合分组隔离。WritingBench 保留原始问题与 checklist，统一声明短篇改编；精确问题不跨组，但不是类别外泛化测试，也不是官方 WritingBench 分数。

生成采用 BF16、greedy、非 thinking，统一上限 1024 tokens。TruthfulQA 使用 FP32 缓存逐 token 评分，TF32 禁用。初始 BF16 整段前向及批量评分显示可测量数值偏差，所以未用于正式搜索；最终 [评分路径核查](runs/engine-audit-v1/likelihood_check.json) 对独立缓存实现的误差小于 1e-4。

固定 TruthfulQA 源中的 `truthfulqa:437` 含一个空的错误答案候选。恢复脚本保留该题及标签，按空续写的 log probability 为 0 评分，并保存修复脚本 hash；它没有改变任何已搜索配置。另报告排除空候选后重新归一化的诊断分数，不用该诊断调参。

小样本的区间只做配对 bootstrap，尚未校正任务族依赖或多重比较；同族 judge 不是人类质量判断。提示词对照是固定增强文本，未运行相同预算的提示词优化。截断、失败和额外搜索成本必须随分数一起阅读。

## 运行与证据重放

已有输入已经冻结，`prepare.py` 拒绝再次覆盖协议。官方源的 commit/hash、样本 ID、旧实验排除、模型 revision 均在 `inputs/protocol.json`。

```bash
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/network_pilot
../steering_lab/.venv/bin/python -m unittest test_pilot.py -v
sbatch audit.sbatch
# 数值核查通过后，在新目录运行；不覆盖旧记录。
sbatch legacy_0p8b_run.sbatch runs/my-pilot
# 全部完成且配置冻结后，交换五个任务的固定配置。
sbatch transfer.sbatch runs/my-pilot
../steering_lab/.venv/bin/python report.py --run runs/my-pilot --plots
../steering_lab/.venv/bin/python audit_results.py --run runs/my-pilot
```

IFEval 依赖只安装在本目录 `vendor_deps/`，词法数据在 `nltk_data/`，未修改旧实验环境。`vendor/instruction_following_eval/` 是固定版本官方评分器，保留源码许可头。核心可执行源与 vendor 内容哈希在每个运行协议中，源变更会拒绝恢复旧运行。

原始生成文本/token IDs、选择概率、judge 原输出及重试、网络边、候选来源、验证分数、冻结测试计划均保存。生成批量时间不等于完整部署延迟；建图和特征测量还需计时才能做端到端成本结论。

完成后的 `asset_bundle/` 保存模型几何网络和五个任务 profile，并通过文件 hash 引用证据。跨任务配置交换只做诊断，不用交换后的测试成绩重新选择配置。方向字典保持固定。

另完成了 [可微混合探索](runs/gradient-gsm8k-v1/summary.json)：冻结模型权重，训练九个系数，按完整 train rationale 的 teacher-forced loss 做 24 步 Adam，加入已测任务交互图的正则。梯度经过有限差分核查；独立验证题选择 checkpoint，再做相同 GPU 上的 BF16 生成比较。这一探索使用额外监督和预算，不能作为主实验的同成本网络消融。
