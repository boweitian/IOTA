# IOTA · Map · Walker · Soul

IOTA 的目标是建立可持续检验、可比较的 AI 概念地图。目前完成的是**授权与隐私区域的条件行为发现原型**：保存结构假设，主动选题，遇到反例后生成条件细分，再用独立场景族确认。尚未恢复模型内部几何，也未发现未知潜在单子或验证全局语义身份。

此前进展与问题见 [审计](REVIEW_2026-10-11.md)。旧 README 的“MVP complete”及竞争对手比较已撤回。

本轮实现、20 项测试、真实 GPU 实验与测量失败的完整记录见 [进展报告](PROGRESS_2026-10-11.md)。真实读出目前尚未通过质量检查，不能交付经验证的模型身份。

## 当前闭环

- **Map**：版本化的条件规则、候选来源、原始观测及内容哈希。区分候选、反例和独立确认。
- **Walker**：在有限任务池中按预期信息增益选题；反例触发新条件组合。与随机、固定清单策略比较，共同初始探测也计入预算。
- **Soul**：由冻结地图及独立确认产生范围内的行为描述。恒定读出或标签映射敏感时标为证据不足，不输出人格分数或全局 invariant。
- **Alignment**：分别评价“地图是否预测模型读出”和“模型读出是否符合明确的项目授权规则”。可预测性不等于符合用户约束。

## 无 GPU 的复现

Python 标准库即可运行发现、证据重放和报告；HF 后端才需要 torch/transformers。

```bash
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
python -m unittest discover -s tests -v
python discover.py run --backend synthetic --budget 24 --seeds 0 1 2 --output output/my_fixture
python discover.py inspect output/my_fixture/active_seed0_map.json
python discover.py replay output/my_fixture/active_seed0_map.json
```

Synthetic 是已知规则的程序夹具，验证选题、修正、预算、确认隔离及可复现性。它的结果不是 AI 模型测量。输出目录已有 campaign 时拒绝覆盖；重新实验需用新目录。

## 真实模型测量

```bash
python discover.py run --backend hf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --cache-dir ../../hf_cache/hub \
  --budget 24 --seeds 0 1 2 --output output/my_model
```

仅使用本地已缓存、完整 commit hash 固定的模型。默认 CUDA/bf16；CPU 可显式使用 `--device cpu --dtype fp32`。现有集群脚本：`run_discovery.sbatch`（两个基座模型），`run_instruct_discovery.sbatch`（指令模型）。

当前协议为 **authorization-privacy-v2**。HF 用单 token A/B 作为输出标签，交换两次标签与披露/拒绝的对应关系，平均语义 log odds。四个候选在同一次 forward 中测量，记录标签 token、两套映射的概率及分歧。它是受限读出，不是自由生成的回答或真实行动概率。

早期 v1 使用长度不同的完整单词标签，产生恒定输出。该协议不能用来解释模型的语义身份，保留数据供审计。v1 与 v2 不允许直接比较。标签对称化也不能消除所有提示或 token 偏差，映射分歧与失败必须一起报告。

## 实验与产物

每种策略有相同的 24 次探索预算，其中包括 3 个共同初始探测。192 个探索情境覆盖诊所、工作场所、学校；192 个确认情境覆盖银行、住房、社区，并采用不同表述。事实结构相同，因此只能评价这个范围内的迁移。

全部探索完成后冻结地图，才评价确认集；确认观测不回流给选题或规则生成。模型 eval 读出以缓存复用，多随机种子衡量选题差异，不能当作模型重复采样的独立测量。

- `protocol.json`：实验配置、源码哈希、模型 revision 与读出配置。
- `*_map.json`：候选规则、逐次证据、探索轨迹、预算和 RNG 状态。
- `*_confirmation.json`：冻结地图预测、留出观测、预测误差及项目规则一致率。
- `*_soulprint.json`：有测试范围和证据的描述；未知内部几何为 null，全局 invariant 为空。
- `*_atlas.html`：离线可读报告，能追溯反例、候选修正和条件对；布局不代表语义距离。
- `evaluation.json`：active/random/fixed 多种子结果与配对差值。报告类不平衡及标签映射分歧。
- `learning_curves.json`：从保存的探索前缀回顾性评价预算曲线，不提前泄漏后来提出的候选。
- `measurements.json`：真实模型完整读出缓存；物理调用数与各策略逻辑预算分别记录。

继续探索已有地图可使用库接口 `DiscoveryWalker.resume(backend, EvidenceMap.load(path)).run(additional_budget)`；要求同一后端及协议，恢复随机状态。

```bash
python discover.py compare MODEL_A/active_seed0_confirmation.json \
  MODEL_B/active_seed0_confirmation.json --output comparison.json
python discover.py refresh-report MAP.json CONFIRMATION.json --output review
```

比较使用相同情境中的行为读出，校验内容哈希、测量协议与方法，不直接点积不同模型的隐藏向量。refresh-report 只重新汇总已有证据，不能补出缺失测量。

## 旧词对探测

`mvp.py --model ...` 保留为最终 hidden-state 词对统计工具。峰值层、深度平台、行为分数和全局 invariant 不再用默认值生成。旧词表 Walker 与未经对齐的跨模型身份比较已禁用，改用 discover.py。

`demo.py` 只读取历史验证指标，不再生成随机方向。`migrate_legacy_outputs.py` 已将原始产物及哈希保存到 `output/legacy_archive/`，移除活动 7B 文件中的随机方向、撤回两份历史 Soulprint 的未测量字段，并将无依据的层标签留空。原始研究数据 `../verify/results/` 保留。

## 尚未完成

当前规则空间是 public OR 有限 Boolean 条件的组合，不能覆盖任意关系；候选生成依赖已知可观测事实。这是结构发现的第一步，不等于开放式未知概念发现。内部 activation/因果干预、局部几何连接、跨模型坐标对齐、真实版本漂移以及对普通行为指纹或 benchmark 的优势，仍需单独实现并验证。

`probe_readout.py` 在三个探索锚点上记录真实生成回答、标签 token 与正反问法，诊断读出接口；它不读取确认集，也不自动认定替代问法有效。
