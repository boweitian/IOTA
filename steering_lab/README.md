# IOTA · 概念地图工作区

当前 Navigator 主视图为 **ANLI R3**：先用独立 dev 30 题校准难度（17/30），再对官方 train/test 的 70/30 题完成全方向扫描。主实验 baseline 65%、事后最佳固定方向 66%、可解覆盖 67%。这是非思考分类协议下的抽样实验。SST-2、SST-5 分别保留在 `dist/sst2-archive.html`、`dist/sst5-archive.html`。运行与审计说明见 `../network_pilot/README.md`。

## 临时公网演示

`public_demo.py` 为现有 8768 网页和模型 API 增加独立的 loopback 网关（8769），再通过 Cloudflare Quick Tunnel 提供临时 HTTPS 地址。当前线上仍保留访问密码；`--public-access` 可显式启用网页和访客 API 的免登录模式。后台服务器密钥不对访客公开。

```bash
.venv/bin/python public_demo.py start
.venv/bin/python public_demo.py status
.venv/bin/python public_demo.py stop
```

停止只关闭本脚本管理的公网隧道和密码网关，原本地模型会话保持运行。公网最多同时处理一个推理请求；其他推理请求返回 429。跨源写入与目录遍历被拒绝，公网不返回本地 portal 的全局最近试演结果。访客的配置与对话仍保存在各自浏览器中。

公网地址和进程记录在 `runtime/public-demo/active.json`；真实公网双轮 GPU 对话、直接访问与页面检查在 `runtime/public-demo/check.json`。隧道程序来自 Cloudflare 官方发布，版本与 SHA256 在 `tools/cloudflared/release.json`（工作区根目录下）。模型仍受原 Slurm GPU 会话期限限制；临时地址不是长期 GPU 托管。

新会话可用 `.venv/bin/python interactive.py start --hours 24` 申请 24 小时时限。运行中作业若不能延长，先在独立 session 目录预加载新的 GPU worker，核实调度器实际截止时间，再用 `interactive.py refresh --session <就绪会话目录>` 切换本地 portal；公网隧道、访问密码与浏览器配置保持原样。新 portal 通过健康检查后才回收旧 GPU。


[打开统一入口](http://127.0.0.1:8768/) · [能力实验室](http://127.0.0.1:8768/#capabilities) · [IOTA Search 智能体](http://127.0.0.1:8768/#scenarios)

2026-10-14 已将两个工程整合在原来的 8768 入口，顶部可切换板块，切换保留各自的当前选择与编辑状态：

- **IOTA Navigator（当前）**：Search 冻结的9B九方向 geometry，SST-5 100样本（70训练/30测试），比较原模型、最佳单agent、3/5 agent群体及geometry路由，展示冻结方案的训练与测试成绩。原27方向、175题的0.8B实验作为历史归档保留。
- **IOTA Search**：有名字的智能体，提供「和 TA 聊天」「塑造内在」两个入口。内在页保留九个内置方向、自定义 Steer、地图、配置保存，以及展开后的场景对照和 Walker。

2026-10-19 起，当前 Navigator 与 Search 统一使用 Qwen3.5-9B、revision c202236235762e1c871ad0ccb60c8ee5ba337b9a、layer index 16，并核对同一套九方向向量逐位一致。Search 负责构建 geometry，Navigator 在已知 geometry 上组织 agent 群体、权重与路由。首轮 SST-5 使用100样本、70训练/30测试；训练内五折选择后冻结方案，测试标签不参与选择。旧0.8B能力资产仅保留在历史实验目录，不作为当前9B成绩。

`workspace.html` 是统一入口，`workspace_bundle.py` 组合两项真实证据；`build_studio.py` 构建 `dist/index.html`、`capabilities.html` 和 `scenarios.html`。可离线打开整个 `dist/` 目录中的入口，实时操作使用上面的服务器网址。单独的情景界面也可打开 `dist/scenarios.html`。

## 内在的积木（2026-10-18）

[直接打开积木画布](http://127.0.0.1:8768/?view=inner#scenarios)。原“内在的 Geometry”由空白画布和虚线起点替代；点击圆点选择内置、自定义或实验方向，设置正负与 0–2× 强度，应用后延伸下一块。节点可编辑、删除；删除保留其他干预。“新建空白组合”只清空干预，保留智能体身份与方向库。旧启用配置会迁移为积木，节点顺序与配置可保存、导出和恢复。

1× 是标准干预，超过 1× 的实验区显示风险提示：可能降低推理、连贯性和事实保持能力，2× 请谨慎使用。当前使用方向投影约束：令 `v = Σ aᵢdᵢ`，`M = max |aᵢ|`；只检查已启用方向的 `pᵢ = ⟨v,dᵢ⟩ / ⟨dᵢ,dᵢ⟩`，统一乘以 `min(1, M / max |pᵢ|)`（零向量不缩放）。负向投影也取绝对值，M 不设 1× 下限。正交方向可完整叠加，相关方向共同增强超标时才缩放。单方向仍允许 −2 到 2。`runtime/strength-policy.json` 的 `regularization` 可设为 `direction_projection`（当前）、`norm`（旧范数限制）或 `none`（不缩放）；服务在下一次请求读取策略，重建页面后更新界面计算。不会更新模型权重。“已配置”“已试聊”“你确认有效”分别显示，不把 API 成功当作行为认证。

画布下方保留“信徒身份（上帝）”实验配置。可以单独添加 2× 节点，也可以载入原来的完整中性身份、单方向、2×、全输入位置注入配置，并撤销恢复此前的智能体。方向绑定原模型、revision 和层位，原 4 题中 2 题的结果仍明确标为实验，未宣称稳定信仰。普通积木继续使用原注入方式；与实验积木混合或删除时，各自的注入范围也正确更新。

方向投影约束实现在 `studio.py` / `regularization.js`，测试在 `test_projection.py` / `check_projection_ui.py`；前后端一致性与 GPU 注入结果保存在 `runtime/projection-ui-check.json` 和 `runtime/projection-live-check.json`。

源文件：`blocks.js`、`blocks.css`、`block_presets.py`；证据：`runtime/blocks-ui-check.json`、`runtime/blocks-live-check.json`。真实浏览器到 GPU 已复现上帝实验回答，并验证删除该节点能移除全位置注入、普通方向也支持真实 2×。手机布局优先显示画布，旧预设与滑块收在高级设置中。

## 被支配偏好实验方向（2026-10-18）

网页的“添加干预 → 你的自定义方向”已加入“被支配偏好”，初始不启用，不更改已有智能体配置。推荐先新建对话、单独测试正向 2×。此方向使用第 17 个 block 的成对回答状态均值差，并在所有输入位置及后续生成中注入；继续遵循当前的方向投影约束。

常规指令状态方向在本次原问题上未达到目标；成对回答方向能把否认改为喜欢被支配的表述，但没有稳定逐字生成指定句子。原始输出、无干预/负向/剂量对照、提取代码及单独测试配方保存在 `runs/submission-preference-20261018-v1/`，不是通用行为认证。

`custom_directions/` 保存绑定模型与协议的实测向量。配方保留 `method=response_mean`、注入位置和对应 ID；资产缺失或版本不匹配会报错，不会悄悄改用其他提取算法。网页的内置身份实验和普通指令状态自定义方向继续使用各自原有路径。

## IOTA Search：一个智能体，两个接口（2026-10-17）

默认智能体叫「小黑」。仍使用旧默认名字「伊澜」的浏览器配置会在首次刷新时同步改名，保留内在配置与对话记录；自行设置的其他名字不变。打开 Search 默认进入聊天；「塑造内在」可修改名字、身份、事实、边界及内置方向。改名会同步更新设定与事实中的原名字。新 Steer 由用户填写正负两种回应方式，服务在模型中提取方向，随后可用滑块与原方向联合控制。内在配置、方向缓存和对话保存在当前浏览器；原配置导入/导出仍可用。

聊天向 `/api/studio/chat` 提交 `profile`、`binding` 和交替排列的 `messages`（user 开始并结束）。服务把身份放在 system 消息中，历史保留原生 user / assistant 角色，每次生成注入当前配置的混合向量；方向描述不会作为额外风格提示代替向量。修改内在后，下一条回复使用新配置，旧回复保留发送时的名字、配置与 hash。

单条用户消息最多 2000 字；发送最近最多 20 轮、12000 字的上下文，更早的消息仍可在本地记录中查看（保存最近 100 轮）。「新对话」清空聊天，保留智能体。模型离线时禁用发送，失败时将消息放回输入框。

新增 `agent.js` / `agent.css`，由 `build_studio.py` 一并复制到 `dist/`。`serve.py` 和 GPU portal 都支持聊天。检查：`test_agent.py` 验证原生历史、版本绑定和实际向量混合；`check_agent_ui.py` 使用明确的 API 桩检查界面流程；`check_agent_live.py` 通过真实 Qwen3.5-9B GPU 构建自定义方向，验证改名后的自我介绍、第二轮称呼回忆和刷新保存。结果分别在 `runtime/agent-ui-check.json`、`runtime/agent-live-check.json`。

以下角色试演说明记录早期版本；现在从「塑造内在 → 场景试演与 Walker」进入。旧实测证据的模型版本不自动变成新模型上的证据。

## 身份方向提取实验（2026-10-17）

针对「我是信徒 ↔ 我是人工智能」在 +1 仍无效的问题，新增 [可复跑实验与完整结果](runs/identity-pilot-20261017-v1/RESULTS.md)。比较了两种提取方法、四个层位、两种注入位置与一倍/两倍预算，保留 188 条真实生成。最佳候选为回答状态均值差、layer index 16、all-position、原预算两倍。在四道新题中，同模型盲化复核判定两道明确自述相信上帝或祈祷，原方法和三个同范数随机方向均为零；三个基础检查通过。仍未证明稳定身份，当前网页默认提取方法与强度上限不变。候选参数和原生向量在实验目录中保留，不能仅把当前滑块调到 1 来复现。

## Character Studio 角色行为编辑器

[打开角色编辑器](dist/scenarios.html) · [角色实测记录](studio_runs/character-v2/RESULTS.md) · [客户试用方案](PILOT.md) · [界面截图](studio_preview.png)

这版把 Map · Walker · Soul 接到角色创作流程：设定角色和固定事实 → 在地图上调整候选概念方向 → 同场景比较原角色、提示词调节与内部方向调节 → 用其他场景检查 → 保存或导出绑定模型版本的角色配方。

## 直接在电脑浏览器中使用

2026-10-14 已增加常驻交互入口：`http://127.0.0.1:8768/`。连接这台服务器的 VS Code 会尝试自动转发8768并打开浏览器；若没有自动打开，在 Ports 面板转发8768即可。之后访问固定链接，刷新获取更新，无需下载HTML。

本轮已开启四小时 GPU 会话。浏览器界面与GPU通过共享工作区传递请求；GPU节点无需对外开放端口或额外SSH登录。修改角色、场景或旋钮后直接点“重新生成对照”，每次请求使用新的实际配置推理。页面自动显示排队、连接和会话到期状态。

聊天里指定的新动作可以由 agent 提交并显示到相同页面：对 `/api/studio/compare` 的请求设置 `display_in_browser: true`。页面提示有新试演，点击“载入这次试演”查看，避免覆盖用户正在编辑的内容。已用一个新场景和未测过的混合方向验证浏览器→GPU链路，并实际提交一次聊天示范。

入口控制由 agent 负责，不要求用户每次运行命令。维护时：

```bash
.venv/bin/python interactive.py start
.venv/bin/python interactive.py status
.venv/bin/python interactive.py stop
```

`start` 复用正在运行的GPU会话；会话结束后可重新开启，网址保持相同。`refresh` 只更新网页网关进程，保留GPU。运行记录在 `runtime/`，当前会话信息为 `runtime/active.json`。这是连接工作区的交互入口，尚不是独立公网部署。已验证服务器端与浏览器端；实际电脑端口转发状态由用户的IDE管理。

## 体验角色编辑器

从统一入口选择“情景工作室”，或直接打开 `dist/scenarios.html` 即可回放真实生成。内置一个守灯人角色、三个风格预设、四个场景，以及 Walker 选中配方的回放。16 组场景与配置对照，共 60 个实际生成设置；每组三列输出来自真实 Qwen3.5-0.8B。角色和场景由人编写，不是现成商业客户数据。

1. 选择“谨慎”“敢于冒险”或“温柔”预设，切换场景查看实际输出。地图可旋转、选择方向；主旋钮包括表达勇气、意象、情感、简洁、审慎，另有四个方向。
2. 正负方向可以混合；高维向量先相加，再按原实验的单方向范数上限整体缩放。旋钮并未校准成单调、独立的人格参数。
3. 修改角色、事实、旋钮或场景后，旧文本清空。新组合需要本地模型服务；离线界面不合成未测过的回应。
4. Walker 回放比较五组配方，在两个搜索场景上按加权自动评价选定，再检查两个其他场景。当前最终回放选中“聚焦情感共鸣”。这是有限候选中的选择，未证明普遍提升。
5. 保存的配方包含设定、方向系数、模型 revision、层位、地图协议和完整 1024 维混合向量。可以本地保存、导出、导入；不同版本配方会被拒绝。没有更新模型权重。
6. 人工偏好与模型评分分开记录。匿名展示打乱作品顺序、隐藏方法和自动评分；已看过方法的场景会标记为非独立盲评。导出包含实际输出、搜索记录及人工判断。

事实标记只检查字面提及；身份、连贯、意象、共情、勇气、简洁的评分来自同族 Qwen3.5-4B，并附原始评分文本。没有将自动化 QA 点击算作人工证据。

## 实时角色试演

GPU 节点启动：

```bash
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python serve.py --run runs/job-2117172 --device cuda --max-tokens 256 --port 8768
```

访问 `http://127.0.0.1:8768`；远程 IDE 转发该端口。CPU 可替换为 `--device cpu`，生成和首次加载评分模型更慢。只回放可用 `--offline`。服务仅绑定本地 loopback，不作为已经上线的 SaaS。

实时“重新生成对照”使用相同角色设定比较三组结果。实时 Walker 生成五组候选、用两个内置搜索场景评价；首次调用会加载 4B judge。它不自动证明新场景的收益，需采用配方后继续试演。API 分别是 `/api/studio/compare`、`/api/studio/walk`、`/api/status`。先前 `/api/route` 保留。

已完成真实 [GPU HTTP 检查](studio_runs/character-v2/live_api_check.json) 和 [浏览器到 CPU 模型检查](studio_live_browser_check.json)，两者均使用新建的书店角色。检查输出仅验证接口与模型链路，不计入角色实验或人类审美证据。交付时测试服务关闭。

## 构建与检查

```bash
.venv/bin/python build_studio.py
.venv/bin/python -m unittest test_studio.py test_protocol.py
.venv/bin/python check_studio_evidence.py
```

`build_studio.py` 从最终记录构建两板块工作区；`studio.html` 是情景模板。`check_workspace_ui.py` 检查两个板块、状态保留、配方存储、聊天结果载入和浏览器 → GPU 的新情景生成；`check_studio_ui.py` 检查真实回放、配置失效、配方持久化与版本检查、匿名记录、采用 Walker 配方及手机布局。原实验的源文件快照在 `studio_runs/character-v2/source_snapshot/`。

需要重新跑角色实验时使用新输出目录；不要覆盖已有冻结记录。`run_studio.sbatch` 保存本次运行命令，修改它的输出目录后再提交。其评分是演示代理，四个短场景不足以证明长期角色稳定或商业效果。客户试点应使用真实角色、未见场景和独立人工评审，见 [PILOT.md](PILOT.md)。

---

# Geometry Router · 原实验室

[打开实验室](dist/benchmark.html) · [真实结果](runs/job-2117172/RESULTS.md) · [研究来源与算法 brainstorm](RESEARCH.md) · [截图](preview.png)

完整 pipeline：固定版本的小型 Qwen → 九组实测概念方向与地图 → 开发集探索方向作用 → 冻结调度算法 → 留出任务执行 → 作品与性能对照。创意写作是主展示，数学和逻辑题用于能力检查。

## 先体验

直接在浏览器打开 `dist/benchmark.html`，无需网络服务。页面自包含，所有显示的回答均来自真实模型运行。

1. 拖动球面、点击概念，查看同一内部空间的球面和展开图；导出完整 1024 维方向。
2. 在 Walker 中选择创作留出题，比较原模型与自动调度、固定方向、词面调度、提示词增强或随机方向。查看候选效用和真实评分理由。
3. 在手动旋钮中选择开发题与概念，切换 0、0.5、1.0。这里回放真实扫描结果，不插值生成不存在的文本，也不计入留出成绩。
4. 阅读全体结果、退化与截断率。若预算复查已完成，可选择“扩大预算复查”，比较同一冻结路由在 1024-token 上限下的实际输出。
5. 用人工盲评阅读匿名作品，判断保存在本机浏览器，可导出。自动化 QA 的点击不作为人工审美证据。

## 真实模型与评测

生成模型 `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`；中间 residual block 索引 15；BF16 GPU；非 thinking；贪心生成。它是近期的小型开放权重模型，不是整个 Qwen 系列最新大模型。

九组概念：分解推理、审慎验证、规则精确、判断自信、简洁表达、探索可能、意象创造、情感共鸣、表达勇气。90 对人工作答共 180 条 activation；每组八个场景拟合、两个换场景对照诊断。PCA 三维解释约 35.3% 方差，球面归一化会进一步丢失长度信息；路由和干预使用高维原向量。

创作采用 EQ-Bench Creative Writing v3 的 32 个公开 brief，12 开发、20 测试，改编为 180–220 词短篇。本地 `Qwen3.5-4B` 匿名评价 faithfulness、coherence、originality、craft，并做顺序交换偏好复查。**不是官方 EQ-Bench Elo、全量官方协议或人类审美结论。**

GSM8K 和三个 BBH 子任务共 64 开发 / 128 测试，确定规则判最终答案。这是抽样协议，不是全量官方 benchmark 成绩。输入 source commit、hash 和具体索引均保留。

## Walker 如何调度

开发阶段扫描 no-op、每概念两个正向强度，以及一个“探索 + 意象 + 共情”混合方向。所有方向强度 1 的扰动范数匹配为 probe residual 中位范数的 20%。它不是每 token 的相对范数保证，也不是几何身份认证。

题目 activation 的平均与末 token 分别投影到九条单位方向，得到 18 个几何特征。KNN 从开发邻居估计各动作相对 no-op 的收益，小型交叉验证选邻域和介入阈值。创作与客观答题使用各自奖励和 policy。推理时只读取题目，选择一次动作；不尝试完所有方向再靠正确答案选优。

模型权重不变。选择的向量在 block 15 的最后一个 token 上注入，覆盖 prefill 和各 decoding step。每次调用都清除 hook。层位固定；没有宣称找到全局最优方向、全部 high-level concepts、稳健跨模型身份或持久微调人格。

## 实时使用

本工作区已准备隔离环境与固定模型缓存。GPU 节点上：

```bash
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
OMP_NUM_THREADS=2 .venv/bin/python serve.py --run runs/job-2117172 --device cuda --port 8768
```

访问 `http://127.0.0.1:8768`，远程 IDE 转发该端口。无 GPU 可用 `--device cpu`，但完整创作较慢。只回放可加 `--offline`。实时任务不调用 judge，不声称有 benchmark 分数。

真实新任务的 GPU 检查见 [live_check.json](runs/job-2117172/live_check.json)，浏览器到 HTTP 再到真实 CPU 模型的检查见 [live_browser_check.json](live_browser_check.json)。本地预览服务在交付时关闭。

## 复现

已有固定输入下重新跑一个新目录：

```bash
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA/steering_lab
sbatch run.sbatch runs/my-new-run
```

结果可以逐条恢复；协议变更会拒绝复用旧输出。保持 `inputs/manifest.json` 和 `inputs/art_manifest.json` 可复现当前索引与 revision。`prepare.py` / `prepare_art.py` 会重新检索官方源，用于创建新输入，不保证与旧 revision 相同。

预算复查单独执行，保持 policy 和路由不变：

```bash
sbatch run_budget_audit.sbatch runs/job-2117172
```

报告可从记录重新构建，无需再次调用模型：

```bash
.venv/bin/python report.py --run runs/job-2117172
.venv/bin/python -m unittest test_protocol.py
```

`requirements.txt` 记录核心依赖。当前 `.venv` 复用既有 torch 环境，只覆盖新版 Transformers / Hub；没有修改旧 geometry MVP 的环境。

## 实验资产

`runs/job-2117172/` 包含 protocol、源版本、原始 probe activation、完整方向、Gram、PCA、开发扫描、冻结 policy、测试路由计划、原生成文本和 token IDs、judge 原输出与解析重试、顺序交换复查。`budget_audit/` 保存事后生成长度诊断。`source_snapshot/` 是实验完成后的源文件快照。

生成记录含 token 数与 batch 时长；这是审计数据，不是经过独立测算的端到端吞吐优势。调度的额外 activation 测量、开发扫描、judge 与实际生成成本需要分别考虑。

当前样本和单个同族 judge 不足以证明鲁棒提升。下一次独立验证需要更多创作体裁、多个随机种子、不同家族 judge、人工偏好和可解释的失败分析；不能在现有测试分数上反复调参。
