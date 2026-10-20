<p align="center">
  <img src="Moonlit%20Cat%20by%20the%20Sea.png" alt="IOTA — Moonlit Cat by the Sea" width="100%">
</p>

# IOTA

**让智能拥有一个人能够理解、塑造的内在世界。**

[体验 Demo](https://bowei.netlify.app/iota/) · [English](README.md) · [项目介绍 PPT](IOTA%20general.pptx)

IOTA 探索如何把人的意图转化为模型内部可测试的变化：理解行为，通过概念方向调整内部激活，并在持续交互中研究数字个体的身份与成长。

项目共享一个基础 **Concept Geometry（概念几何）**，发展两条产品路径：

| | IOTA Navigation · Navigator | IOTA Search |
|---|---|---|
| 核心问题 | 对当前输入，怎样组织已有方向才能更好地完成任务？ | 怎样创造、塑造一个持续存在的数字个体？ |
| 工作方式 | 在固定 geometry 中选择方向、强度或不干预 | 连接内在编辑、对话、情景测试和配置保存 |
| 验证重点 | 新输入上的任务收益、成本与副作用 | 行为变化、身份连续性与可恢复的版本 |

## IOTA Navigation

**Navigator 在已经建立的 geometry 中，为每个输入选择有用的 steering。**

基本流程是：任务输入 → 方向选择 → 设置强度并生成 → 评估收益与代价。这个阶段的 geometry 保持固定，**不干预也是有效选择**。

同一个方向可能修复某些题，也可能改坏原本答对的题。因此，Navigator 的目标是依据当前输入组织内部控制，而不是给所有输入套用同一组最强方向。

验证需要比较原模型、优化提示词、固定 steering、随机选择和动态 Navigator，并在未参与选择的新输入上报告任务成功率、额外延迟、计算成本和其他能力的损失。

代码主要位于 [`network_pilot/`](network_pilot/)：候选扫描、路由与组合实验、逐题可行配置、输出解析和独立审计。最新 ANLI R3 探索扫描的摘要见 [实验记录](docs/ANLI_R3.md)。逐题知道标准答案后得到的覆盖率不等于 Navigator 已实现的准确率。

## IOTA Search

**Search 面向一个有名字、可持续交互、内在结构能够被塑造的数字个体。**

两个界面连接同一个 agent：

- **Inner World · 塑造内在**：调整方向和强度，创建并测试自定义 steering，保存、比较和恢复配置。
- **Outer World · 外部交互**：与 agent 对话、测试情景，观察内在编辑如何改变它的实际回应。

发展路径是创建个体 → 交互观察 → 验证变化 → 启用有来源记录、可以恢复的版本。现有代码提供编辑、对话和配置工作流；长期身份一致性、经验积累和可靠的个体成长仍需要进一步验证。

代码主要位于 [`steering_lab/`](steering_lab/)：激活干预引擎、方向构造、角色与场景、agent 界面、配置和模型服务。

## 共同基础：Concept Geometry

谨慎、勇气、共情、好奇等人类概念，被连接到具体模型的方向、层位和干预强度。每个可复用方向都需要记录来源、适用范围、评估和版本，并通过可重复的行为测试证明其作用。

当前主要机制是 **activation steering**：在推理期间向选定模型层的激活加入方向向量，模型权重保持不变。方向名称描述构造意图；实际效果取决于模型、层位、输入和强度，三维展示中的距离也不直接等于原生高维距离。

Navigator 积累输入到方向的效用数据；Search 提供个体上下文、新方向需求和行为反馈。二者共同积累经过评估的 geometry 和版本化个体配置。

## 代码入口

| 路径 | 内容 |
|---|---|
| [`steering_lab/`](steering_lab/) | Search 工作区、实时干预、方向构造和交互界面 |
| [`network_pilot/`](network_pilot/) | Navigator 实验、benchmark 扫描、评分、诊断与审计 |
| [`geometry_mvp/`](geometry_mvp/) | 轻量概念几何浏览器与随附实测样例 |
| [`iota/`](iota/) | 早期地图、探索与证据原型 |
| [`tests/`](tests/) | 早期核心模块测试 |

主演示地址：[https://bowei.netlify.app/iota/](https://bowei.netlify.app/iota/)。实时生成需要连接模型服务；静态托管页面本身不提供 GPU 推理。

轻量离线原型的本地启动、完整工程环境和数据说明见 [English README 的 Getting started](README.md#getting-started)。历史工作区说明保留在 [README_LEGACY.md](README_LEGACY.md)。
