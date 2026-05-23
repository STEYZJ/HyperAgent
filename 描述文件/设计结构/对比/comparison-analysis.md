# HyperAgent 与三个参考项目对比分析

## 总体结论

HyperAgent 和 Hermes Agent、Claude Code、DeepSeek Reasonix 的核心差异不在“是否能聊天或调用工具”，而在目标函数不同：

- **HyperAgent**：面向高光谱图像分类科研，重点是数据审计、光谱知识、模型/参数/模块设计、自动实验、评估报告和证据链。
- **Hermes Agent**：通用自改进 agent 产品，强在多平台 gateway、skills/memory、工具生态、cron、子代理和部署形态。
- **Claude Code 参考仓库**：公开部分主要体现插件/命令/agent/hook 的交互范式，强在操作逻辑和开发工作流模板。
- **DeepSeek Reasonix**：DeepSeek-native coding agent，强在 cache-first loop、工具调用修复、成本控制、事件日志和回放。

## 对比范围

本次对比覆盖 4 类已有设计图，并为每一类都生成横向对比图：

| 类型 | HyperAgent 原图 | 三个参考项目原图 | 对比图 |
|---|---|---|---|
| 结构拓扑 | `../01-project-topology.svg` | `../{DeepSeek-Reasonix,claude-code,hermes-agent}/01-project-topology.svg` | `01-topology-comparison.svg` |
| 业务流程 | `../02-business-workflow.svg` | `../{DeepSeek-Reasonix,claude-code,hermes-agent}/02-business-workflow.svg` | `02-workflow-comparison.svg` |
| 用例覆盖 | `../03-use-case.svg` | `../{DeepSeek-Reasonix,claude-code,hermes-agent}/03-use-case.svg` | `03-use-case-comparison.svg` |
| 模块通信 | `../04-module-communication.svg` | `../{DeepSeek-Reasonix,claude-code,hermes-agent}/04-module-communication.svg` | `04-module-communication-comparison.svg` |

## 四类图的直接结论

| 图类型 | 看出的主要不同 | 对 HyperAgent 的判断 |
|---|---|---|
| 结构拓扑 | HyperAgent 把 `schemas/core/tools/data/models/training/evaluation/knowledge` 作为科研闭环模块；Hermes 更像平台产品；Claude Code 参考更像插件/命令/agent/hook 范式；Reasonix 更像围绕 DeepSeek API 的可靠执行层。 | HyperAgent 的结构更垂直、更领域化，科研证据链是主轴；通用平台完整度仍弱于 Hermes。 |
| 业务流程 | HyperAgent 是数据审计、光谱诊断、模型推荐、实验计划、训练、评估、Council、下一轮实验；Hermes 是多端入口到 skills/memory/tools；Claude 是开发任务到插件/命令工作流；Reasonix 是 cache-stable request、repair、tool、event/cost/replay。 | HyperAgent 的改进在于把“下一轮实验为何这样做”结构化；不足是事件回放、成本缓存和命令体验还不够系统。 |
| 用例覆盖 | HyperAgent 在 HSI 数据审计、实验计划、光谱知识、文献到模块、消融评估上最强；Hermes 在多平台、skills、memory、产品化部署上更强；Claude 在开发操作模板上更成熟；Reasonix 在低成本可靠调用和 replay 上更强。 | HyperAgent 的独特用例是科研实验自动化，不是通用 coding agent 替代品；后续应补齐开发者体验和运行时可靠性。 |
| 模块通信 | HyperAgent 以 schema 和工具层连接科研流程；Hermes 强调 gateway/runtime/skills/memory 的平台通信；Claude 强调 command/plugin/hook 的声明式操作协议；Reasonix 强调 append-only event、repair、cache/cost 统计。 | HyperAgent 已有科研闭环通信雏形，但需要把 slash、tool、skill、agent、event 纳入同一运行协议。 |

## HyperAgent 的改进点

1. **领域闭环更明确**：不是通用 coding agent，而是围绕 HSI classification 的科研循环构建，天然有论文创新叙事。
2. **实验依据更强**：把数据划分、光谱诊断、模型推荐、消融配置、Council 评审都结构化保存，适合写论文和复现实验。
3. **模块解耦更贴近科研扩展**：`schemas/core/tools/data/models/training/evaluation/knowledge` 的边界清楚，新模型、新数据集、新评估器可以独立扩展。
4. **多 Agent 评审有领域角色**：Result Analyst、Hypothesis、Skeptic、Reproducibility、Budget 与 HSI 实验闭环结合，比单纯“跑训练脚本”更有价值。
5. **文献与模块生成连接科研实验**：literature → module proposal → materialize → ablation plan 这条线，是通用 agent 项目里没有直接覆盖的。

## HyperAgent 的不足

1. **产品成熟度不足**：TUI、命令帮助、权限面板、工具输出结构、交互细节还在快速修补阶段，距离 Hermes/Claude 的稳定体验还有距离。
2. **工具调用可靠性仍需加强**：需要吸收 Reasonix 的 tool-call repair、storm breaker、并行安全调度和事件 replay。
3. **插件/skill 生态还小**：已经支持第三方安装，但缺 marketplace、版本管理、审计、更新和 namespace 规范。
4. **多端 gateway 还浅**：Feishu/QQ 已接入，但平台覆盖、消息重试、平台状态、权限隔离还不如 Hermes。
5. **缓存/成本控制不够系统**：已有 usage/Reasonix profile，但还没有 DeepSeek cache-first 的严格 prefix/log/scratch 不变量。
6. **通用 coding workflow 还弱**：feature-dev、code-review、commit/push/PR 的完整体验还不如 Claude Code 插件范式。

## HyperAgent 的创新性

1. **Knowledge-Augmented HSI Research Agent**：把光谱规则、数据集知识、模型选择经验和实验证据接进 agent 决策。
2. **Autonomous Experiment Council**：用多角色 agent 对实验结果进行结构化评审，避免单一 agent 盲目调参或钻牛角尖。
3. **Evidence-Grounded Experiment Cycle**：下一轮实验不是随机 AutoML，而是由指标、光谱诊断、文献证据和预算约束共同驱动。
4. **Paper-to-Module-to-Ablation Path**：把文献思想转成可 materialize 的模块 factory，再自动生成消融配置，形成科研生产链。
5. **Domain-First Agent Framework**：不是做一个通用 Claude Code 克隆，而是把通用 agent 技术落到 HSI classification 的痛点上。

## 可迁移策略经验

### 经验 1：用领域闭环定义 agent 的价值边界

- **claim**：HyperAgent 的最大创新不是“也能调用工具”，而是把 HSI 科研闭环定义成 agent 的主任务。
- **why it works**：领域闭环让每一步都有可审计输入输出，实验建议可以回到数据、光谱、指标、文献和预算证据，而不是停留在聊天建议。
- **evidence span**：`01-topology-comparison.svg` 的结构结论；`02-workflow-comparison.svg` 的“数据审计→模型推荐→实验→评估→Council→下一轮”；HyperAgent 原始 `02-business-workflow.svg`。
- **transferable template**：先固定领域对象和证据对象，再设计 agent loop：`domain audit -> diagnosis -> candidate action -> execution -> evaluation -> critique -> next action`。
- **risk/limit**：领域闭环会牺牲通用性；如果底层训练、数据加载或评估协议不稳定，agent 决策会显得很强但实验不可复现。
- **confidence**：高。

### 经验 2：把参考项目能力拆成可吸收的运行时机制

- **claim**：HyperAgent 不需要复制 Hermes、Claude Code 或 Reasonix，而应吸收它们的关键机制：平台入口、声明式操作协议、事件回放与缓存成本控制。
- **why it works**：这样可以避免产品目标漂移，同时把通用 agent 项目的成熟工程能力接入 HSI 研究流程。
- **evidence span**：`04-module-communication-comparison.svg` 的 Hermes/Claude/Reasonix 到 HyperAgent 启发区；`03-use-case-comparison.svg` 的生态、平台、回放、成本控制差距。
- **transferable template**：为每个参考项目提炼一句机制，而不是一整套克隆目标：`Hermes -> gateway/skill hub`，`Claude -> command/agent/hook protocol`，`Reasonix -> event/cache/repair ledger`。
- **risk/limit**：吸收机制时容易把框架复杂度抬高；每加入一个通用能力，都要问它是否服务科研闭环。
- **confidence**：高。

### 经验 3：创新叙事应落在证据链，而不是单个模块名

- **claim**：HyperAgent 的论文/项目叙事应强调 evidence-grounded experiment cycle，而不是强调某个模型或某个 agent 名称。
- **why it works**：审稿或工程评审更关心实验选择是否公平、可复现、可解释；证据链能把数据、模型、文献和评估连接成可防守的贡献。
- **evidence span**：`comparison-analysis.md` 中改进点 1-5；`02-workflow-comparison.svg` 的流程结论；`03-use-case-comparison.svg` 的 HSI audit/plan/run/eval/council 强项。
- **transferable template**：把创新写成 `evidence source -> decision rule -> generated artifact -> validation result -> next hypothesis`。
- **risk/limit**：如果没有多数据集、多 seed、固定 split 和强 baseline，对证据链的声称会被认为只是自动化包装。
- **confidence**：中高。

## 后续建议

优先级最高的不是继续堆模型，而是补齐三条基础能力：

1. **Reasonix 方向**：事件日志、replay/diff/stats、tool repair、cache/cost 统计。
2. **Claude 方向**：声明式 commands/agents/hooks/skills、统一 slash registry、TUI palette 和权限面板。
3. **Hermes 方向**：多端 gateway、subagent tree、skill hub、memory/search 和平台化部署。

当前实施进展：Reasonix 方向已进入第一轮工程落地，重点补强了 action-loop 累计 token budget、usage event 追踪、cache hit/miss 元数据、repair 事件、cache-first 稳定前缀 hash，以及 `replay/stats` 的可观测性输出。Claude 方向也完成第一轮体验补强：本地 remembered permissions、`/permissions list|forget|clear`、TUI status/context/cache/permission 面板，以及 action-loop `TaskComplete` hook 已落地；下一步可继续做 command palette、权限详情面板和更完整的插件 bundle。

## 对比图文件

| 图 | SVG | PNG |
|---|---|---|
| 结构拓扑对比图 | `01-topology-comparison.svg` | `01-topology-comparison.png` |
| 业务流程对比图 | `02-workflow-comparison.svg` | `02-workflow-comparison.png` |
| 用例覆盖对比图 | `03-use-case-comparison.svg` | `03-use-case-comparison.png` |
| 模块通信机制对比图 | `04-module-communication-comparison.svg` | `04-module-communication-comparison.png` |
