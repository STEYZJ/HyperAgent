# HyperAgent 功能设计说明

整理日期：2026-05-23

## 一句话定位

HyperAgent 要实现的是一个面向高光谱图像分类科研的 Agent 编排层。它不是单纯的聊天机器人，也不是只会跑训练脚本的工具箱，而是把论文学习、数据审计、光谱诊断、实验设计、训练评估、多 Agent 评审、论文叙事和工程治理连接成一条可复盘、可验证、可持续迭代的科研工作流。

核心目标可以概括为：

```text
学习论文经验 -> 转化科研策略 -> 生成实验计划 -> 执行可复现实验 -> 形成证据链 -> 进入下一轮假设
```

## 设计动机

高光谱图像分类研究里，很多工作的问题不在于“没有模型”，而在于研究过程难以持续积累：论文读完后只记住方法名，实验结果很难解释为什么这么设计，baseline 和 split 不够稳定，失败实验没有沉淀，最终论文叙事和实验选择之间缺少可审计证据链。

HyperAgent 想解决的是这条链路：

- 从论文中提取可迁移的研究经验，而不是只复述“论文提出了什么方法”。
- 从数据和光谱事实出发，自动形成下一步实验假设。
- 用固定 split、多 seed、强 baseline 和统一协议保障实验公平性。
- 用事件日志、运行轨迹、cache/cost 统计和 replay 保证 Agent 行为可以复盘。
- 用 Git、工作日志、环境快照和隐私规则保证项目能长期维护。

## 目标用户

主要用户是正在做高光谱图像分类研究的人，包括：

- 需要快速学习论文并提取研究策略的研究生或科研人员。
- 需要组织多数据集、多 baseline、多消融实验的实验负责人。
- 需要把实验结果整理成论文叙事、对比分析和可复现实证材料的作者。
- 需要一个本地可控、可审计、可扩展 Agent 运行时的工程使用者。

## 功能总览

HyperAgent 的功能可以分成七组。

| 功能组 | 要解决的问题 | 主要产物 |
|---|---|---|
| 论文经验学习 | 论文读完后经验不可迁移 | 策略 lesson、实验设计模板、baseline 选择规则 |
| HSI 科研闭环 | 实验决策缺少证据链 | 数据审计、光谱诊断、模型推荐、实验计划、报告 |
| 可复现实证套件 | 结果不可比、不可复现 | benchmark protocol、固定 split fingerprint、多 seed baseline 矩阵 |
| Reasonix 风格可靠运行时 | Agent 工具调用成本高、失败难复盘 | cache-first loop、tool-call repair、usage ledger、replay/stats |
| Claude 风格交互体验 | 本地工具执行不透明、操作摩擦大 | TUI/REPL、命令面板、权限记忆、hook、任务完成事件 |
| Hermes 风格平台视角 | provider/channel/session/skill 状态分散 | platform-status、session-search、skill-usage、channel retry |
| 工程治理 | 项目难维护、依赖和隐私不可控 | worklog、Git 阶段提交、conda 环境、README、environment.txt、About 元数据 |

## 论文经验学习模块

这一模块的重点是“学习研究经验”，不是做论文摘要。每篇论文都要被拆成可迁移经验，例如作者如何包装贡献、如何选择 baseline、如何设计消融、如何讲故事、如何防守审稿质疑。

推荐命令形态：

```bash
HyperAgent research extract --paper <path-or-id> --json
HyperAgent research pattern --paper <path-or-id> --json
HyperAgent research experiment --paper <path-or-id> --json
HyperAgent research storytelling --paper <path-or-id> --json
HyperAgent research taste --field <field> --papers <paper-a,paper-b> --json
HyperAgent research consolidate --topic "baseline selection" --json
HyperAgent research-mcp-serve
```

每条策略经验必须包含：

- claim：这篇论文展示了什么可迁移做法。
- why it works：为什么这种做法有效。
- evidence span：证据来自论文中的哪一段、图、表或实验设置。
- transferable template：以后可以复用的模板。
- risk/limit：这种做法的风险和边界。
- confidence：对该经验可靠性的置信度。

这个模块的价值在于把“读论文”变成可复用的研究资产。例如，不只记录“论文提出 X 模型”，而是记录“作者如何把 X 包装成解决某个瓶颈的必要模块，如何安排消融证明它不是堆模块”。

## HSI 科研闭环模块

科研闭环是 HyperAgent 的主线。理想流程是：

```text
dataset audit
-> spectral diagnosis
-> model recommendation
-> experiment plan
-> baseline training
-> result analysis
-> council review
-> next hypothesis / paper narrative
```

核心功能包括：

- 数据集审计：检查数据形状、类别分布、标注稀疏性、训练/测试划分、潜在泄漏风险。
- 光谱诊断：分析波段冗余、噪声、空间纹理需求、类别混淆可能来源。
- 模型推荐：根据数据特征和研究目标推荐传统 baseline、深度模型或生成模块。
- 实验计划：把目标、数据集、split、seed、baseline、指标和预算写成结构化配置。
- 自动实验：执行 baseline、保存结果、生成报告，并把失败原因也记录下来。
- 结果评审：用多角色 Council 从指标、公平性、消融、复现性和论文叙事角度审查结果。
- 下一轮假设：根据证据提出下一步实验，而不是凭感觉调参。

这一模块的设计原则是：每个建议都必须能追溯到数据、文献、实验结果或预算约束。

## 可复现实证套件

为了让论文式结论站得住，HyperAgent 需要把实验协议先固定下来，再执行训练。

关键能力包括：

- 多数据集协议：例如 Indian Pines、PaviaU、WHU-Hi 等数据集统一纳入协议。
- 多 seed：同一模型在多个 seed 下报告均值和标准差。
- 固定 split fingerprint：保存 split 的 hash、样本数和配置，不上传完整私有 index。
- 强 baseline 矩阵：至少包含 SVM、MLP、Random Forest、KNN，以及后续生成模型。
- 指标一致：OA、AA、Kappa、per-class accuracy 等指标统一输出。
- 训练预算公平：限制 epoch、参数量、数据增强、搜索次数，避免不公平比较。
- 报告生成：按 dataset x baseline x seed 输出 planned/completed、mean/std 和失败原因。

这个模块的目标不是“自动刷分”，而是把实证比较变成可审计对象。

## Reasonix 风格可靠运行时

Reasonix 给 HyperAgent 的启发是：Agent 的可靠性和成本不是靠提示词解决，而是靠运行时不变量解决。

HyperAgent 需要吸收的能力包括：

- cache-first prompt layout：稳定 system/cache guidance 前缀尽量不变，把任务、工具结果和最新用户请求放到 volatile suffix。
- stable prefix hash：每次 action run 记录稳定前缀 hash，用于发现 prefix churn。
- usage ledger：记录 prompt tokens、completion tokens、cache hit/miss tokens、cache-hit ratio 和预算消耗。
- tool-call repair：支持 native tool_calls、JSON list/actions/tool_calls、reasoning-content scavenge、dict 参数归一化和重复调用 storm breaker。
- replay/stats：每次响应、repair、tool step、final、paused、max-step 都可回放和统计。
- token budget：单次 run 有累计预算，超过预算时暂停而不是继续烧成本。

长期目标是让 Agent 每次做了什么、为什么修复、花了多少 token、cache 是否命中，都可以被后来的人审查。

## Claude 风格交互体验

Claude Code 的启发在于低摩擦开发体验和清晰权限边界。HyperAgent 需要提供一套本地可控的交互界面。

关键能力包括：

- REPL：适合普通终端使用，支持 slash commands、手动工具调用、权限确认。
- TUI：展示 provider/model、session、permission、context、tokens、cache、wait status。
- command palette：快速发现内置 slash、Markdown command、skill shortcut 和 plugin bundle。
- remembered permissions：对精确 tool/risk/args 记住授权，支持 list/show/forget/clear。
- hooks：支持任务完成、工具执行前后等事件，为后续自动测试、提醒、审查留接口。
- plan mode：只做规划不执行工具，适合设计阶段和高风险操作前讨论。
- thinking 显示控制：对 DeepSeek/Reasonix 类模型的 reasoning 内容明确标识并可折叠。

这个模块要解决的是信任问题：用户要知道 Agent 现在在哪个会话、准备用哪个模型、拥有什么权限、上下文用了多少、是否在消耗 cache。

## Hermes 风格平台视角

Hermes Agent 的启发在于平台化组织能力。HyperAgent 也需要把 provider、channel、session、skill、runtime event 放在同一个可观测面板里。

关键能力包括：

- platform-status：聚合 provider 配置、channel 环境、session 数量、skill/bundle 数量、runtime event 和 usage。
- live health：显式触发 provider/channel reachability 检查，不默认发送密钥或发起外部请求。
- channel gateway：Feishu/QQ 等外部渠道只做聊天/查询入口，不暴露 shell、训练、写文件等本地工具。
- channel retry：失败消息只重发已经生成的回复，不重新跑 LLM，避免重复消耗。
- session-search：本地索引会话，只返回短 snippet，不暴露完整对话。
- skill-usage telemetry：记录 skill 使用频率、缺失 metadata、未使用 skill，辅助维护 skill 生态。
- plugin/skill bundle metadata：只读扫描 bundle 描述，不执行其中 hook/MCP/command。

这个模块的边界非常重要：外部平台可以进入对话，但不能绕过本地权限体系获得工具执行能力。

## 工程治理与仓库维护

HyperAgent 需要从一开始按研究工程项目维护，而不是临时脚本堆积。

必须遵守的规则包括：

- 每一次工作都写入 `logs/worklog/YYYY-MM-DD.md`。
- 日志要说明上一步是什么、本步做了什么、为什么这么干、效果如何、下一步是什么。
- 按阶段和任务推进，每个阶段形成明确交付物。
- 代码、文档和配置用 Git 管理，阶段性 commit，稳定后 push。
- 使用项目 conda 环境承载依赖，当前环境为 `HyperAgent`。
- 维护 `README.md`、`README.zh-CN.md`、`environment.yml` 和 `environment.txt`。
- 新增依赖后更新 `environment.txt`，没有依赖变化时明确记录“不需要更新”。
- `.env`、API key、runtime state、实验大文件、受限论文 PDF、私有 reviewer notes 不上传 GitHub。
- 仓库 About 信息在 `configs/repository.yaml` 中维护，包括 description、topics、homepage 建议。

这组规则看似“杂事”，但它决定了项目能不能长期演进、复现和公开。

## 阶段路线图

| 阶段 | 名称 | 重点任务 | 完成标志 |
|---|---|---|---|
| Phase 0 | 项目治理基线 | worklog、Git flow、conda 环境、README、隐私规则 | 新任务都有日志，依赖和隐私边界清楚 |
| Phase 1 | 论文经验学习 | research extract/pattern/experiment/storytelling/taste/consolidate | 输出满足 claim/why/evidence/template/risk/confidence |
| Phase 2 | HSI 基础闭环 | audit、spectral diagnosis、recommendation、plan、baseline、report | synthetic demo 和至少一个真实数据集可跑通 |
| Phase 3 | Reasonix 可靠性 | cache-first、repair、usage、replay、stats、budget | action run 可复盘，cache/cost 可解释 |
| Phase 4 | Claude 体验 | TUI/REPL、权限记忆、命令面板、hooks、plan mode | 日常交互低摩擦且权限透明 |
| Phase 5 | Hermes 平台化 | platform-status、channel gateway、session-search、skill-usage、retry | 平台状态可观测，外部渠道安全接入 |
| Phase 6 | 科研实证协议 | benchmark protocol、fixed split、多 seed、强 baseline、消融 | 多数据集结果能支撑论文式比较 |
| Phase 7 | 论文与发布 | 系统图、对比分析、实验报告、文档导航、About 完善 | 仓库能被他人理解、运行、复查 |

## 验收标准

一个功能只有同时满足以下条件，才算真正完成：

- 有明确用户场景和输入输出。
- 有结构化产物，不只是在对话里说过。
- 有日志说明上一步、本步、原因、效果、下一步。
- 有必要测试或校验。
- 没有引入未记录依赖；如有依赖变化，更新 `environment.txt`。
- 没有提交隐私信息、密钥、运行态状态或大数据文件。
- README 或相关文档能让后续使用者找到入口。
- Git 提交只包含当前任务范围，避免混入无关改动。

## 当前状态摘要

目前 HyperAgent 已经具备这些基础：

- 已有 HSI 研究闭环的项目结构和 CLI 入口。
- 已有 Reasonix 方向的 cache-first action loop、tool repair、usage/replay/stats 基础。
- 已有 Claude 方向的权限、TUI/REPL、命令和 hook 增强。
- 已有 Hermes 方向的平台状态、session search、skill usage、channel retry 等能力。
- 已有 benchmark protocol、fixed split 和传统 baseline 矩阵的第一轮实现。
- 已有对比图、Reasonix 命中率研究文档、环境维护文档和 Git 工作流文档。

下一步最自然的是把这些能力统一成更清晰的产品路线：先补论文经验学习命令的稳定输出，再把 HSI 实验协议跑成多数据集、多 seed、强 baseline 的实证结果，最后整理成论文/README/演示材料。
