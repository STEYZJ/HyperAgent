# HyperAgent 与参考项目对比图

本目录比较 HyperAgent、Hermes Agent、Claude Code 参考仓库、DeepSeek Reasonix 在结构拓扑、业务流程、用例覆盖和模块通信机制上的差异。

| 图 | SVG | PNG |
|---|---|---|
| 结构拓扑对比图 | `01-topology-comparison.svg` | `01-topology-comparison.png` |
| 业务流程对比图 | `02-workflow-comparison.svg` | `02-workflow-comparison.png` |
| 用例覆盖对比图 | `03-use-case-comparison.svg` | `03-use-case-comparison.png` |
| 模块通信机制对比图 | `04-module-communication-comparison.svg` | `04-module-communication-comparison.png` |

详细文字分析见 `comparison-analysis.md`。

## 如何阅读

- `01-topology-comparison.*`：回答“我的项目结构和三个参考项目的定位有什么不同”。
- `02-workflow-comparison.*`：回答“业务闭环、任务阶段、反馈机制分别强在哪里”。
- `03-use-case-comparison.*`：回答“我的能力覆盖有哪些改进，哪些通用 agent 能力还不足”。
- `04-module-communication-comparison.*`：回答“模块间怎么通信，哪些机制值得从 Hermes/Claude/Reasonix 吸收”。

## 主要结论

HyperAgent 的创新点集中在 HSI 科研闭环、证据驱动实验、光谱知识接入、多 Agent Council 和 paper-to-module-to-ablation 链路。它的不足主要在通用开发者体验、插件生态、多平台产品化、事件回放、工具修复和缓存成本控制。

本目录只包含设计分析和生成图，不包含 `.env`、API key、私有 reviewer notes、受限论文 PDF 或运行态数据。
