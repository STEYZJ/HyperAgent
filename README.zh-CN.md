# HyperAgent 中文文档

HyperAgent 是一个面向高光谱图像分类研究的解耦式 Agent 框架。当前版本重点打通可复现实验闭环：

```text
数据审计 -> 光谱诊断 -> 模型推荐 -> 实验计划 -> baseline 训练 -> 报告
```

架构图和横向对比图保存在 `描述文件/设计结构/`。其中 `描述文件/设计结构/对比/`
按结构拓扑、业务流程、用例覆盖和模块通信四类图，对比 HyperAgent、Hermes
Agent、Claude Code 参考和 DeepSeek Reasonix。
`docs/reasonix_cache_hit_study.md` 记录了本轮 Reasonix 命中率研究，说明
DeepSeek prefix cache、append-only log、tool-call repair 和 telemetry 是如何共同
支撑高命中率的，以及哪些策略可迁移到 HyperAgent。
`docs/hyperagent_feature_design_summary.md` 汇总了计划设计实现的功能、阶段路线、
验收标准和仓库治理规则。
`docs/hyperagent_production_potential.md` 进一步分析了框架作用、还能扩展成什么，
以及如何落到课题组 ResearchOps、可复现实验、团队机器人和受控 Agent Runtime。

## 快速开始

推荐使用项目 conda 环境：

```bash
conda activate HyperAgent
HyperAgent demo --synthetic
```

环境维护见 `docs/environment_maintenance.md`：`environment.yml` 用于兼容安装，
`environment.txt` 记录当前验证环境的精确包快照。

如果出现 `HyperAgent: command not found`，说明命令还没有安装到当前 shell。可以先使用仓库内启动器：

```bash
./HyperAgent demo --synthetic
```

也可以在仓库根目录执行 editable install，让 `HyperAgent` 变成可直接调用的命令：

```bash
/home/lzj/miniconda3/envs/HyperAgent/bin/python -m pip install -e .
```

## 密钥安全

API key 只能放在环境变量或本地 `.env` 文件中，`.env` 已被 git 忽略。不要把原始 API key 粘贴到 README、worklog、配置示例、prompt 或已提交的会话记录里。HyperAgent 写入 worklog 前会自动脱敏明显的密钥形态，测试套件也会扫描已跟踪文件中的疑似密钥。

## 第三方许可证

已知开源依赖和参考项目记录在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。新增运行时依赖、复制第三方源码或准备发布二进制/Docker 镜像前，需要同步更新该文件并做许可证复查。

## Bot 渠道接入

HyperAgent 可以启动 FastAPI webhook 网关，接入官方飞书 Bot 和 QQ 官方 Bot。第一版渠道只允许聊天/查询：外部消息会进入持久化 AgentLoop，但不会暴露 shell、训练、写文件或通用 agent 工具。微信适配器暂不接入，后续单独扩展。

```bash
HyperAgent channel-init --provider feishu
HyperAgent channel-init --provider qq
HyperAgent channel-list
HyperAgent channel-test --provider feishu --text "规划下一轮 HSI 实验"
HyperAgent channel-run --host 0.0.0.0 --port 8765
```

在本地 `.env` 或环境变量中配置平台凭据，然后在飞书/QQ 官方平台把回调 URL 指向：

```text
POST /webhooks/feishu
POST /webhooks/qq
```

生成的 `.hyperagent/channels.yaml` 只保存环境变量名，不保存真实密钥。

## 受控联网与功能面板

HyperAgent 让模型通过本地受控工具联网，而不是依赖模型厂商不透明的原生浏览器。搜索需要在本地 `.env` 或环境变量中至少配置一个 provider：`BRAVE_SEARCH_API_KEY`、`TAVILY_API_KEY`、`SERPAPI_API_KEY` 或 `SEARXNG_BASE_URL`。用户明确给出公网 HTTP(S) URL 时，`web fetch` 不需要搜索 provider。

```bash
HyperAgent web status
HyperAgent web search --query "latest hyperspectral image classification agent"
HyperAgent web fetch --url https://example.org/paper
HyperAgent /web
```

联网工具会拒绝 `file:`、`data:`、`javascript:`、localhost 和私网 IP。结果保存在 `.hyperagent/web_runs/`，会话中只注入摘要、来源 URL、抓取时间和 citation id。图片入口第一版会生成受权限控制的请求工件，保存在 `.hyperagent/image_runs/`：

```bash
HyperAgent image status
HyperAgent image generate --prompt "高光谱实验工作流示意图"
```

截图中的 Codex/Claude Code 风格功能入口已经统一到 CLI、REPL、TUI 和斜杠命令：

```bash
HyperAgent ide-context status
HyperAgent ide-context set-open-files hyperagent/cli.py hyperagent/runtime/repl.py
HyperAgent plan-mode on "只做设计"
HyperAgent plan-mode off
HyperAgent personality status
HyperAgent feedback add "TUI 面板需要突出联网状态"
HyperAgent worktree
HyperAgent /mcp status
```

计划模式会暂停 action-loop 类工具执行，包括 `run`、`agent-act`、`agent-run`、REPL `/act` 和手动 `/tool`，直到执行 `HyperAgent plan-mode off` 或 `/plan-mode off`。

## 常用命令

```bash
HyperAgent init --dataset-root /data2/lzj/lab/Mamba_test/dataset
HyperAgent status
HyperAgent demo --synthetic
HyperAgent tui
HyperAgent repl --permission ask
HyperAgent agent-run --agent reviewer --instruction "检查最近一次实验报告"
HyperAgent experiment-cycle --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json
HyperAgent web status
HyperAgent web search --query "latest hyperspectral image classification agent"
HyperAgent web fetch --url https://example.org
HyperAgent image status
HyperAgent ide-context status
HyperAgent plan-mode status
HyperAgent feedback list
HyperAgent worktree
HyperAgent channel-run --host 0.0.0.0 --port 8765
```

Claude Code 风格入口也可用：

```bash
HyperAgent "分析最新实验结果并提出下一步实验"
HyperAgent plan "把 module_proposal.json 物化为模型 factory"
HyperAgent act "检查 reports 并选择下一条安全命令"
HyperAgent /help
HyperAgent /tui
HyperAgent /language
HyperAgent /commands
HyperAgent /todos
HyperAgent /doctor
HyperAgent run --loop-mode cache-first --token-budget 4096 "检查 reports 并决定下一步行动"
HyperAgent events --limit 20
HyperAgent replay --run-id <run_id>
HyperAgent stats
HyperAgent checkpoint --path hyperagent/runtime/action_loop.py --reason "修改前"
HyperAgent restore --checkpoint-id <checkpoint_id>
HyperAgent index --root hyperagent --root tests
HyperAgent skill-run --name review-experiment --arguments "评审最新实验结果"
```

## TUI / REPL

`HyperAgent tui` 提供全屏交互界面，支持：

- 鼠标滚轮查看历史内容。
- `PageUp/PageDown/Home/End` 滚动输出。
- `↑/↓` 切换历史命令。
- 鼠标点击输入区移动光标。
- `←/→`、`Backspace`、`Delete` 编辑输入。
- `/thinking on|off|toggle|status` 展开或折叠模型返回的思考内容。
- `/web`、`/image`、`/ide-context`、`/plan-mode`、`/feedback`、`/worktree` 等功能入口。

`HyperAgent repl` 复用同一套会话、工具权限和 Agent 逻辑，适合普通终端使用。

## 命令、Agents、Hooks 与 Todos

HyperAgent 支持 Claude Code 风格的 Markdown 命令和内置科研 subagent，但不复制 Claude Code 源码。内置命令包括 `/feature-dev`、`/code-review`、`/review-experiment`、`/commit`、`/commit-push-pr`、`/doctor`、`/permissions`、`/export` 和 `/bug`。

```bash
HyperAgent command-list
HyperAgent command-render --name feature-dev --arguments "增加一个结果评审 Agent"
HyperAgent agent-run --agent code-reviewer --instruction "审查当前 diff"
HyperAgent agent-tool todo-write --owner project --item "检查实验报告"
HyperAgent todos --owner project
HyperAgent doctor
```

项目级命令可以放到 `.hyperagent/commands/*.md`，项目级 agent 可以放到 `.hyperagent/agents/*.md`，hook 可以通过 `/hooks add` 或 `.hyperagent/hooks/*.md` 管理。外部 Bot 渠道仍然只允许聊天/查询，不能触发 shell、训练、写文件或通用 agent 工具。

## Reasonix 风格运行时

HyperAgent 参考 DeepSeek-Reasonix 的架构思想重新用 Python 实现，不复制其 TypeScript 源码。当前阶段已经补上：

- `--loop-mode cache-first`：ActionLoop 只对稳定 system/cache guidance 前缀记录 hash，把 repo、task、session 和 tool 输出放到后续 volatile 消息，便于后续利用 DeepSeek prefix cache。
- native `tool_calls` 解析、JSON list/actions/tool_calls 修复、reasoning-content 修复、直接工具 action 归一化，以及重复工具调用 storm breaker。
- `.hyperagent/events/runtime_events.jsonl` 事件日志，可用 `HyperAgent events`、`HyperAgent replay`、`HyperAgent stats` 查看；replay 会展示 response、repair、tool step、final、paused 和 max-step 里程碑。
- `action_run.json` 记录单次 action run 的累计 token budget、usage event id、cache hit/miss token 和 cache-hit ratio。
- `.hyperagent/checkpoints` 可恢复文件检查点，可用 `HyperAgent checkpoint`、`HyperAgent restore`、`/checkpoint`、`/restore`。
- 内置 HSI skills：`review-experiment`、`spectral-critic`、`paper-method-extractor` 等。
- `HyperAgent index` 轻量词法索引，作为后续 embedding 语义检索的稳定接口。

这还不是完整 Reasonix 克隆：实时 MCP client、富 dashboard/desktop、编辑审查 gate 和后台 job 管理仍在后续阶段。

## DeepSeek / Reasonix

DeepSeek 兼容 OpenAI 风格接口，支持模型选择、thinking 模式、reasoning effort 和 JSON 输出：

```bash
HyperAgent llm-send \
  --provider deepseek \
  --model deepseek-v4-pro \
  --thinking enabled \
  --reasoning-effort max \
  --json-output \
  --user "返回下一轮 HSI 实验计划 JSON"
```

预设配置：

```bash
HyperAgent llm-profile
HyperAgent --reasonix-profile reasonix-deep "诊断失败实验"
```

## 语言切换

默认界面语言是中文 `zh-CN`。临时切换英文：

```bash
HyperAgent --lang en status --help
HYPERAGENT_LANG=en HyperAgent status --help
```

设置工作区默认语言：

```bash
HyperAgent language-list
HyperAgent language-set zh-CN
HyperAgent language-set en
```

安装和导出语言包：

```bash
HyperAgent language-install --path my-language.json
HyperAgent language-export --locale zh-CN --output zh-CN.template.json
```

语言包只影响界面、帮助和固定提示文本，不会翻译模型输出、system prompt、实验报告正文或历史 worklog。

## 数据集与大文件

大型数据集和实验产物不进入 git。数据集下载链接应记录在 `dataset/README.md` 或数据集目录说明中。

当前工作区默认数据集根目录：

```text
/data2/lzj/lab/Mamba_test/dataset
```

## 当前限制

- `.mat` 支持最成熟，TIFF 和 ENVI 已有基础读取能力。
- baseline 主要是 SVM 和轻量 MLP。
- SSRN、SpectralFormer、HyperMamba、GCN 等复杂模型仍属于后续阶段。
- 中文语言包第一版优先覆盖 CLI/launcher/REPL/TUI 的常用文案。
