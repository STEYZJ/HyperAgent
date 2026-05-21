# HyperAgent 中文文档

HyperAgent 是一个面向高光谱图像分类研究的解耦式 Agent 框架。当前版本重点打通可复现实验闭环：

```text
数据审计 -> 光谱诊断 -> 模型推荐 -> 实验计划 -> baseline 训练 -> 报告
```

## 快速开始

推荐使用项目 conda 环境：

```bash
conda activate HyperAgent
HyperAgent demo --synthetic
```

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

## 常用命令

```bash
HyperAgent init --dataset-root /data2/lzj/lab/Mamba_test/dataset
HyperAgent status
HyperAgent demo --synthetic
HyperAgent tui
HyperAgent repl --permission ask
HyperAgent agent-run --agent reviewer --instruction "检查最近一次实验报告"
HyperAgent experiment-cycle --plan experiments/run/plan.yaml --result experiments/run/result.json --audit reports/audit.json
```

Claude Code 风格入口也可用：

```bash
HyperAgent "分析最新实验结果并提出下一步实验"
HyperAgent plan "把 module_proposal.json 物化为模型 factory"
HyperAgent act "检查 reports 并选择下一条安全命令"
HyperAgent /help
HyperAgent /tui
HyperAgent /language
```

## TUI / REPL

`HyperAgent tui` 提供全屏交互界面，支持：

- 鼠标滚轮查看历史内容。
- `PageUp/PageDown/Home/End` 滚动输出。
- `↑/↓` 切换历史命令。
- 鼠标点击输入区移动光标。
- `←/→`、`Backspace`、`Delete` 编辑输入。
- `/thinking on|off|toggle|status` 展开或折叠模型返回的思考内容。

`HyperAgent repl` 复用同一套会话、工具权限和 Agent 逻辑，适合普通终端使用。

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
