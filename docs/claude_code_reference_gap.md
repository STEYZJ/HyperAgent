# Claude Code Reference Gap Notes

Source reviewed locally:

- `参考/秋芝 2046/Claude Code完全教程【视频文档】.pdf`

Large local videos and reference PDFs are ignored by Git via `参考/`.

## Extracted Behaviors

- Default interactive terminal mode plus planning and edit-acceptance modes.
- Permission choices for local commands: one-time approval, remembered approval,
  and denial.
- File/context input conventions such as local file references and image input.
- Slash commands for help, model switching, temporary side chat, simplification,
  rewind, compact, clear, context, resume, init, memory, agents, plugin.
- Project memory via a project-level Claude markdown file.
- Auto memory for project-specific habits and lessons.
- Extension ecosystem covering Skills, MCP, CLI tools, SubAgents, Hooks, and Plugins.
- Context management workflow: inspect context, compact above a threshold, clear
  when needed, resume prior sessions.

## HyperAgent Mapping

| Claude Code concept | HyperAgent status |
| --- | --- |
| `claude` interactive entry | `HyperAgent` / `HyperAgent repl` |
| `/help` | Implemented |
| `/model` | Implemented as provider/model list |
| `/context` | Implemented in REPL |
| `/compact` | Implemented in REPL and CLI |
| `/resume` | Implemented in launcher/REPL |
| `/clear` | Implemented in REPL with rewind snapshot |
| `/init` | Implemented as project `HyperAgent.md` memory |
| `/memory` | Implemented as project/user/auto markdown stores |
| `/btw` | Implemented as isolated non-persistent question |
| `/rewind` | Implemented as snapshot list/save |
| `/agents` | Implemented as lightweight subagent registry |
| `/hooks` | Implemented as lightweight hook registry |
| `/plugin` | Implemented as lightweight plugin registry |
| `/simplify` | Implemented as local council prompt scaffold |
| Accept Edits mode | Partially covered by `--permission ask` and `deny-write` |
| Status line | Not yet persistent; `/context` gives status details |
| Rich TUI | Not yet; stdlib REPL only |
| Remembered command approvals | Not yet; permission mode is per run |

## Next Gaps

- Persistent command approval rules, similar to remembered permissions.
- Full-screen TUI with status line and context meter.
- Native multi-agent simplify council that runs three reviewers and applies a
  guarded patch.
- Hook execution lifecycle around commits, tests, and task completion.
- Plugin bundles that package skills, subagents, hooks, and MCP specs together.
