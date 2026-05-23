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
| Status line | First-round TUI status line shows provider/model, session, permission, context, tokens, cache, and wait status |
| Rich TUI | First-round stdlib TUI exists with side panel for context, usage, subagents, jobs, suggestions, todos, artifacts, and permission counts |
| Remembered command approvals | First-round exact remembered approvals stored locally under `.hyperagent/permissions/remembered.json` |

## Next Gaps

- Broader permission UI, such as grouped risk views and expiry policies.
- Deeper TUI interaction polish, such as a command palette and permission-detail panel.
- Native multi-agent simplify council that runs three reviewers and applies a
  guarded patch.
- Hook execution lifecycle around commits and tests; task completion now has a
  lightweight `TaskComplete` event.
- Plugin bundles that package skills, subagents, hooks, and MCP specs together.
