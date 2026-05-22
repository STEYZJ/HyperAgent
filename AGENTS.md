# HyperAgent Agent Instructions

HyperAgent is the orchestration layer for research workflows. For paper-learning tasks, extract transferable research experience rather than paper content.

Use these commands:

- `HyperAgent research extract --paper <path-or-id> --json`
- `HyperAgent research pattern --paper <path-or-id> --json`
- `HyperAgent research experiment --paper <path-or-id> --json`
- `HyperAgent research storytelling --paper <path-or-id> --json`
- `HyperAgent research taste --field <field> --papers <paper-a,paper-b> --json`
- `HyperAgent research consolidate --topic "baseline selection" --json`
- `HyperAgent research-mcp-serve`

HyperVault remains the storage/RAG backend. Configure it with `HYPERVAULT_URL`, `HYPERVAULT_ROOT`, or `HYPERVAULT_VAULT_PATH`. Do not commit `.env`, API keys, runtime state, paper PDFs with restricted licenses, or private reviewer notes.

Every strategy lesson must include a claim, why it works, evidence span, transferable template, risk/limit, and confidence. Avoid method summaries such as "the paper proposes X" unless the sentence explains how the authors package, justify, or defend X.
