"""Command line interface for HyperAgent."""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from hyperagent.agents import BenchmarkAgent, CoordinatorAgent, ExperimentAutopilotAgent, ResearchExperienceAgent
from hyperagent.core.io import read_json, read_yaml, write_json
from hyperagent.core.worklog import append_worklog
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor, tool_catalog
from hyperagent.runtime.channels import (
    ChannelConfigStore,
    ChannelRouter,
    register_builtin_channel_platforms,
)
from hyperagent.runtime.channels.gateway import create_channel_app
from hyperagent.runtime.checkpoints import CheckpointStore
from hyperagent.runtime.commands import SlashCommandStore
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.events import RuntimeEventLog
from hyperagent.runtime.feature_state import (
    FeedbackStore,
    IDEContextStore,
    PersonalityStore,
    PlanModeStore,
    image_status,
    web_status,
    worktree_status,
)
from hyperagent.runtime.general_agent import GeneralAgentRunner
from hyperagent.runtime.hooks import HookEngine
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.tui import HyperAgentTui
from hyperagent.runtime.todos import TodoStore
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.env import load_env_file
from hyperagent.runtime.command_aliases import command_help_text
from hyperagent.runtime.deepseek_reasonix import (
    get_reasonix_profile,
    list_reasonix_profiles,
    reasonix_cache_guidance,
)
from hyperagent.runtime.i18n import I18nStore, Translator
from hyperagent.runtime.llm import LLMClient, LLMProviderStore, LLMRequestBuilder
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.obsidian import ObsidianVaultIndex
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.skills import SkillStore
from hyperagent.runtime.repl import HyperAgentRepl
from hyperagent.runtime.semantic_index import SemanticIndexStore
from hyperagent.schemas import (
    ChannelInboundMessage,
    DatasetAudit,
    ExperimentCycle,
    ExperimentPlan,
    ExperimentResult,
    LLMMessage,
    MCPServerSpec,
    LiteratureSearchResult,
    ModuleProposal,
    ModelRecommendation,
    SpectralReport,
)
from hyperagent.runtime.subagents import SubagentRuntimeRegistry
from hyperagent.tools.module_materializer import ModuleMaterializer
from hyperagent.training.experiment_suite import ExperimentSuiteRunner


DEFAULT_DATASET_ROOT = "/data2/lzj/lab/Mamba_test/dataset"
PACKAGE_ROOT = Path(__file__).resolve().parent
_ACTIVE_TRANSLATOR: Optional[Translator] = None


def _txt(translator: Optional[Translator], key: str, default: str, **kwargs) -> str:
    if translator is not None:
        return translator.t(key, default=default, **kwargs)
    try:
        return default.format(**kwargs)
    except (KeyError, ValueError):
        return default


def _label(translator: Optional[Translator], name: str, default: str) -> str:
    return _txt(translator, f"cli.label.{name}", default)


def _value(translator: Optional[Translator], value) -> str:
    if isinstance(value, bool):
        return _txt(translator, "cli.value.true", "true") if value else _txt(translator, "cli.value.false", "false")
    if value is None or value == "":
        return _txt(translator, "cli.value.none", "none")
    return str(value)


def _configured(translator: Optional[Translator], is_configured: bool) -> str:
    return (
        _txt(translator, "cli.value.configured", "configured")
        if is_configured
        else _txt(translator, "cli.value.missing", "missing")
    )


def _kv(translator: Optional[Translator], name: str, default: str, value) -> str:
    return f"{_label(translator, name, default)}: {_value(translator, value)}"


def _warning(translator: Optional[Translator], warning: str) -> str:
    return f"{_label(translator, 'warning', 'warning')}: {warning}"


def _parser_class(translator: Optional[Translator]):
    class LocalizedArgumentParser(argparse.ArgumentParser):
        def format_help(self) -> str:
            return _localize_argparse_text(super().format_help(), translator)

        def format_usage(self) -> str:
            return _localize_argparse_text(super().format_usage(), translator)

        def error(self, message: str) -> None:
            if translator is not None and translator.locale == "zh-CN":
                self.print_usage(sys.stderr)
                localized = _localize_argparse_message(message)
                self.exit(2, f"{self.prog}: 错误: {localized}\n")
            super().error(message)

    return LocalizedArgumentParser


def _localize_argparse_message(message: str) -> str:
    replacements = {
        "unrecognized arguments:": "无法识别的参数:",
        "the following arguments are required:": "缺少必需参数:",
        "invalid choice:": "无效选项:",
        "expected one argument": "需要一个参数",
        "argument": "参数",
    }
    localized = message
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized


def _localize_argparse_text(text: str, translator: Optional[Translator]) -> str:
    if translator is None or translator.locale != "zh-CN":
        return text
    replacements = {
        "usage:": "用法:",
        "positional arguments:": "位置参数:",
        "optional arguments:": "可选参数:",
        "options:": "可选参数:",
        "show this help message and exit": "显示此帮助信息并退出",
    }
    localized = text
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized


def _build_parser(translator: Optional[Translator] = None) -> argparse.ArgumentParser:
    parser_cls = _parser_class(translator)
    parser = parser_cls(
        description=_txt(
            translator,
            "cli.description",
            "HyperAgent HSI research workflow",
        )
    )
    parser.add_argument(
        "--lang",
        default=None,
        help=_txt(
            translator,
            "cli.lang.help",
            "Interface language, for example zh-CN or en.",
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=parser_cls,
    )

    audit = subparsers.add_parser(
        "audit",
        help=_txt(translator, "cli.command.audit.help", "Inspect an HSI dataset"),
    )
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--reader", default=None)

    plan = subparsers.add_parser(
        "plan",
        help=_txt(
            translator,
            "cli.command.plan.help",
            "Build an experiment plan from an audit",
        ),
    )
    plan.add_argument("--audit", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--output-dir", default=None)
    plan.add_argument("--seed", type=int, default=42)

    run = subparsers.add_parser("run-baseline", help="Run a baseline experiment")
    run.add_argument("--config", required=True)

    run_suite = subparsers.add_parser(
        "run-suite",
        help="Run one experiment plan across multiple seeds and summarize variance",
    )
    run_suite.add_argument("--config", required=True)
    run_suite.add_argument("--seeds", default="42,43,44")
    run_suite.add_argument("--output-dir", default=None)
    run_suite.add_argument("--suite-name", default=None)

    benchmark_list = subparsers.add_parser(
        "benchmark-list",
        help="List catalogued HSI benchmark datasets",
    )
    benchmark_list.add_argument("--catalog", default="dataset/datasets.yaml")
    benchmark_list.add_argument("--json", action="store_true")

    benchmark_matrix = subparsers.add_parser(
        "benchmark-matrix",
        help="Audit, plan, and optionally run multi-seed suites for catalogued benchmarks",
    )
    benchmark_matrix.add_argument("--catalog", default="dataset/datasets.yaml")
    benchmark_matrix.add_argument("--datasets", default="")
    benchmark_matrix.add_argument("--reports-root", default="reports/benchmark_matrix")
    benchmark_matrix.add_argument("--experiments-root", default="experiments/benchmark_matrix")
    benchmark_matrix.add_argument("--seeds", default="42,43")
    benchmark_matrix.add_argument("--run-suite", action="store_true")

    report = subparsers.add_parser("report", help="Build a Markdown report")
    report.add_argument("--experiment", required=True)
    report.add_argument("--output", default=None)

    demo = subparsers.add_parser("demo", help="Run a synthetic end-to-end demo")
    demo.add_argument("--synthetic", action="store_true", required=True)
    demo.add_argument("--root", default="experiments/synthetic_demo")
    demo.add_argument("--seed", type=int, default=42)

    init = subparsers.add_parser("init", help="Initialize a HyperAgent CLI workspace")
    init.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    init.add_argument("--output-root", default="experiments")
    init.add_argument("--reports-root", default="reports")
    init.add_argument("--literature-root", default="literature/papers")
    init.add_argument("--default-provider", default="arxiv")
    init.add_argument("--default-year-from", type=int, default=2024)

    status_help = _txt(
        translator,
        "cli.command.status.help",
        "Show HyperAgent workspace status",
    )
    status = subparsers.add_parser(
        "status",
        help=status_help,
        description=status_help,
    )
    status.add_argument(
        "--json",
        action="store_true",
        help=_txt(translator, "cli.arg.status_json.help", "Print status as JSON."),
    )

    task_create = subparsers.add_parser("task-create", help="Create a research task")
    task_create.add_argument("--goal", required=True)
    task_create.add_argument("--dataset", required=True)
    task_create.add_argument("--objective", default="maximize_oa_with_reproducible_baseline")
    task_create.add_argument("--keywords", default="")

    task_list = subparsers.add_parser("task-list", help="List research tasks")
    task_list.add_argument("--json", action="store_true")

    task_show = subparsers.add_parser("task-show", help="Show a research task")
    task_show.add_argument("--task-id", required=True)
    task_show.add_argument("--json", action="store_true")

    task_run = subparsers.add_parser("task-run", help="Run task planning workflow")
    task_run.add_argument("--task-id", required=True)
    task_run.add_argument("--with-literature", action="store_true")
    task_run.add_argument("--run-baseline", action="store_true")
    task_run.add_argument("--provider", default=None)
    task_run.add_argument("--max-literature-results", type=int, default=5)
    task_run.add_argument("--seed", type=int, default=42)

    literature = subparsers.add_parser("literature", help="Search latest related literature")
    literature.add_argument("--query", required=True)
    literature.add_argument("--output", required=True)
    literature.add_argument("--provider", default="arxiv")
    literature.add_argument("--max-results", type=int, default=10)
    literature.add_argument("--year-from", type=int, default=None)
    literature.add_argument("--sort-by", default="latest")


    research = subparsers.add_parser(
        "research",
        help="Extract and search transferable paper research experience",
    )
    research_sub = research.add_subparsers(dest="research_command", required=True)

    research_extract = research_sub.add_parser("extract", help="Extract a full paper strategy card")
    research_extract.add_argument("--paper", required=True)
    research_extract.add_argument("--provider", default="")
    research_extract.add_argument("--model", default=None)
    research_extract.add_argument("--field", default="")
    research_extract.add_argument("--title", default="")
    research_extract.add_argument("--venue", default="")
    research_extract.add_argument("--year", type=int, default=None)
    research_extract.add_argument("--no-write", action="store_true")
    research_extract.add_argument("--json", action="store_true")

    for command_name, help_text in [
        ("pattern", "Extract why-this-paper-can-publish research-pattern lessons"),
        ("experiment", "Extract baseline, ablation, control-variable, robustness, and visualization strategy"),
        ("storytelling", "Extract scientific storytelling and reviewer-persuasion strategy"),
    ]:
        parser_item = research_sub.add_parser(command_name, help=help_text)
        parser_item.add_argument("--paper", required=True)
        parser_item.add_argument("--provider", default="")
        parser_item.add_argument("--model", default=None)
        parser_item.add_argument("--field", default="")
        parser_item.add_argument("--no-write", action="store_true")
        parser_item.add_argument("--json", action="store_true")

    research_taste = research_sub.add_parser("taste", help="Compare papers to extract research taste")
    research_taste.add_argument("--field", required=True)
    research_taste.add_argument("--papers", default="")
    research_taste.add_argument("--provider", default="")
    research_taste.add_argument("--model", default=None)
    research_taste.add_argument("--json", action="store_true")

    research_consolidate = research_sub.add_parser("consolidate", help="Consolidate research experience into long-term memory")
    research_consolidate.add_argument("--topic", required=True)
    research_consolidate.add_argument("--field", default="")
    research_consolidate.add_argument("--papers", default="")
    research_consolidate.add_argument("--provider", default="")
    research_consolidate.add_argument("--model", default=None)
    research_consolidate.add_argument("--json", action="store_true")

    research_search = research_sub.add_parser("search", help="Search existing research-experience strategy cards")
    research_search.add_argument("--query", required=True)
    research_search.add_argument("--dimension", required=True)
    research_search.add_argument("--field", default="")
    research_search.add_argument("--top-k", type=int, default=8)
    research_search.add_argument("--json", action="store_true")

    subparsers.add_parser("research-mcp-serve", help="Serve research-experience tools over stdio MCP")

    auto = subparsers.add_parser(
        "auto-experiment",
        help="Build an evidence-backed experiment agenda",
    )
    auto.add_argument("--audit", required=True)
    auto.add_argument("--spectral", required=True)
    auto.add_argument("--recommendation", required=True)
    auto.add_argument("--output", required=True)
    auto.add_argument("--objective", default="maximize_oa_with_reproducible_baseline")
    auto.add_argument("--max-candidates", type=int, default=4)

    tune = subparsers.add_parser("tune-next", help="Suggest purposeful parameter changes")
    tune.add_argument("--plan", required=True)
    tune.add_argument("--result", required=True)
    tune.add_argument("--audit", required=True)
    tune.add_argument("--output", required=True)

    experiment_cycle = subparsers.add_parser(
        "experiment-cycle",
        help="Analyze a completed experiment and build or run the next one",
    )
    experiment_cycle.add_argument("--plan", default=None)
    experiment_cycle.add_argument("--result", default=None)
    experiment_cycle.add_argument("--audit", default=None)
    experiment_cycle.add_argument("--resume-paused", default=None)
    experiment_cycle.add_argument("--output-root", default="experiments/autopilot")
    experiment_cycle.add_argument("--objective", default="maximize_oa_with_reproducible_baseline")
    experiment_cycle.add_argument("--target-oa", type=float, default=0.9)
    experiment_cycle.add_argument("--run-next", action="store_true")
    experiment_cycle.add_argument("--max-repeated-parameter", type=int, default=2)
    experiment_cycle.add_argument(
        "--council-mode",
        choices=["executable", "static"],
        default="executable",
    )
    experiment_cycle.add_argument("--llm-council", action="store_true")
    experiment_cycle.add_argument("--council-profile", default="reasonix-balanced")
    experiment_cycle.add_argument("--council-llm-budget", type=int, default=3)
    experiment_cycle.add_argument("--llm-gate-token-budget", type=int, default=4096)
    experiment_cycle.add_argument("--llm-gate-retry-sec", type=int, default=30)

    module = subparsers.add_parser(
        "propose-module",
        help="Suggest an evidence-backed module addition",
    )
    module.add_argument("--audit", required=True)
    module.add_argument("--spectral", required=True)
    module.add_argument("--literature", required=True)
    module.add_argument("--output", required=True)
    module.add_argument("--objective", default="improve_spectral_spatial_modeling")

    llm_list = subparsers.add_parser("llm-providers", help="List configured LLM providers")
    llm_list.add_argument("--json", action="store_true")

    llm_profile = subparsers.add_parser(
        "llm-profile",
        help="List DeepSeek Reasonix-inspired model/runtime profiles",
    )
    llm_profile.add_argument("--profile", default=None)
    llm_profile.add_argument("--json", action="store_true")

    llm_usage = subparsers.add_parser(
        "llm-usage",
        help="Summarize LLM token usage, cache hits, and optional cost estimates",
    )
    llm_usage.add_argument("--limit", type=int, default=None)
    llm_usage.add_argument("--json", action="store_true")

    web = subparsers.add_parser(
        "web",
        help=_txt(translator, "cli.command.web.help", "Use controlled web tools"),
    )
    web_sub = web.add_subparsers(dest="web_command", required=True)
    web_status_cmd = web_sub.add_parser("status", help="Show controlled web status")
    web_status_cmd.add_argument("--json", action="store_true")
    web_search = web_sub.add_parser("search", help="Search the web through a configured provider")
    web_search.add_argument("--query", required=True)
    web_search.add_argument("--provider", default="auto")
    web_search.add_argument("--max-results", type=int, default=5)
    web_search.add_argument("--timeout-sec", type=int, default=20)
    web_search.add_argument("--permission", choices=["auto", "ask", "session-ask", "deny"], default="session-ask")
    web_search.add_argument("--json", action="store_true")
    web_fetch = web_sub.add_parser("fetch", help="Fetch and extract a public HTTP(S) URL")
    web_fetch.add_argument("--url", required=True)
    web_fetch.add_argument("--max-chars", type=int, default=12000)
    web_fetch.add_argument("--timeout-sec", type=int, default=20)
    web_fetch.add_argument("--permission", choices=["auto", "ask", "session-ask", "deny"], default="session-ask")
    web_fetch.add_argument("--json", action="store_true")
    web_cite = web_sub.add_parser("cite", help="List recent web citations")
    web_cite.add_argument("--citation-id", default="")
    web_cite.add_argument("--limit", type=int, default=20)
    web_cite.add_argument("--json", action="store_true")

    image = subparsers.add_parser(
        "image",
        help=_txt(translator, "cli.command.image.help", "Use controlled image tools"),
    )
    image_sub = image.add_subparsers(dest="image_command", required=True)
    image_status_cmd = image_sub.add_parser("status", help="Show image tool status")
    image_status_cmd.add_argument("--json", action="store_true")
    image_gen = image_sub.add_parser("generate", help="Create an image-generation request artifact")
    image_gen.add_argument("--prompt", required=True)
    image_gen.add_argument("--permission", choices=["auto", "ask", "session-ask", "deny"], default="session-ask")
    image_gen.add_argument("--json", action="store_true")
    image_edit = image_sub.add_parser("edit", help="Create an image-edit request artifact")
    image_edit.add_argument("--image", required=True)
    image_edit.add_argument("--instruction", required=True)
    image_edit.add_argument("--permission", choices=["auto", "ask", "session-ask", "deny"], default="session-ask")
    image_edit.add_argument("--json", action="store_true")

    ide_context = subparsers.add_parser("ide-context", help="Manage manually supplied IDE context")
    ide_context.add_argument("action", choices=["status", "on", "off", "set-open-files", "clear"])
    ide_context.add_argument("paths", nargs="*")
    ide_context.add_argument("--json", action="store_true")

    plan_mode = subparsers.add_parser("plan-mode", help="Toggle plan-only mode")
    plan_mode.add_argument("action", choices=["status", "on", "off"])
    plan_mode.add_argument("reason", nargs="*")
    plan_mode.add_argument("--json", action="store_true")

    personality = subparsers.add_parser("personality", help="Manage local interaction personality note")
    personality.add_argument("action", choices=["status", "set", "clear"])
    personality.add_argument("text", nargs="*")
    personality.add_argument("--json", action="store_true")

    feedback = subparsers.add_parser("feedback", help="Record or list local feedback notes")
    feedback.add_argument("action", choices=["add", "list"])
    feedback.add_argument("text", nargs="*")
    feedback.add_argument("--json", action="store_true")
    feedback.add_argument("--limit", type=int, default=20)

    worktree = subparsers.add_parser("worktree", help="Show git worktree status without modifying it")
    worktree.add_argument("--json", action="store_true")

    llm_dry = subparsers.add_parser("llm-dry-run", help="Build a vendor request payload without sending it")
    llm_dry.add_argument("--provider", required=True)
    llm_dry.add_argument("--model", default=None)
    llm_dry.add_argument("--system", default="You are HyperAgent.")
    llm_dry.add_argument("--user", required=True)
    llm_dry.add_argument("--temperature", type=float, default=0.2)
    llm_dry.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(llm_dry, translator)

    llm_send = subparsers.add_parser("llm-send", help="Send a prompt to a configured LLM provider")
    llm_send.add_argument("--provider", required=True)
    llm_send.add_argument("--model", default=None)
    llm_send.add_argument("--system", default="You are HyperAgent.")
    llm_send.add_argument("--user", required=True)
    llm_send.add_argument("--temperature", type=float, default=0.2)
    llm_send.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(llm_send, translator)
    llm_send.add_argument("--output", default=None)

    agent_chat = subparsers.add_parser(
        "agent-chat",
        help="Run one persistent conversation-backed LLM agent turn",
    )
    agent_chat.add_argument("--session-id", default=None)
    agent_chat.add_argument("--new-title", default=None)
    agent_chat.add_argument("--provider", default="deepseek")
    agent_chat.add_argument("--model", default=None)
    agent_chat.add_argument("--message", required=True)
    agent_chat.add_argument(
        "--mode",
        choices=["research", "code", "algorithm"],
        default="research",
    )
    agent_chat.add_argument("--task-id", default=None)
    agent_chat.add_argument("--temperature", type=float, default=0.2)
    agent_chat.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(agent_chat, translator)
    agent_chat.add_argument("--max-context-chars", type=int, default=12000)
    agent_chat.add_argument("--no-auto-compress", action="store_true")
    agent_chat.add_argument("--output", default=None)

    headless_run = subparsers.add_parser(
        "run",
        help="Run one headless HyperAgent action-loop instruction",
    )
    headless_run.add_argument("instruction", nargs="+")
    headless_run.add_argument("--session-id", default=None)
    headless_run.add_argument("--new-title", default=None)
    headless_run.add_argument("--provider", default="deepseek")
    headless_run.add_argument("--model", default=None)
    headless_run.add_argument("--task-id", default=None)
    headless_run.add_argument("--max-steps", type=int, default=3)
    headless_run.add_argument("--temperature", type=float, default=0.2)
    headless_run.add_argument("--max-tokens", type=int, default=None)
    headless_run.add_argument("--max-files", type=int, default=12)
    headless_run.add_argument("--max-preview-chars", type=int, default=1000)
    headless_run.add_argument("--loop-mode", choices=["standard", "cache-first"], default="standard")
    headless_run.add_argument("--token-budget", type=int, default=None)
    headless_run.add_argument(
        "--permission",
        choices=["auto", "ask", "session-ask", "deny-write", "deny"],
        default="session-ask",
    )
    add_llm_runtime_args(headless_run, translator)

    repl = subparsers.add_parser(
        "repl",
        help=_txt(
            translator,
            "cli.command.repl.help",
            "Start an interactive Claude-Code-like HyperAgent REPL",
        ),
        description=_txt(
            translator,
            "cli.command.repl.help",
            "Start an interactive Claude-Code-like HyperAgent REPL",
        ),
    )
    repl.add_argument(
        "--session-id",
        default=None,
        help=_txt(translator, "cli.arg.session_id.help", "Conversation session id to resume."),
    )
    repl.add_argument(
        "--new-title",
        default=None,
        help=_txt(translator, "cli.arg.new_title.help", "Title for a new conversation session."),
    )
    repl.add_argument(
        "--provider",
        default="deepseek",
        help=_txt(translator, "cli.arg.provider.help", "LLM provider name."),
    )
    repl.add_argument(
        "--model",
        default=None,
        help=_txt(translator, "cli.arg.model.help", "Provider model name."),
    )
    repl.add_argument(
        "--mode",
        choices=["research", "code", "algorithm"],
        default="research",
        help=_txt(translator, "cli.arg.mode.help", "Agent working mode."),
    )
    repl.add_argument(
        "--task-id",
        default=None,
        help=_txt(translator, "cli.arg.task_id.help", "Optional research task id."),
    )
    repl.add_argument(
        "--permission",
        choices=["auto", "ask", "session-ask", "deny-write", "deny"],
        default="ask",
        help=_txt(translator, "cli.arg.permission.help", "Tool permission policy."),
    )
    repl.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help=_txt(translator, "cli.arg.max_context_chars.help", "Maximum context characters before compression."),
    )
    repl.add_argument(
        "--keep-last",
        type=int,
        default=6,
        help=_txt(translator, "cli.arg.keep_last.help", "Messages kept verbatim during context compression."),
    )
    add_llm_runtime_args(repl, translator)

    tui = subparsers.add_parser(
        "tui",
        help=_txt(
            translator,
            "cli.command.tui.help",
            "Start the fullscreen curses HyperAgent interface",
        ),
        description=_txt(
            translator,
            "cli.command.tui.help",
            "Start the fullscreen curses HyperAgent interface",
        ),
    )
    tui.add_argument(
        "--session-id",
        default=None,
        help=_txt(translator, "cli.arg.session_id.help", "Conversation session id to resume."),
    )
    tui.add_argument(
        "--new-title",
        default=None,
        help=_txt(translator, "cli.arg.new_title.help", "Title for a new conversation session."),
    )
    tui.add_argument(
        "--provider",
        default="deepseek",
        help=_txt(translator, "cli.arg.provider.help", "LLM provider name."),
    )
    tui.add_argument(
        "--model",
        default=None,
        help=_txt(translator, "cli.arg.model.help", "Provider model name."),
    )
    tui.add_argument(
        "--mode",
        choices=["research", "code", "algorithm"],
        default="research",
        help=_txt(translator, "cli.arg.mode.help", "Agent working mode."),
    )
    tui.add_argument(
        "--task-id",
        default=None,
        help=_txt(translator, "cli.arg.task_id.help", "Optional research task id."),
    )
    tui.add_argument(
        "--permission",
        choices=["auto", "ask", "session-ask", "deny-write", "deny"],
        default="session-ask",
        help=_txt(translator, "cli.arg.permission.help", "Tool permission policy."),
    )
    tui.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help=_txt(translator, "cli.arg.max_context_chars.help", "Maximum context characters before compression."),
    )
    tui.add_argument(
        "--keep-last",
        type=int,
        default=6,
        help=_txt(translator, "cli.arg.keep_last.help", "Messages kept verbatim during context compression."),
    )
    add_llm_runtime_args(tui, translator)

    agent_context = subparsers.add_parser(
        "agent-context",
        help="Build a compact repository context snapshot",
    )
    agent_context.add_argument("--query", default="")
    agent_context.add_argument("--max-files", type=int, default=20)
    agent_context.add_argument("--max-preview-chars", type=int, default=1200)
    agent_context.add_argument("--output", default=None)
    agent_context.add_argument("--format", choices=["markdown", "json"], default="markdown")

    agent_plan = subparsers.add_parser(
        "agent-plan",
        help="Generate a saved Claude-Code-like coding/algorithm plan",
    )
    agent_plan.add_argument("--session-id", default=None)
    agent_plan.add_argument("--new-title", default=None)
    agent_plan.add_argument("--provider", default="deepseek")
    agent_plan.add_argument("--model", default=None)
    agent_plan.add_argument("--instruction", required=True)
    agent_plan.add_argument(
        "--mode",
        choices=["research", "code", "algorithm"],
        default="code",
    )
    agent_plan.add_argument("--task-id", default=None)
    agent_plan.add_argument("--temperature", type=float, default=0.2)
    agent_plan.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(agent_plan, translator)
    agent_plan.add_argument("--max-context-chars", type=int, default=18000)
    agent_plan.add_argument("--max-files", type=int, default=20)
    agent_plan.add_argument("--max-preview-chars", type=int, default=1200)

    agent_act = subparsers.add_parser(
        "agent-act",
        help="Run a short LLM-controlled local tool-call loop",
    )
    agent_act.add_argument("--session-id", default=None)
    agent_act.add_argument("--new-title", default=None)
    agent_act.add_argument("--provider", default="deepseek")
    agent_act.add_argument("--model", default=None)
    agent_act.add_argument("--message", required=True)
    agent_act.add_argument("--task-id", default=None)
    agent_act.add_argument("--max-steps", type=int, default=3)
    agent_act.add_argument("--temperature", type=float, default=0.2)
    agent_act.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(agent_act, translator)
    agent_act.add_argument("--max-files", type=int, default=12)
    agent_act.add_argument("--max-preview-chars", type=int, default=1000)
    agent_act.add_argument("--loop-mode", choices=["standard", "cache-first"], default="standard")
    agent_act.add_argument("--token-budget", type=int, default=None)
    agent_act.add_argument(
        "--permission",
        choices=["auto", "ask", "session-ask", "deny-write", "deny"],
        default="session-ask",
    )

    agent_run = subparsers.add_parser(
        "agent-run",
        help="Run a registered project subagent with user-authorized tools",
    )
    agent_run.add_argument("--agent", required=True)
    agent_run.add_argument("--instruction", required=True)
    agent_run.add_argument("--session-id", default=None)
    agent_run.add_argument("--new-title", default=None)
    agent_run.add_argument("--provider", default="deepseek")
    agent_run.add_argument("--model", default=None)
    agent_run.add_argument("--task-id", default=None)
    agent_run.add_argument("--max-steps", type=int, default=3)
    agent_run.add_argument("--temperature", type=float, default=0.2)
    agent_run.add_argument("--max-tokens", type=int, default=None)
    agent_run.add_argument("--max-files", type=int, default=12)
    agent_run.add_argument("--max-preview-chars", type=int, default=1000)
    agent_run.add_argument("--loop-mode", choices=["standard", "cache-first"], default="standard")
    agent_run.add_argument("--token-budget", type=int, default=None)
    agent_run.add_argument(
        "--permission",
        choices=["ask", "session-ask", "deny-write", "deny"],
        default="session-ask",
    )
    add_llm_runtime_args(agent_run, translator)

    agent_status = subparsers.add_parser(
        "agent-status",
        help="Show active and recent subagent runtime state",
    )
    agent_status.add_argument("--json", action="store_true")

    agent_pause = subparsers.add_parser(
        "agent-pause",
        help="Pause new subagent spawning in this workspace",
    )
    agent_pause.add_argument("--reason", default="")
    agent_pause.add_argument("reason_text", nargs="*")

    subparsers.add_parser(
        "agent-resume",
        help="Resume subagent spawning in this workspace",
    )

    agent_stop = subparsers.add_parser(
        "agent-stop",
        help="Request stop for a running subagent",
    )
    agent_stop.add_argument("subagent_id")

    agent_tool = subparsers.add_parser(
        "agent-tool",
        help="Run a controlled Claude-Code-like local tool",
    )
    agent_tool.add_argument(
        "--permission",
        choices=["auto", "ask", "session-ask", "deny-write", "deny"],
        default="auto",
    )
    agent_tool_sub = agent_tool.add_subparsers(dest="tool_command", required=True)

    tool_read = agent_tool_sub.add_parser("read-file", help="Read a project text file")
    tool_read.add_argument("--path", required=True)
    tool_read.add_argument("--start-line", type=int, default=1)
    tool_read.add_argument("--max-lines", type=int, default=200)
    tool_read.add_argument("--run-id", default=None)
    tool_read.add_argument("--json", action="store_true")

    tool_search = agent_tool_sub.add_parser("search-code", help="Search text in project files")
    tool_search.add_argument("--query", required=True)
    tool_search.add_argument("--path", default=".")
    tool_search.add_argument("--max-results", type=int, default=50)
    tool_search.add_argument("--run-id", default=None)
    tool_search.add_argument("--json", action="store_true")

    tool_run = agent_tool_sub.add_parser("run-command", help="Run an allowlisted command")
    tool_run.add_argument("--timeout-sec", type=int, default=60)
    tool_run.add_argument("--run-id", default=None)
    tool_run.add_argument("--json", action="store_true")
    tool_run.add_argument("argv", nargs=argparse.REMAINDER)

    tool_experiment = agent_tool_sub.add_parser(
        "run-experiment",
        help="Run a HyperAgent experiment YAML through the controlled tool layer",
    )
    tool_experiment.add_argument("--plan", required=True)
    tool_experiment.add_argument("--seeds", default="")
    tool_experiment.add_argument("--output-dir", default=None)
    tool_experiment.add_argument("--suite-name", default=None)
    tool_experiment.add_argument("--run-id", default=None)
    tool_experiment.add_argument("--json", action="store_true")

    tool_check_patch = agent_tool_sub.add_parser("check-patch", help="Validate a unified diff with git apply --check")
    tool_check_patch.add_argument("--patch-file", required=True)
    tool_check_patch.add_argument("--run-id", default=None)
    tool_check_patch.add_argument("--json", action="store_true")

    tool_apply_patch = agent_tool_sub.add_parser("apply-patch", help="Apply a unified diff through git apply")
    tool_apply_patch.add_argument("--patch-file", required=True)
    tool_apply_patch.add_argument("--run-id", default=None)
    tool_apply_patch.add_argument("--json", action="store_true")

    tool_todo = agent_tool_sub.add_parser("todo-write", help="Replace TodoWrite items")
    tool_todo.add_argument("--owner", default="project")
    tool_todo.add_argument("--item", action="append", default=[])
    tool_todo.add_argument("--run-id", default=None)
    tool_todo.add_argument("--json", action="store_true")

    command_list = subparsers.add_parser("command-list", help="List Markdown slash commands")
    command_list.add_argument("--include-hidden", action="store_true")
    command_list.add_argument("--json", action="store_true")

    command_render = subparsers.add_parser("command-render", help="Render a Markdown slash command")
    command_render.add_argument("--name", required=True)
    command_render.add_argument("--arguments", default="")
    command_render.add_argument("--expand-shell", action="store_true")
    command_render.add_argument("--json", action="store_true")

    todos = subparsers.add_parser("todos", help="List, clear, or export TodoWrite state")
    todos.add_argument("--owner", default="project")
    todos.add_argument("--clear", action="store_true")
    todos.add_argument("--export", default=None)
    todos.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Run a local HyperAgent self-check")
    doctor.add_argument("--json", action="store_true")

    events = subparsers.add_parser("events", help="List runtime event-log records")
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--type", default=None)
    events.add_argument("--session-id", default=None)
    events.add_argument("--run-id", default=None)
    events.add_argument("--json", action="store_true")

    replay = subparsers.add_parser("replay", help="Replay runtime events for a session or run")
    replay.add_argument("--session-id", default=None)
    replay.add_argument("--run-id", default=None)
    replay.add_argument("--limit", type=int, default=200)
    replay.add_argument("--json", action="store_true")

    diff_cmd = subparsers.add_parser("diff", help="Show a unified diff between two text artifacts")
    diff_cmd.add_argument("--left", required=True)
    diff_cmd.add_argument("--right", required=True)
    diff_cmd.add_argument("--context", type=int, default=3)
    diff_cmd.add_argument("--json", action="store_true")

    stats = subparsers.add_parser("stats", help="Summarize runtime events and LLM usage")
    stats.add_argument("--json", action="store_true")

    prune_sessions = subparsers.add_parser("prune-sessions", help="List or prune archived sessions")
    prune_sessions.add_argument("--dry-run", action="store_true")
    prune_sessions.add_argument("--json", action="store_true")

    checkpoint = subparsers.add_parser("checkpoint", help="Create or list reversible file checkpoints")
    checkpoint.add_argument("--path", action="append", default=[])
    checkpoint.add_argument("--reason", default="")
    checkpoint.add_argument("--list", action="store_true")
    checkpoint.add_argument("--json", action="store_true")

    restore = subparsers.add_parser("restore", help="Restore files from a checkpoint")
    restore.add_argument("--checkpoint-id", required=True)
    restore.add_argument("--json", action="store_true")

    session_new = subparsers.add_parser("session-new", help="Create a saved conversation session")
    session_new.add_argument("--title", required=True)

    session_add = subparsers.add_parser("session-add", help="Append a message to a conversation")
    session_add.add_argument("--session-id", required=True)
    session_add.add_argument("--role", required=True, choices=["system", "user", "assistant", "tool"])
    session_add.add_argument("--content", required=True)

    session_list = subparsers.add_parser("session-list", help="List conversation sessions")
    session_list.add_argument("--include-archived", action="store_true")
    session_list.add_argument("--json", action="store_true")

    session_show = subparsers.add_parser("session-show", help="Show a conversation session")
    session_show.add_argument("--session-id", required=True)
    session_show.add_argument("--json", action="store_true")

    session_archive = subparsers.add_parser("session-archive", help="Archive a conversation")
    session_archive.add_argument("--session-id", required=True)

    session_delete = subparsers.add_parser("session-delete", help="Delete a conversation")
    session_delete.add_argument("--session-id", required=True)
    session_delete.add_argument("--hard", action="store_true")

    session_compress = subparsers.add_parser("session-compress", help="Compress conversation context")
    session_compress.add_argument("--session-id", required=True)
    session_compress.add_argument("--keep-last", type=int, default=4)
    session_compress.add_argument("--max-chars", type=int, default=None)

    skills = subparsers.add_parser("skill-list", help="List compatible SKILL.md skills")
    skills.add_argument("--json", action="store_true")

    skill_search = subparsers.add_parser("skill-search", help="Search compatible SKILL.md skills")
    skill_search.add_argument("query", nargs="?", default="")
    skill_search.add_argument("--json", action="store_true")

    skill_inspect = subparsers.add_parser("skill-inspect", help="Inspect a SKILL.md skill")
    skill_inspect.add_argument("--name", required=True)
    skill_inspect.add_argument("--json", action="store_true")

    skill_install = subparsers.add_parser(
        "skill-install",
        help="Install a local SKILL.md directory into this workspace",
    )
    skill_install.add_argument("--path", required=True)
    skill_install.add_argument("--name", default="")
    skill_install.add_argument("--json", action="store_true")

    skill_bundles = subparsers.add_parser("skill-bundles", help="List skill bundles")
    skill_bundles.add_argument("--json", action="store_true")

    skill_run = subparsers.add_parser("skill-run", help="Render or run a SKILL.md skill")
    skill_run.add_argument("--name", required=True)
    skill_run.add_argument("--arguments", default="")
    skill_run.add_argument("--run", action="store_true")
    skill_run.add_argument("--provider", default="deepseek")
    skill_run.add_argument("--model", default=None)
    skill_run.add_argument("--max-steps", type=int, default=2)
    skill_run.add_argument("--json", action="store_true")
    add_llm_runtime_args(skill_run, translator)

    mcp_add = subparsers.add_parser("mcp-add", help="Register an MCP server launch spec")
    mcp_add.add_argument("--name", required=True)
    mcp_add.add_argument("--command", dest="server_command", required=True)
    mcp_add.add_argument("--arg", action="append", default=[])
    mcp_add.add_argument("--env", action="append", default=[])
    mcp_add.add_argument("--description", default="")

    mcp_list = subparsers.add_parser("mcp-list", help="List MCP server specs")
    mcp_list.add_argument("--json", action="store_true")

    mcp_export = subparsers.add_parser("mcp-export", help="Export MCP client-style config")
    mcp_export.add_argument("--output", default=None)

    mcp_inspect = subparsers.add_parser("mcp-inspect", help="Inspect a registered MCP server spec")
    mcp_inspect.add_argument("--name", required=True)
    mcp_inspect.add_argument("--json", action="store_true")

    mcp_health = subparsers.add_parser("mcp-health", help="Report configured MCP server health metadata")
    mcp_health.add_argument("--json", action="store_true")

    index = subparsers.add_parser("index", help="Build or search a lightweight project semantic index")
    index.add_argument("--root", action="append", default=[])
    index.add_argument("--query", default="")
    index.add_argument("--limit", type=int, default=10)
    index.add_argument("--json", action="store_true")

    obsidian_index = subparsers.add_parser("obsidian-index", help="Index an Obsidian vault")
    obsidian_index.add_argument("--vault", required=True)

    obsidian_search = subparsers.add_parser("obsidian-search", help="Search indexed Obsidian notes")
    obsidian_search.add_argument("--query", required=True)
    obsidian_search.add_argument("--limit", type=int, default=10)
    obsidian_search.add_argument("--json", action="store_true")

    prompt_list = subparsers.add_parser("prompt-list", help="List prebuilt prompt templates")
    prompt_list.add_argument("--json", action="store_true")

    prompt_render = subparsers.add_parser("prompt-render", help="Render a prompt template")
    prompt_render.add_argument("--name", required=True)
    prompt_render.add_argument("--var", action="append", default=[])

    materialize = subparsers.add_parser("materialize-module", help="Materialize module proposal into model code")
    materialize.add_argument("--proposal", required=True)
    materialize.add_argument("--output-dir", default="hyperagent/models/generated")
    materialize.add_argument("--base-plan", default=None)
    materialize.add_argument("--ablation-output", default=None)
    materialize.add_argument("--force", action="store_true")

    channel_init = subparsers.add_parser(
        "channel-init",
        help=_txt(
            translator,
            "cli.command.channel_init.help",
            "Initialize Feishu or QQ bot channel config without storing secrets",
        ),
    )
    channel_init.add_argument("--provider", required=True, choices=["feishu", "qq"])

    channel_list = subparsers.add_parser(
        "channel-list",
        help=_txt(
            translator,
            "cli.command.channel_list.help",
            "List configured external bot channels",
        ),
    )
    channel_list.add_argument("--json", action="store_true")

    channel_run = subparsers.add_parser(
        "channel-run",
        help=_txt(
            translator,
            "cli.command.channel_run.help",
            "Run the FastAPI Feishu/QQ bot webhook gateway",
        ),
    )
    channel_run.add_argument("--host", default="0.0.0.0")
    channel_run.add_argument("--port", type=int, default=8765)
    channel_run.add_argument(
        "--reload",
        action="store_true",
        help=_txt(translator, "cli.arg.channel_reload.help", "Enable uvicorn reload mode."),
    )

    channel_test = subparsers.add_parser(
        "channel-test",
        help=_txt(
            translator,
            "cli.command.channel_test.help",
            "Test a channel route with a synthetic text message",
        ),
    )
    channel_test.add_argument("--provider", required=True, choices=["feishu", "qq"])
    channel_test.add_argument("--text", required=True)
    channel_test.add_argument("--chat-id", default="test-chat")
    channel_test.add_argument("--user-id", default="test-user")
    channel_test.add_argument(
        "--send",
        action="store_true",
        help=_txt(
            translator,
            "cli.arg.channel_send.help",
            "Call the configured LLM instead of using dry-run echo.",
        ),
    )
    channel_test.add_argument("--json", action="store_true")

    language_list = subparsers.add_parser(
        "language-list",
        help=_txt(
            translator,
            "cli.command.language_list.help",
            "List installed HyperAgent language packs",
        ),
    )
    language_list.add_argument(
        "--json",
        action="store_true",
        help=_txt(translator, "cli.arg.language_json.help", "Print language information as JSON."),
    )

    language_set = subparsers.add_parser(
        "language-set",
        help=_txt(
            translator,
            "cli.command.language_set.help",
            "Set the workspace default interface language",
        ),
    )
    language_set.add_argument(
        "locale",
        help=_txt(translator, "cli.arg.language_locale.help", "Locale code such as zh-CN or en."),
    )

    language_install = subparsers.add_parser(
        "language-install",
        help=_txt(
            translator,
            "cli.command.language_install.help",
            "Install a local JSON language pack",
        ),
    )
    language_install.add_argument(
        "--path",
        required=True,
        help=_txt(translator, "cli.arg.language_path.help", "Path to a JSON language pack."),
    )

    language_export = subparsers.add_parser(
        "language-export",
        help=_txt(
            translator,
            "cli.command.language_export.help",
            "Export a language pack template",
        ),
    )
    language_export.add_argument(
        "--locale",
        required=True,
        help=_txt(translator, "cli.arg.language_locale.help", "Locale code such as zh-CN or en."),
    )
    language_export.add_argument(
        "--output",
        required=True,
        help=_txt(translator, "cli.arg.language_output.help", "Output path."),
    )

    subparsers.add_parser(
        "hyperagent-commands",
        help="Show Claude-Code-like HyperAgent command aliases",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    global _ACTIVE_TRANSLATOR
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    load_env_file(Path(".env"), override=False)
    workspace = HyperAgentWorkspace()
    i18n_store = I18nStore(workspace.project_root, workspace.workspace_dir)
    translator = i18n_store.translator(i18n_store.resolve_locale(raw_argv))
    args = _build_parser(translator).parse_args(raw_argv)
    translator = i18n_store.translator(args.lang or translator.locale)
    _ACTIVE_TRANSLATOR = translator
    agent = CoordinatorAgent()
    llm_store = LLMProviderStore(workspace.workspace_dir)
    session_store = ConversationStore(workspace.workspace_dir)
    channel_store = ChannelConfigStore(workspace.workspace_dir)
    mcp_store = MCPServerStore(workspace.workspace_dir)
    obsidian_store = ObsidianVaultIndex(workspace.workspace_dir)
    prompt_library = PromptLibrary([PACKAGE_ROOT / "prompts", workspace.workspace_dir / "prompts"])

    if args.command == "init":
        config = workspace.init(
            Path(args.dataset_root),
            output_root=args.output_root,
            reports_root=args.reports_root,
            literature_root=args.literature_root,
            default_provider=args.default_provider,
            default_year_from=args.default_year_from,
        )
        append_worklog(
            "初始化 HyperAgent 工作台",
            "Agent CLI 工作台命令已实现。",
            f"创建 `.hyperagent` 配置，数据集根目录为 {config.dataset_root}。",
            "项目级配置可以让后续任务命令不重复传入数据集根路径。",
            f"配置已写入 {workspace.config_path}。",
            "HyperAgent 工作台已初始化。",
            "下一步可用 task-create 创建科研任务。",
        )
        print(_txt(translator, "cli.output.initialized_workspace", "Initialized HyperAgent workspace: {path}", path=workspace.workspace_dir))
        return 0

    if args.command == "hyperagent-commands":
        print(command_help_text(translator))
        return 0

    if args.command == "language-list":
        packs = i18n_store.list_packs()
        current = i18n_store.resolve_locale(raw_argv)
        if args.json:
            print_json(
                {
                    "current_locale": current,
                    "languages": [
                        {
                            "locale": pack.locale,
                            "source": pack.source,
                            "path": pack.path,
                            "key_count": len(pack.translations),
                        }
                        for pack in packs
                    ],
                }
            )
        else:
            print(_kv(translator, "current_locale", "current_locale", current))
            for pack in packs:
                marker = "*" if pack.locale == current else " "
                print(f"{marker} {pack.locale}\t{pack.source}\t{pack.path}")
        return 0

    if args.command == "language-set":
        path = i18n_store.set_workspace_locale(args.locale)
        print(
            f"{translator.t('cli.output.language_set', default='language set')}: "
            f"{args.locale}"
        )
        print(_kv(translator, "config", "config", path))
        return 0

    if args.command == "language-install":
        pack = i18n_store.install(Path(args.path))
        print(
            f"{translator.t('cli.output.language_installed', default='language installed')}: "
            f"{pack.locale}"
        )
        print(_kv(translator, "path", "path", pack.path))
        return 0

    if args.command == "language-export":
        path = i18n_store.export(args.locale, Path(args.output))
        print(
            f"{translator.t('cli.output.language_exported', default='language exported')}: "
            f"{path}"
        )
        return 0

    if args.command == "channel-init":
        config = channel_store.init_provider(args.provider)
        append_worklog(
            "初始化 Bot 渠道配置",
            "HyperAgent 已具备会话和 LLM provider 运行时。",
            f"初始化 provider={config.provider} 的外部 Bot 渠道配置。",
            "渠道配置只保存环境变量名，不保存真实密钥，避免把平台 token 写入仓库。",
            f"配置已写入 {channel_store.path}，需要的环境变量包括 {', '.join(channel_store.env_summary()[config.provider])}。",
            "Bot 渠道配置已生成，可接入 FastAPI webhook 网关。",
            "下一步在飞书或 QQ 官方平台配置回调 URL，并用 channel-run 启动服务。",
        )
        print(_kv(translator, "provider", "provider", config.provider))
        print(_kv(translator, "config", "config", channel_store.path))
        print(f"{_label(translator, 'env_vars', 'env_vars')}:")
        for name in channel_store.env_summary()[config.provider]:
            print(f"  {name}")
        return 0

    if args.command == "channel-list":
        configs = channel_store.ensure_defaults()
        registry = register_builtin_channel_platforms()
        env_summary = channel_store.env_summary()
        env_configured = channel_store.env_configured_summary()
        payload = {
            "channels": [
                {
                    "provider": item.provider,
                    "enabled": item.enabled,
                    "display_name": item.display_name,
                    "default_llm_provider": item.default_llm_provider,
                    "default_model": item.default_model,
                    "default_mode": item.default_mode,
                    "env_vars": env_summary.get(item.provider, []),
                    "env_configured": env_configured.get(item.provider, {}),
                    "chat_query_only": True,
                }
                for item in configs
            ],
            "platforms": [entry.to_dict() for entry in registry.list()],
        }
        if args.json:
            print_json(payload)
        else:
            for item in payload["channels"]:
                print(
                    f"{item['provider']}\t{_label(translator, 'enabled', 'enabled')}={_value(translator, item['enabled'])}\t"
                    f"llm={item['default_llm_provider']}\tmode={item['default_mode']}"
                )
                print(f"  {_label(translator, 'env', 'env')}: {', '.join(item['env_vars'])}")
                missing = [
                    name for name, configured in item["env_configured"].items() if not configured
                ]
                if missing:
                    print(f"  {_label(translator, 'missing_env', 'missing_env')}: {', '.join(missing)}")
            print(f"{_label(translator, 'platforms', 'platforms')}:")
            for item in payload["platforms"]:
                print(
                    f"  {item['provider']}\tchat_query_only={item['chat_query_only']}\t"
                    f"webhook={item['supports_webhook']}"
                )
        return 0

    if args.command == "channel-run":
        channel_store.ensure_defaults()
        router = ChannelRouter(
            workspace,
            session_store,
            llm_store,
            prompt_library=prompt_library,
            config_store=channel_store,
        )
        app = create_channel_app(router, channel_store)
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is not installed. Install the HyperAgent environment "
                "dependencies or run `python -m pip install uvicorn`."
            ) from exc
        append_worklog(
            "启动 Bot 渠道 FastAPI 网关",
            "Feishu/QQ channel 配置和 FastAPI app factory 已实现。",
            f"启动 host={args.host} port={args.port} 的 Bot webhook 服务。",
            "Bot 网关只允许聊天/查询路径，不接入 shell、训练或写入工具，降低外部平台触发本地操作的风险。",
            "服务将暴露 /health、/channels、/webhooks/feishu 和 /webhooks/qq。",
            "FastAPI 服务开始运行。",
            "下一步在飞书或 QQ 官方 Bot 平台把 webhook URL 指向对应路径。",
        )
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "channel-test":
        config = channel_store.init_provider(args.provider)
        if not args.send:
            config.dry_run = True
            config.send_enabled = False
        inbound = ChannelInboundMessage(
            provider=args.provider,
            channel_user_id=args.user_id,
            chat_id=args.chat_id,
            message_id="test-message",
            text=args.text,
            chat_type="group" if args.provider == "qq" else "p2p",
            metadata={"source": "channel-test"},
        )
        router = ChannelRouter(
            workspace,
            session_store,
            llm_store,
            prompt_library=prompt_library,
            config_store=channel_store,
        )
        result = router.handle_message(
            inbound,
            config,
            dry_run_agent=not args.send,
        )
        if args.json:
            print_json(result.to_dict())
        else:
            print(_kv(translator, "status", "status", result.status))
            print(_kv(translator, "session_id", "session_id", result.session_id))
            if result.outbound:
                print(_kv(translator, "reply", "reply", result.outbound.text))
            for warning in result.warnings:
                print(_warning(translator, warning))
        return 0

    if args.command == "status":
        status = workspace.status()
        if args.json:
            print_json(status.to_dict())
        else:
            print(_kv(translator, "initialized", "initialized", status.initialized))
            print(_kv(translator, "workspace", "workspace", status.workspace_dir))
            print(_kv(translator, "dataset_root", "dataset_root", status.dataset_root))
            print(_kv(translator, "tasks", "tasks", status.task_count))
            print(_kv(translator, "tasks_by_status", "tasks_by_status", status.tasks_by_status))
        return 0

    if args.command == "task-create":
        task = workspace.create_task(
            goal=args.goal,
            dataset=args.dataset,
            objective=args.objective,
            keywords=parse_keywords(args.keywords),
        )
        append_worklog(
            "创建科研任务",
            "HyperAgent 工作台已初始化。",
            f"创建任务 {task.task_id}，数据集为 {task.dataset}。",
            "任务记录把目标、数据集、关键词和 objective 固定下来，后续 artifact 可追踪。",
            f"任务文件已写入 {workspace.task_path(task.task_id)}。",
            "科研任务已创建。",
            "下一步可用 task-run 生成实验计划和自动实验议程。",
        )
        print(task.task_id)
        return 0

    if args.command == "benchmark-list":
        datasets = BenchmarkAgent().list_catalog(Path(args.catalog))
        if args.json:
            print_json({"datasets": datasets})
        else:
            for name, spec in datasets.items():
                print(
                    f"{name}\t{spec.get('local_example', '')}\t"
                    f"{spec.get('source_url', '')}"
                )
        return 0

    if args.command == "benchmark-matrix":
        seeds = parse_int_list(args.seeds)
        dataset_names = parse_keywords(args.datasets) or None
        matrix = BenchmarkAgent().run_matrix(
            Path(args.catalog),
            dataset_names=dataset_names,
            reports_root=Path(args.reports_root),
            experiments_root=Path(args.experiments_root),
            seeds=seeds,
            run_suite=args.run_suite,
        )
        completed = sum(1 for row in matrix["datasets"] if row["status"] == "completed")
        planned = sum(1 for row in matrix["datasets"] if row["status"] == "planned")
        failed = sum(
            1
            for row in matrix["datasets"]
            if row["status"] not in {"completed", "planned"}
        )
        append_worklog(
            "运行 Benchmark 矩阵",
            "数据集目录和多 seed suite runner 已具备。",
            f"基于 {args.catalog} 处理 {len(matrix['datasets'])} 个 benchmark，run_suite={args.run_suite}。",
            "批量 benchmark 矩阵把数据审计、计划和可选多 seed 运行统一落盘，便于比较不同数据集和后续论文表格。",
            f"completed={completed}, planned={planned}, failed={failed}, seeds={seeds}。",
            f"矩阵已写入 {Path(args.reports_root) / 'benchmark_matrix.json'}。",
            "下一步可打开 benchmark_matrix.md，选择低分或失败数据集进入自动实验闭环。",
        )
        print(_kv(translator, "matrix", "matrix", Path(args.reports_root) / "benchmark_matrix.json"))
        print(_kv(translator, "report", "report", Path(args.reports_root) / "benchmark_matrix.md"))
        print(_kv(translator, "completed", "completed", completed))
        print(_kv(translator, "planned", "planned", planned))
        print(_kv(translator, "failed", "failed", failed))
        return 0

    if args.command == "task-list":
        tasks = workspace.list_tasks()
        if args.json:
            print_json({"tasks": [task.to_dict() for task in tasks]})
        else:
            for task in tasks:
                print(f"{task.task_id}\t{task.status}\t{task.dataset}\t{task.goal}")
        return 0

    if args.command == "task-show":
        task = workspace.load_task(args.task_id)
        if args.json:
            print_json(task.to_dict())
        else:
            print(_kv(translator, "task_id", "task_id", task.task_id))
            print(_kv(translator, "status", "status", task.status))
            print(_kv(translator, "dataset", "dataset", task.dataset))
            print(_kv(translator, "objective", "objective", task.objective))
            print(_kv(translator, "goal", "goal", task.goal))
            print(_kv(translator, "keywords", "keywords", ", ".join(task.keywords)))
            print(_kv(translator, "artifacts", "artifacts", task.artifacts))
        return 0

    if args.command == "task-run":
        task = workspace.load_task(args.task_id)
        config = workspace.load_config()
        artifact_dir = workspace.task_artifact_dir(task.task_id)
        dataset_path = workspace.resolve_dataset_path(task.dataset)
        audit = agent.audit(dataset_path, artifact_dir / "audit.json")
        spectral_report = agent.analyze(audit, artifact_dir / "spectral_report.json")
        recommendation = agent.recommend(
            audit,
            spectral_report,
            artifact_dir / "model_recommendation.json",
        )
        plan = agent.plan(
            audit,
            spectral_report,
            recommendation,
            artifact_dir / "experiment.yaml",
            Path(config.output_root) / task.task_id,
            args.seed,
        )
        agenda = agent.design_auto_experiments(
            audit,
            spectral_report,
            recommendation,
            objective=task.objective,
        )
        write_json(artifact_dir / "auto_experiment_agenda.json", agenda)
        task.artifacts.update(
            {
                "audit": str(artifact_dir / "audit.json"),
                "spectral_report": str(artifact_dir / "spectral_report.json"),
                "model_recommendation": str(artifact_dir / "model_recommendation.json"),
                "experiment_plan": str(artifact_dir / "experiment.yaml"),
                "auto_experiment_agenda": str(artifact_dir / "auto_experiment_agenda.json"),
            }
        )

        if args.with_literature:
            query = " ".join(task.keywords) if task.keywords else task.goal
            literature = agent.search_literature(
                query,
                artifact_dir / "literature.json",
                provider_name=args.provider or config.default_provider,
                max_results=args.max_literature_results,
                year_from=config.default_year_from,
            )
            task.artifacts["literature"] = str(artifact_dir / "literature.json")
            if literature.warnings:
                task.notes.extend(literature.warnings)

        if args.run_baseline:
            result = agent.run(plan)
            report_path = agent.write_report(
                result,
                Path(result.experiment_dir) / "report.md",
            )
            task.artifacts["result"] = str(Path(result.experiment_dir) / "result.json")
            task.artifacts["report"] = str(report_path)

        task.status = "completed"
        workspace.save_task(task)
        append_worklog(
            "执行科研任务工作流",
            "科研任务已创建。",
            f"执行任务 {task.task_id}，数据集路径为 {dataset_path}。",
            "task-run 将目标转化为审计、光谱诊断、模型推荐、实验计划和自动实验议程，形成可追踪 artifact。",
            f"任务 artifact 已写入 {artifact_dir}。",
            "任务规划工作流已完成。",
            "下一步可查看 task-show 或选择候选实验运行 baseline。",
        )
        print(_txt(translator, "cli.output.task_run_complete", "Task run complete: {task_id}", task_id=task.task_id))
        return 0

    if args.command == "audit":
        audit = agent.audit(Path(args.data_root), Path(args.output), args.reader)
        append_worklog(
            "执行数据审计命令",
            "CLI 已初始化。",
            f"审计数据集 {args.data_root} 并写入 {args.output}。",
            "数据审计是后续光谱分析和实验规划的输入。",
            f"调用 CoordinatorAgent.audit，得到 {audit.band_count} 个波段和 {audit.class_count} 个类别。",
            "审计 JSON 已生成。",
            "下一步可运行 plan 命令生成实验配置。",
        )
        print(_txt(translator, "cli.output.wrote_audit", "Wrote audit: {path}", path=args.output))
        return 0

    if args.command == "llm-providers":
        providers = llm_store.ensure_defaults()
        if args.json:
            print_json(
                {
                    "providers": [
                        {
                            **provider.to_dict(),
                            "api_key_configured": bool(os.environ.get(provider.api_key_env)),
                        }
                        for provider in providers
                    ]
                }
            )
        else:
            for provider in providers:
                configured = _configured(translator, bool(os.environ.get(provider.api_key_env)))
                print(
                    f"{provider.name}\t{provider.kind}\t{provider.default_model}\t"
                    f"{provider.api_key_env}\t{configured}"
                )
        return 0

    if args.command == "llm-profile":
        if args.profile:
            profile = get_reasonix_profile(args.profile)
            data = profile.to_dict() if profile else {}
        else:
            data = {
                "profiles": [profile.to_dict() for profile in list_reasonix_profiles()],
                "cache_guidance": reasonix_cache_guidance(),
            }
        if args.json:
            print_json(data)
        else:
            if args.profile:
                print_profile(data, translator)
            else:
                for profile in data["profiles"]:
                    print_profile(profile, translator)
                print(f"{_label(translator, 'cache_rule', 'cache rule')}: {data['cache_guidance']['rule']}")
        return 0

    if args.command == "llm-usage":
        summary = LLMUsageLedger(workspace.workspace_dir).summarize(limit=args.limit)
        if args.json:
            print_json(summary)
        else:
            print(_kv(translator, "requests", "requests", summary["request_count"]))
            print(_kv(translator, "total_tokens", "total_tokens", summary["total_tokens"]))
            print(_kv(translator, "prompt_tokens", "prompt_tokens", summary["prompt_tokens"]))
            print(_kv(translator, "completion_tokens", "completion_tokens", summary["completion_tokens"]))
            print(_kv(translator, "prompt_cache_hit_tokens", "prompt_cache_hit_tokens", summary["prompt_cache_hit_tokens"]))
            print(_kv(translator, "prompt_cache_miss_tokens", "prompt_cache_miss_tokens", summary["prompt_cache_miss_tokens"]))
            print(_kv(translator, "cache_hit_ratio", "cache_hit_ratio", summary["cache_hit_ratio"]))
            print(_kv(translator, "cost_estimate_usd", "cost_estimate_usd", summary["cost_estimate_usd"]))
            print(_kv(translator, "ledger", "ledger", summary["ledger_path"]))
        return 0

    if args.command == "web":
        if args.web_command == "status":
            payload = web_status()
            if args.json:
                print_json(payload)
            else:
                print(_kv(translator, "search_configured", "search_configured", payload["search_configured"]))
                for name, configured in payload["providers"].items():
                    print(f"{name}: {_configured(translator, bool(configured))}")
                print(_kv(translator, "fetch_available", "fetch_available", True))
            return 0
        executor = SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
            permission_policy=getattr(args, "permission", "session-ask"),
            permission_callback=(
                confirm_tool_permission
                if getattr(args, "permission", "") in {"ask", "session-ask"}
                else None
            ),
            hook_engine=HookEngine(workspace.workspace_dir),
        )
        if args.web_command == "search":
            result = executor.web_search(
                args.query,
                provider=args.provider,
                max_results=args.max_results,
                timeout_sec=args.timeout_sec,
            )
        elif args.web_command == "fetch":
            result = executor.web_fetch(
                args.url,
                max_chars=args.max_chars,
                timeout_sec=args.timeout_sec,
            )
        elif args.web_command == "cite":
            result = executor.web_cite(args.citation_id, limit=args.limit)
        else:
            raise ValueError(f"Unsupported web command: {args.web_command}")
        if args.json:
            print_json(result.to_dict())
        else:
            print(_kv(translator, "tool", "tool", result.tool_name))
            print(_kv(translator, "status", "status", result.status))
            print(_kv(translator, "artifact", "artifact", result.artifact_path))
            for warning in result.warnings:
                print(_warning(translator, warning))
            if result.content:
                print(result.content)
        return 0

    if args.command == "image":
        if args.image_command == "status":
            payload = image_status()
            if args.json:
                print_json(payload)
            else:
                print(_kv(translator, "provider", "provider", payload["provider"]))
                print(_kv(translator, "required_env", "required_env", payload["required_env"]))
                print(_kv(translator, "configured", "configured", payload["configured"]))
                print(_kv(translator, "output_root", "output_root", payload["output_root"]))
            return 0
        executor = SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
            permission_policy=args.permission,
            permission_callback=(
                confirm_tool_permission
                if args.permission in {"ask", "session-ask"}
                else None
            ),
            hook_engine=HookEngine(workspace.workspace_dir),
        )
        if args.image_command == "generate":
            result = executor.image_generate(args.prompt)
        elif args.image_command == "edit":
            result = executor.image_edit(args.image, args.instruction)
        else:
            raise ValueError(f"Unsupported image command: {args.image_command}")
        if args.json:
            print_json(result.to_dict())
        else:
            print(_kv(translator, "tool", "tool", result.tool_name))
            print(_kv(translator, "status", "status", result.status))
            print(_kv(translator, "artifact", "artifact", result.artifact_path))
            for warning in result.warnings:
                print(_warning(translator, warning))
            if result.content:
                print(result.content)
        return 0

    if args.command == "ide-context":
        store = IDEContextStore(workspace.workspace_dir)
        if args.action == "status":
            payload = store.load()
        elif args.action == "on":
            payload = store.set_enabled(True)
        elif args.action == "off":
            payload = store.set_enabled(False)
        elif args.action == "set-open-files":
            payload = store.set_open_files(args.paths)
        elif args.action == "clear":
            payload = store.clear()
        else:
            raise ValueError(f"Unsupported ide-context action: {args.action}")
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "enabled", "enabled", payload.get("enabled")))
            print(_kv(translator, "open_files", "open_files", ", ".join(payload.get("open_files", [])) or None))
            print(_kv(translator, "updated_at", "updated_at", payload.get("updated_at", "")))
        return 0

    if args.command == "plan-mode":
        store = PlanModeStore(workspace.workspace_dir)
        if args.action == "status":
            payload = store.load()
        else:
            payload = store.set_enabled(args.action == "on", " ".join(args.reason))
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "enabled", "enabled", payload.get("enabled")))
            if payload.get("reason"):
                print(_kv(translator, "reason", "reason", payload.get("reason")))
            print(_kv(translator, "updated_at", "updated_at", payload.get("updated_at", "")))
        return 0

    if args.command == "personality":
        store = PersonalityStore(workspace.workspace_dir)
        if args.action == "status":
            payload = store.load()
        elif args.action == "set":
            payload = store.set(" ".join(args.text))
        elif args.action == "clear":
            payload = store.clear()
        else:
            raise ValueError(f"Unsupported personality action: {args.action}")
        if args.json:
            print_json(payload)
        else:
            print(payload.get("text") or _txt(translator, "cli.output.no_personality_note", "no personality note"))
        return 0

    if args.command == "feedback":
        store = FeedbackStore(workspace.workspace_dir)
        if args.action == "add":
            payload = store.add(" ".join(args.text), source="cli")
            if args.json:
                print_json(payload)
            else:
                print(_txt(translator, "cli.output.feedback_recorded", "feedback recorded"))
            return 0
        items = store.list(limit=args.limit)
        if args.json:
            print_json({"feedback": items})
        else:
            if not items:
                print(_txt(translator, "cli.output.no_feedback", "no feedback"))
            for item in items:
                print(f"{item.get('created_at', '')}\t{item.get('text', '')}")
        return 0

    if args.command == "worktree":
        payload = worktree_status(workspace.project_root)
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "branch", "branch", payload["branch"]))
            print(_kv(translator, "head", "head", payload["head"]))
            print(f"{_label(translator, 'dirty_files', 'dirty_files')}:")
            for item in payload["dirty_files"] or [_txt(translator, "cli.value.none", "none")]:
                print(f"  {item}")
            print(f"{_label(translator, 'recent_commits', 'recent_commits')}:")
            for item in payload["recent_commits"]:
                print(f"  {item}")
        return 0

    if args.command == "llm-dry-run":
        llm_store.ensure_defaults()
        spec = llm_store.get(args.provider)
        payload = LLMRequestBuilder().build(
            spec,
            [
                LLMMessage(role="system", content=args.system),
                LLMMessage(role="user", content=args.user),
            ],
            model=resolve_llm_model(args),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **build_llm_runtime_kwargs(args),
        )
        print_json(payload)
        return 0

    if args.command == "llm-send":
        llm_store.ensure_defaults()
        spec = llm_store.get(args.provider)
        response = LLMClient().send(
            spec,
            [
                LLMMessage(role="system", content=args.system),
                LLMMessage(role="user", content=args.user),
            ],
            model=resolve_llm_model(args),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **build_llm_runtime_kwargs(args),
        )
        LLMUsageLedger(workspace.workspace_dir).record_response(
            response,
            spec=spec,
            event_type="llm_send.response",
            context_chars=len(args.system) + len(args.user),
            metadata={"command": "llm-send"},
        )
        if args.output:
            write_json(Path(args.output), response)
        if response.warnings:
            for warning in response.warnings:
                print(_warning(translator, warning))
        else:
            if response.reasoning_content:
                print(f"{_label(translator, 'reasoning_content', 'reasoning_content')}:")
                print(response.reasoning_content)
            if response.tool_calls:
                print(f"{_label(translator, 'tool_calls', 'tool_calls')}:")
                print_json({"tool_calls": response.tool_calls})
            print(response.content)
        return 0

    if args.command == "agent-chat":
        llm_store.ensure_defaults()
        if args.session_id:
            session_id = args.session_id
            session_store.load(session_id)
        else:
            title = args.new_title or args.message.strip().splitlines()[0][:80]
            session = session_store.new(title or "HyperAgent session")
            session_id = session.session_id
        result = AgentLoop(
            session_store,
            llm_store,
            workspace,
            prompt_library=prompt_library,
        ).run(
            session_id=session_id,
            provider=args.provider,
            user_message=args.message,
            model=resolve_llm_model(args),
            mode=args.mode,
            task_id=args.task_id,
            auto_compress=not args.no_auto_compress,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            **build_llm_runtime_kwargs(args),
            output_path=Path(args.output) if args.output else None,
        )
        append_worklog(
            "运行持续对话 Agent Loop",
            "会话、LLM provider、任务 artifact 和 prompt library 已具备。",
            f"执行 session={result.session_id} provider={result.provider} mode={result.mode} 的一轮 agent-chat。",
            "持续对话能力是后续自动写代码、做实验和设计算法的统一入口。",
            f"上下文消息数 {result.context_message_count}，上下文字数 {result.context_chars}，输出文件 {result.output_path or 'none'}。",
            "用户消息和模型回复已写回会话；LLM 调用结果已结构化返回。",
            "下一步可在同一 session 继续对话，或绑定 task-id 让回答读取实验 artifact。",
        )
        print(_kv(translator, "session_id", "session_id", result.session_id))
        if result.warnings:
            for warning in result.warnings:
                print(_warning(translator, warning))
        if result.response.content:
            print(result.response.content)
        return 0

    if args.command == "run":
        if PlanModeStore(workspace.workspace_dir).load().get("enabled"):
            print(_txt(translator, "cli.output.plan_mode_run_paused", "plan-mode is enabled; headless action-loop tool execution is paused."))
            print(_txt(translator, "cli.output.plan_mode_resume_hint", "Use `HyperAgent plan-mode off` to resume tool execution, or use `HyperAgent plan ...` for planning."))
            return 0
        llm_store.ensure_defaults()
        instruction = " ".join(args.instruction).strip()
        if args.session_id:
            session_id = args.session_id
            session_store.load(session_id)
        else:
            title = args.new_title or instruction.splitlines()[0][:80]
            session_id = session_store.new(title or "HyperAgent run").session_id
        run = AgentActionLoop(
            session_store,
            llm_store,
            workspace,
            permission_policy=args.permission,
            permission_callback=(
                confirm_tool_permission
                if args.permission in {"ask", "session-ask"}
                else None
            ),
        ).run(
            session_id=session_id,
            provider=args.provider,
            instruction=instruction,
            model=resolve_llm_model(args),
            task_id=args.task_id,
            max_steps=args.max_steps,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            loop_mode=args.loop_mode,
            token_budget=args.token_budget,
            **build_llm_runtime_kwargs(args),
        )
        append_worklog(
            "运行 Reasonix 风格 Headless Run",
            "ActionLoop、事件日志和 cache-first metadata 已接入。",
            f"执行 run={run.run_id} session={run.session_id} loop_mode={run.loop_mode} steps={len(run.steps)}。",
            "headless run 是脚本化 agent 操作入口，便于后续 CI、实验自动化和 channel 安全隔离复用。",
            f"artifact={Path(run.run_dir) / 'action_run.json'}，event_log={run.event_log_path}。",
            f"run 状态为 {run.status}。",
            "下一步可用 events/replay/stats 查看运行轨迹，或继续同一 session。",
        )
        print(_kv(translator, "run_id", "run_id", run.run_id))
        print(_kv(translator, "session_id", "session_id", run.session_id))
        print(_kv(translator, "status", "status", run.status))
        print(_kv(translator, "action_run", "action_run", Path(run.run_dir) / "action_run.json"))
        if run.stable_prefix_hash:
            print(_kv(translator, "stable_prefix_hash", "stable_prefix_hash", run.stable_prefix_hash))
        if run.event_log_path:
            print(_kv(translator, "event_log", "event_log", run.event_log_path))
        if run.final_response:
            print(run.final_response)
        for warning in run.warnings:
            print(_warning(translator, warning))
        return 0

    if args.command == "agent-context":
        builder = RepoContextBuilder(workspace.project_root)
        snapshot = builder.build(
            query=args.query,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
        )
        if args.format == "json":
            if args.output:
                write_json(Path(args.output), snapshot)
                print(_txt(translator, "cli.output.wrote_repo_context", "Wrote repo context: {path}", path=args.output))
            else:
                print_json(snapshot.to_dict())
        else:
            markdown = builder.to_markdown(snapshot)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(markdown, encoding="utf-8")
                print(_txt(translator, "cli.output.wrote_repo_context", "Wrote repo context: {path}", path=args.output))
            else:
                print(markdown)
        return 0

    if args.command == "repl":
        llm_store.ensure_defaults()
        HyperAgentRepl(
            workspace=workspace,
            conversations=session_store,
            providers=llm_store,
            prompt_library=prompt_library,
            translator=translator,
            provider=args.provider,
            model=resolve_llm_model(args),
            mode=args.mode,
            task_id=args.task_id,
            session_id=args.session_id,
            new_title=args.new_title,
            permission_policy=args.permission,
            max_context_chars=args.max_context_chars,
            keep_last=args.keep_last,
            llm_kwargs=build_llm_runtime_kwargs(args),
        ).run()
        append_worklog(
            "运行交互式 HyperAgent REPL",
            "HyperAgent launcher 已实现。",
            f"启动 provider={args.provider} mode={args.mode} permission={args.permission} 的 REPL。",
            "交互式 REPL 是持续对话、工具确认、上下文压缩和本地行动闭环的统一入口。",
            "REPL 已退出。",
            "交互式会话已保存到 .hyperagent/sessions。",
            "下一步可继续同一 session，或使用 /context 与 /compact 管理上下文。",
        )
        return 0

    if args.command == "tui":
        llm_store.ensure_defaults()
        code = HyperAgentTui(
            workspace=workspace,
            conversations=session_store,
            providers=llm_store,
            prompt_library=prompt_library,
            translator=translator,
            provider=args.provider,
            model=resolve_llm_model(args),
            mode=args.mode,
            task_id=args.task_id,
            session_id=args.session_id,
            new_title=args.new_title,
            permission_policy=args.permission,
            max_context_chars=args.max_context_chars,
            keep_last=args.keep_last,
            llm_kwargs=build_llm_runtime_kwargs(args),
        ).run()
        append_worklog(
            "运行全屏 HyperAgent TUI",
            "REPL 命令处理、会话和工具权限已经实现。",
            f"启动 provider={args.provider} mode={args.mode} permission={args.permission} 的 curses TUI。",
            "TUI 只负责全屏输入输出，业务流程复用 REPL，避免交互界面和 agent 逻辑耦合。",
            "TUI 已退出。",
            "交互式会话和工具记录已按原 runtime 规则保存。",
            "下一步可继续完善工具面板和实验故障暂停提示。",
        )
        return code

    if args.command == "agent-plan":
        llm_store.ensure_defaults()
        if args.session_id:
            session_id = args.session_id
            session_store.load(session_id)
        else:
            title = args.new_title or args.instruction.strip().splitlines()[0][:80]
            session = session_store.new(title or "HyperAgent coding run")
            session_id = session.session_id
        run = CodingAgent(
            workspace,
            session_store,
            llm_store,
            prompt_library=prompt_library,
        ).plan(
            session_id=session_id,
            provider=args.provider,
            instruction=args.instruction,
            model=resolve_llm_model(args),
            mode=args.mode,
            task_id=args.task_id,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            **build_llm_runtime_kwargs(args),
        )
        append_worklog(
            "运行 Claude Code 式 Agent Plan",
            "仓库上下文采集器和 coding-agent run 管理已实现。",
            f"执行 run={run.run_id} session={run.session_id} mode={run.mode} 的 agent-plan。",
            "agent-plan 将仓库快照、会话历史和任务 artifact 组合，生成可归档的代码/实验/算法计划。",
            f"repo context={run.repo_context_markdown_path}，plan={run.plan_path}。",
            f"run 状态为 {run.status}。",
            "下一步可依据 plan 进入受控补丁生成、实验执行或继续同一 session 追问。",
        )
        print(_kv(translator, "run_id", "run_id", run.run_id))
        print(_kv(translator, "session_id", "session_id", run.session_id))
        print(_kv(translator, "plan", "plan", run.plan_path))
        if run.warnings:
            for warning in run.warnings:
                print(_warning(translator, warning))
        return 0

    if args.command == "agent-status":
        registry = SubagentRuntimeRegistry(workspace.workspace_dir)
        states = registry.list(include_completed=True)
        payload = {
            "control": registry.control(),
            "subagents": [state.to_dict() for state in states],
        }
        if args.json:
            print_json(payload)
        else:
            control = payload["control"]
            print(_kv(translator, "paused", "paused", control.get("paused", False)))
            stop_ids = control.get("stop_ids", [])
            if stop_ids:
                print(_kv(translator, "stop_ids", "stop_ids", ", ".join(stop_ids)))
            for state in states[-20:]:
                print(
                    f"{state.subagent_id}\t{state.status}\t{state.agent_name}\t"
                    f"{_label(translator, 'role', 'role')}={state.role}\t"
                    f"{_label(translator, 'depth', 'depth')}={state.depth}"
                )
        return 0

    if args.command == "agent-pause":
        registry = SubagentRuntimeRegistry(workspace.workspace_dir)
        reason = args.reason or " ".join(args.reason_text)
        registry.pause(reason)
        print(_txt(translator, "cli.output.subagent_spawning_paused", "subagent spawning paused"))
        return 0

    if args.command == "agent-resume":
        registry = SubagentRuntimeRegistry(workspace.workspace_dir)
        registry.resume()
        print(_txt(translator, "cli.output.subagent_spawning_resumed", "subagent spawning resumed"))
        return 0

    if args.command == "agent-stop":
        registry = SubagentRuntimeRegistry(workspace.workspace_dir)
        registry.stop(args.subagent_id)
        print(_txt(translator, "cli.output.stop_requested", "stop requested: {target}", target=args.subagent_id))
        return 0

    if args.command == "agent-run":
        if PlanModeStore(workspace.workspace_dir).load().get("enabled"):
            print(_txt(translator, "cli.output.plan_mode_agent_run_paused", "plan-mode is enabled; agent-run tool execution is paused."))
            print(_txt(translator, "cli.output.plan_mode_off_hint", "Use `HyperAgent plan-mode off` to resume tool execution."))
            return 0
        llm_store.ensure_defaults()
        session_id = args.session_id
        if not session_id and args.new_title:
            session_id = session_store.new(args.new_title).session_id
        run = GeneralAgentRunner(
            workspace,
            session_store,
            llm_store,
            permission_policy=args.permission,
            permission_callback=(
                confirm_tool_permission
                if args.permission in {"ask", "session-ask"}
                else None
            ),
        ).run(
            args.agent,
            args.instruction,
            session_id=session_id,
            provider=args.provider,
            model=resolve_llm_model(args),
            profile=args.reasonix_profile,
            task_id=args.task_id,
            max_steps=args.max_steps,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
            loop_mode=args.loop_mode,
            token_budget=args.token_budget,
            llm_kwargs=build_llm_runtime_kwargs(args),
        )
        append_worklog(
            "运行通用 SubAgent",
            "SubAgent 注册表、ActionLoop 和受控工具执行器已具备。",
            f"执行 agent={run.agent_name} role={run.role} run={run.run_id}。",
            "通用 Agent 通过人工触发和会话授权执行 shell/训练工具，避免无约束后台调度。",
            f"agent_run={Path(run.run_dir) / 'agent_run.json'}，action_run={run.action_run_path or 'none'}。",
            f"run 状态为 {run.status}。",
            "下一步可查看 agent_run.json 和工具 artifact，或在同一 session 继续执行。",
        )
        print(_kv(translator, "run_id", "run_id", run.run_id))
        print(_kv(translator, "status", "status", run.status))
        print(_kv(translator, "agent_run", "agent_run", Path(run.run_dir) / "agent_run.json"))
        print(_kv(translator, "session_id", "session_id", run.session_id))
        if run.action_run_path:
            print(_kv(translator, "action_run", "action_run", run.action_run_path))
        for artifact in run.tool_artifacts:
            print(_kv(translator, "artifact", "artifact", artifact))
        for warning in run.warnings:
            print(_warning(translator, warning))
        return 0

    if args.command == "agent-act":
        if PlanModeStore(workspace.workspace_dir).load().get("enabled"):
            print(_txt(translator, "cli.output.plan_mode_agent_act_paused", "plan-mode is enabled; agent-act tool execution is paused."))
            print(_txt(translator, "cli.output.plan_mode_resume_hint", "Use `HyperAgent plan-mode off` to resume tool execution, or use `HyperAgent plan ...` for planning."))
            return 0
        llm_store.ensure_defaults()
        if args.session_id:
            session_id = args.session_id
            session_store.load(session_id)
        else:
            title = args.new_title or args.message.strip().splitlines()[0][:80]
            session = session_store.new(title or "HyperAgent action loop")
            session_id = session.session_id
        run = AgentActionLoop(
            session_store,
            llm_store,
            workspace,
            permission_policy=args.permission,
            permission_callback=(
                confirm_tool_permission
                if args.permission in {"ask", "session-ask"}
                else None
            ),
        ).run(
            session_id=session_id,
            provider=args.provider,
            instruction=args.message,
            model=resolve_llm_model(args),
            task_id=args.task_id,
            max_steps=args.max_steps,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            loop_mode=args.loop_mode,
            token_budget=args.token_budget,
            **build_llm_runtime_kwargs(args),
        )
        append_worklog(
            "运行 LLM 受控 Action Loop",
            "会话、LLM provider、repo context 和受控工具执行器已具备。",
            f"执行 run={run.run_id} session={run.session_id} provider={run.provider} steps={len(run.steps)}。",
            "agent-act 让大模型用结构化 JSON 选择安全工具，并把工具结果写回会话，形成 Claude Code 式闭环。",
            f"run 状态为 {run.status}，artifact={Path(run.run_dir) / 'action_run.json'}。",
            "action loop 结果已归档，工具调用也有独立审计记录。",
            "下一步可继续同一 session，或让 agent 读取 benchmark/suite 结果决定实验动作。",
        )
        print(_kv(translator, "run_id", "run_id", run.run_id))
        print(_kv(translator, "session_id", "session_id", run.session_id))
        print(_kv(translator, "status", "status", run.status))
        print(_kv(translator, "action_run", "action_run", Path(run.run_dir) / "action_run.json"))
        if run.final_response:
            print(run.final_response)
        for warning in run.warnings:
            print(_warning(translator, warning))
        return 0

    if args.command == "agent-tool":
        executor = SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
            permission_policy=args.permission,
            permission_callback=(
                confirm_tool_permission
                if args.permission in {"ask", "session-ask"}
                else None
            ),
        )
        if args.tool_command == "read-file":
            result = executor.read_file(
                args.path,
                start_line=args.start_line,
                max_lines=args.max_lines,
                run_id=args.run_id,
            )
        elif args.tool_command == "search-code":
            result = executor.search_code(
                args.query,
                path=args.path,
                max_results=args.max_results,
                run_id=args.run_id,
            )
        elif args.tool_command == "run-command":
            argv = [item for item in args.argv if item != "--"]
            result = executor.run_command(
                argv,
                timeout_sec=args.timeout_sec,
                run_id=args.run_id,
            )
        elif args.tool_command == "run-experiment":
            result = executor.run_experiment(
                args.plan,
                seeds=parse_int_list(args.seeds) if args.seeds else None,
                output_dir=args.output_dir,
                suite_name=args.suite_name,
                run_id=args.run_id,
            )
        elif args.tool_command == "check-patch":
            patch_text = Path(args.patch_file).read_text(encoding="utf-8")
            result = executor.check_patch(patch_text, run_id=args.run_id)
        elif args.tool_command == "apply-patch":
            patch_text = Path(args.patch_file).read_text(encoding="utf-8")
            result = executor.apply_patch(patch_text, run_id=args.run_id)
        elif args.tool_command == "todo-write":
            result = executor.todo_write(
                [{"content": item, "status": "pending"} for item in args.item],
                owner=args.owner,
                run_id=args.run_id,
            )
        else:
            raise ValueError(f"Unsupported agent tool: {args.tool_command}")
        append_worklog(
            "运行 Claude Code 式受控工具",
            "受控工具执行器已实现。",
            f"执行 tool={result.tool_name} status={result.status}。",
            "工具调用必须有审计记录，方便后续 agent 回放和复现实验/代码修改过程。",
            f"工具结果已写入 {result.artifact_path}。",
            f"工具执行状态为 {result.status}。",
            "下一步可由 agent 读取该工具结果，继续计划、补丁或测试流程。",
        )
        if args.json:
            print_json(result.to_dict())
        else:
            print(_kv(translator, "tool", "tool", result.tool_name))
            print(_kv(translator, "status", "status", result.status))
            print(_kv(translator, "artifact", "artifact", result.artifact_path))
            if result.exit_code is not None:
                print(_kv(translator, "exit_code", "exit_code", result.exit_code))
            for warning in result.warnings:
                print(_warning(translator, warning))
            if result.content:
                print(result.content)
        return 0

    if args.command == "command-list":
        commands = SlashCommandStore(workspace.project_root, workspace.workspace_dir).discover(
            include_hidden=args.include_hidden
        )
        if args.json:
            print_json({"commands": [command.to_dict() for command in commands]})
        else:
            for command in commands:
                tools = ",".join(command.allowed_tools)
                hint = f" {command.argument_hint}" if command.argument_hint else ""
                print(f"/{command.name}{hint}\t{command.source}\t{tools}\t{command.description}")
        return 0

    if args.command == "command-render":
        executor = SafeAgentToolExecutor(
            workspace.project_root,
            workspace.workspace_dir,
            permission_policy="auto",
            hook_engine=HookEngine(workspace.workspace_dir),
        )
        rendered = SlashCommandStore(workspace.project_root, workspace.workspace_dir).render(
            args.name,
            args.arguments,
            expand_shell=args.expand_shell,
            executor=executor,
        )
        payload = {
            "command": rendered.spec.to_dict(),
            "arguments": rendered.arguments,
            "prompt": rendered.prompt,
            "warnings": rendered.warnings,
        }
        if args.json:
            print_json(payload)
        else:
            for warning in rendered.warnings:
                print(_warning(translator, warning))
            print(rendered.prompt)
        return 0

    if args.command == "todos":
        store = TodoStore(workspace.workspace_dir)
        if args.clear:
            todo_list = store.clear(args.owner)
        else:
            todo_list = store.load(args.owner)
        if args.export:
            path = store.export_markdown(args.owner, Path(args.export))
            if not args.json:
                print(_kv(translator, "exported", "exported", path))
        if args.json:
            print_json(todo_list.to_dict())
        else:
            if not todo_list.items:
                print(_txt(translator, "cli.output.no_todos", "no todos"))
            for item in todo_list.items:
                print(f"{item.id}\t{item.status}\t{item.priority}\t{item.content}")
        return 0

    if args.command == "doctor":
        status = workspace.status()
        payload = {
            "initialized": status.initialized,
            "workspace": status.workspace_dir,
            "dataset_root": status.dataset_root,
            "providers": [
                {
                    "name": provider.name,
                    "model": provider.default_model,
                    "api_key_env": provider.api_key_env,
                    "api_key_configured": bool(os.environ.get(provider.api_key_env)),
                }
                for provider in llm_store.ensure_defaults()
            ],
            "commands": len(SlashCommandStore(workspace.project_root, workspace.workspace_dir).discover()),
            "subagents": len(RuntimeExtensionStore(workspace.workspace_dir).list_subagents()),
            "hooks": len(HookEngine(workspace.workspace_dir).list_rules()),
        }
        if args.json:
            print_json(payload)
        else:
            print(_txt(translator, "cli.output.doctor_title", "HyperAgent doctor:"))
            for key, value in payload.items():
                print(f"- {_label(translator, key, key)}: {_value(translator, value)}")
        return 0

    if args.command == "events":
        log = RuntimeEventLog(workspace.workspace_dir)
        events = log.records(
            limit=args.limit,
            event_type=args.type,
            session_id=args.session_id,
            run_id=args.run_id,
        )
        if args.json:
            print_json({"events": [event.to_dict() for event in events], "summary": log.summarize(events)})
        else:
            for event in events:
                print(
                    f"{event.timestamp}\t{event.event_type}\t{event.status}\t"
                    f"{_label(translator, 'session_id', 'session')}={event.session_id or ''}\t"
                    f"{_label(translator, 'run_id', 'run')}={event.run_id or ''}\t"
                    f"{_label(translator, 'tool', 'tool')}={event.tool_name or ''}\t{event.message}"
                )
        return 0

    if args.command == "replay":
        log = RuntimeEventLog(workspace.workspace_dir)
        events = log.records(
            limit=args.limit,
            session_id=args.session_id,
            run_id=args.run_id,
        )
        if args.json:
            print_json({"events": [event.to_dict() for event in events]})
        else:
            for index, event in enumerate(events, start=1):
                subject = event.tool_name or event.source
                print(f"{index}. {event.timestamp} {event.event_type} [{event.status}] {subject}")
                if event.message:
                    print(f"   {event.message}")
        return 0

    if args.command == "diff":
        left = Path(args.left)
        right = Path(args.right)
        left_text = left.read_text(encoding="utf-8", errors="replace").splitlines()
        right_text = right.read_text(encoding="utf-8", errors="replace").splitlines()
        diff_lines = list(
            difflib.unified_diff(
                left_text,
                right_text,
                fromfile=str(left),
                tofile=str(right),
                lineterm="",
                n=args.context,
            )
        )
        if args.json:
            print_json({"left": str(left), "right": str(right), "diff": diff_lines})
        else:
            print("\n".join(diff_lines))
        return 0

    if args.command == "stats":
        event_log = RuntimeEventLog(workspace.workspace_dir)
        payload = {
            "events": event_log.summarize(),
            "llm_usage": LLMUsageLedger(workspace.workspace_dir).summarize(),
            "tools": tool_catalog(),
        }
        if args.json:
            print_json(payload)
        else:
            print(_txt(translator, "cli.output.runtime_stats_title", "Runtime stats:"))
            print(f"- {_label(translator, 'events', 'events')}: {payload['events']['event_count']}")
            print(f"- {_label(translator, 'llm_requests', 'llm_requests')}: {payload['llm_usage']['request_count']}")
            print(f"- {_label(translator, 'llm_total_tokens', 'llm_total_tokens')}: {payload['llm_usage']['total_tokens']}")
            print(f"- {_label(translator, 'cache_hit_ratio', 'cache_hit_ratio')}: {payload['llm_usage']['cache_hit_ratio']}")
            print(f"- {_label(translator, 'tools', 'tools')}: {len(payload['tools'])}")
        return 0

    if args.command == "prune-sessions":
        sessions = session_store.list(include_archived=True)
        archived = [session for session in sessions if session.status == "archived"]
        payload = {
            "archived_sessions": [session.to_dict() for session in archived],
            "dry_run": args.dry_run,
            "deleted": [],
        }
        if not args.dry_run:
            for session in archived:
                session_store.delete(session.session_id, hard=True)
                payload["deleted"].append(session.session_id)
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "archived", "archived", len(archived)))
            print(_kv(translator, "deleted", "deleted", len(payload["deleted"])))
            if args.dry_run:
                print(_txt(translator, "cli.output.dry_run_no_sessions_deleted", "dry-run: no sessions deleted"))
        return 0

    if args.command == "checkpoint":
        store = CheckpointStore(workspace.project_root, workspace.workspace_dir)
        if args.list or not args.path:
            checkpoints = store.list()
            if args.json:
                print_json({"checkpoints": [item.to_dict() for item in checkpoints]})
            else:
                for item in checkpoints[-30:]:
                    print(f"{item.checkpoint_id}\t{item.created_at}\t{_label(translator, 'files', 'files')}={len(item.files)}\t{item.reason}")
            return 0
        checkpoint = store.create(args.path, reason=args.reason)
        if args.json:
            print_json(checkpoint.to_dict())
        else:
            print(_kv(translator, "checkpoint_id", "checkpoint_id", checkpoint.checkpoint_id))
            print(_kv(translator, "manifest", "manifest", checkpoint.manifest_path))
            print(_kv(translator, "files", "files", len(checkpoint.files)))
        return 0

    if args.command == "restore":
        checkpoint = CheckpointStore(workspace.project_root, workspace.workspace_dir).restore(args.checkpoint_id)
        if args.json:
            print_json(checkpoint.to_dict())
        else:
            print(_kv(translator, "restored", "restored", checkpoint.checkpoint_id))
            for path in checkpoint.files:
                print(f"  {path}")
        return 0

    if args.command == "session-new":
        session = session_store.new(args.title)
        print(session.session_id)
        return 0

    if args.command == "session-add":
        session = session_store.add_message(args.session_id, args.role, args.content)
        print(f"{session.session_id}\t{_label(translator, 'messages', 'messages')}={len(session.messages)}")
        return 0

    if args.command == "session-list":
        sessions = session_store.list(include_archived=args.include_archived)
        if args.json:
            print_json({"sessions": [session.to_dict() for session in sessions]})
        else:
            for session in sessions:
                print(f"{session.session_id}\t{session.status}\t{len(session.messages)}\t{session.title}")
        return 0

    if args.command == "session-show":
        session = session_store.load(args.session_id)
        if args.json:
            print_json(session.to_dict())
        else:
            print(_kv(translator, "session_id", "session_id", session.session_id))
            print(_kv(translator, "title", "title", session.title))
            print(_kv(translator, "status", "status", session.status))
            print(_kv(translator, "summaries", "summaries", len(session.summaries)))
            for message in session.messages:
                print(f"{message.role}: {message.content}")
        return 0

    if args.command == "session-archive":
        session = session_store.archive(args.session_id)
        print(_kv(translator, "archived", "archived", session.session_id))
        return 0

    if args.command == "session-delete":
        session_store.delete(args.session_id, hard=args.hard)
        print(_kv(translator, "deleted", "deleted", args.session_id))
        return 0

    if args.command == "session-compress":
        session = session_store.compress(
            args.session_id,
            keep_last=args.keep_last,
            max_chars=args.max_chars,
        )
        print(
            f"{session.session_id}\t{_label(translator, 'messages', 'messages')}={len(session.messages)}"
            f"\t{_label(translator, 'summaries', 'summaries')}={len(session.summaries)}"
        )
        return 0

    if args.command == "skill-list":
        roots = skill_roots(workspace)
        skills = SkillStore(roots).list()
        if args.json:
            print_json({"skills": [skill.to_dict() for skill in skills]})
        else:
            print(_label(translator, "skills", "skills"))
            for skill in skills:
                print(format_skill_block(translator, skill))
        return 0

    if args.command == "skill-search":
        roots = skill_roots(workspace)
        skills = SkillStore(roots).search(args.query)
        if args.json:
            print_json({"skills": [skill.to_dict() for skill in skills]})
        else:
            print(_label(translator, "skills", "skills"))
            for skill in skills:
                print(format_skill_block(translator, skill))
        return 0

    if args.command == "skill-inspect":
        roots = skill_roots(workspace)
        skill = SkillStore(roots).get(args.name)
        if skill is None:
            raise KeyError(_txt(translator, "cli.error.skill_not_found", "skill not found: {name}", name=args.name))
        if args.json:
            print_json(skill.to_dict())
        else:
            print(_kv(translator, "name", "name", skill.name))
            print(_kv(translator, "path", "path", skill.path))
            print(_kv(translator, "run_as", "run_as", skill.run_as))
            print(_kv(translator, "allowed_tools", "allowed_tools", ", ".join(skill.allowed_tools)))
            print(_kv(translator, "description", "description", skill.description))
        return 0

    if args.command == "skill-install":
        store = SkillStore([Path(args.path)])
        skill = store.install(
            Path(args.path),
            workspace.workspace_dir / "skills",
            name=args.name,
        )
        if args.json:
            print_json(skill.to_dict())
        else:
            print(_kv(translator, "installed", "installed", skill.name))
            print(_kv(translator, "path", "path", skill.path))
        return 0

    if args.command == "skill-bundles":
        roots = skill_roots(workspace)
        bundles = SkillStore(roots).bundles()
        payload = {
            "bundles": {
                name: [skill.to_dict() for skill in skills]
                for name, skills in bundles.items()
            }
        }
        if args.json:
            print_json(payload)
        else:
            for name, skills in sorted(bundles.items()):
                print(f"{name}\t{len(skills)}")
                for skill in skills:
                    print(f"  {skill.name}\t{skill.description}")
        return 0

    if args.command == "skill-run":
        roots = skill_roots(workspace)
        skill = SkillStore(roots).render(args.name, args.arguments)
        if not args.run:
            payload = skill.to_dict()
            if args.json:
                print_json(payload)
            else:
                print(skill.body)
            return 0
        llm_store.ensure_defaults()
        session_id = session_store.new(f"skill:{skill.name}").session_id
        run = AgentActionLoop(
            session_store,
            llm_store,
            workspace,
            permission_policy="session-ask",
            permission_callback=confirm_tool_permission,
        ).run(
            session_id=session_id,
            provider=args.provider,
            instruction=(
                f"Run skill `{skill.name}` ({skill.run_as}) for this task.\n\n"
                f"{skill.body}"
            ),
            model=resolve_llm_model(args) or skill.model or None,
            max_steps=args.max_steps,
            **build_llm_runtime_kwargs(args),
        )
        payload = {
            "skill": skill.to_dict(),
            "session_id": session_id,
            "action_run": str(Path(run.run_dir) / "action_run.json"),
            "status": run.status,
            "final_response": run.final_response,
            "warnings": run.warnings,
        }
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "skill", "skill", skill.name))
            print(_kv(translator, "status", "status", run.status))
            print(_kv(translator, "action_run", "action_run", payload["action_run"]))
            if run.final_response:
                print(run.final_response)
            for warning in run.warnings:
                print(_warning(translator, warning))
        return 0

    if args.command == "mcp-add":
        server = MCPServerSpec(
            name=args.name,
            command=args.server_command,
            args=args.arg,
            env=parse_env(args.env),
            description=args.description,
        )
        mcp_store.upsert(server)
        print(_txt(translator, "cli.output.registered_mcp_server", "registered MCP server: {name}", name=server.name))
        return 0

    if args.command == "mcp-list":
        servers = mcp_store.list()
        if args.json:
            print_json({"servers": [server.to_dict() for server in servers]})
        else:
            for server in servers:
                print(f"{server.name}\t{server.command}\t{' '.join(server.args)}")
        return 0

    if args.command == "mcp-export":
        payload = mcp_store.export_client_config()
        if args.output:
            write_json(Path(args.output), payload)
            print(_txt(translator, "cli.output.wrote_mcp_config", "Wrote MCP config: {path}", path=args.output))
        else:
            print_json(payload)
        return 0

    if args.command == "mcp-inspect":
        matched = None
        for server in mcp_store.list():
            if server.name == args.name:
                matched = server
                break
        if matched is None:
            raise KeyError(_txt(translator, "cli.error.mcp_server_not_found", "MCP server not found: {name}", name=args.name))
        payload = {
            **matched.to_dict(),
            "runtime_client": "not_connected",
            "note": "HyperAgent currently stores MCP launch specs; live MCP client runtime is planned.",
        }
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "name", "name", matched.name))
            print(_kv(translator, "enabled", "enabled", matched.enabled))
            print(_kv(translator, "command", "command", f"{matched.command} {' '.join(matched.args)}"))
            print(_kv(translator, "runtime_client", "runtime_client", _txt(translator, "cli.value.not_connected", "not_connected")))
        return 0

    if args.command == "mcp-health":
        payload = {
            "servers": [
                {
                    **server.to_dict(),
                    "configured": True,
                    "runtime_client": "not_connected",
                    "health": "registered",
                }
                for server in mcp_store.list()
            ]
        }
        if args.json:
            print_json(payload)
        else:
            if not payload["servers"]:
                print(_txt(translator, "cli.output.no_mcp_servers", "no MCP servers"))
            for server in payload["servers"]:
                print(f"{server['name']}\t{server['enabled']}\t{server['health']}\t{server['runtime_client']}")
        return 0

    if args.command == "index":
        store = SemanticIndexStore(workspace.project_root, workspace.workspace_dir)
        if args.query:
            results = store.search(args.query, limit=args.limit)
            if args.json:
                print_json({"results": results, "index_path": str(store.path)})
            else:
                for result in results:
                    print(f"{result['score']}\t{result['path']}\t{', '.join(result['matched_terms'][:8])}")
            return 0
        roots = args.root or ["hyperagent", "tests", "README.md", "README.zh-CN.md"]
        payload = store.build(roots)
        if args.json:
            print_json(payload)
        else:
            print(_kv(translator, "index", "index", store.path))
            print(_kv(translator, "documents", "documents", len(payload["documents"])))
            print(_kv(translator, "engine", "engine", payload["engine"]))
        return 0


    if args.command == "research-mcp-serve":
        from hyperagent.runtime.research_mcp import run_research_mcp_server

        return run_research_mcp_server(workspace.project_root)

    if args.command == "research":
        research_agent = ResearchExperienceAgent(
            workspace.project_root,
            workspace.workspace_dir,
            llm_store=llm_store,
        )
        if args.research_command == "extract":
            card = research_agent.extract(
                args.paper,
                provider=args.provider,
                model=args.model,
                field=args.field,
                title=args.title,
                venue=args.venue,
                year=args.year,
                write=not args.no_write,
            )
            payload = card.to_dict()
        elif args.research_command in {"pattern", "experiment", "storytelling"}:
            section_map = {
                "pattern": "research_pattern",
                "experiment": "experiment_strategy",
                "storytelling": "scientific_storytelling",
            }
            payload = research_agent.extract_section(
                section_map[args.research_command],
                args.paper,
                provider=args.provider,
                model=args.model,
                field=args.field,
                write=not args.no_write,
            )
        elif args.research_command == "taste":
            papers = parse_csv(args.papers)
            payload = research_agent.compare_paper_strategies(
                papers,
                provider=args.provider,
                model=args.model,
                field=args.field,
            )
        elif args.research_command == "consolidate":
            payload = research_agent.consolidate_research_experience(
                args.topic,
                papers=parse_csv(args.papers),
                provider=args.provider,
                model=args.model,
                field=args.field,
            )
        elif args.research_command == "search":
            payload = research_agent.search_dimension(
                args.dimension,
                args.query,
                top_k=args.top_k,
                field=args.field,
            )
        else:
            raise ValueError(f"Unsupported research command: {args.research_command}")
        append_worklog(
            "运行科研经验提炼命令",
            "HyperAgent 已具备基础科研实验与文献工具。",
            f"执行 research {args.research_command}，面向 paper strategy 而非论文内容摘要。",
            "科研经验需要进入可审计的 agent 工作流，方便后续 Codex、Claude Code 和 HyperVault 共享。",
            f"命令输出包含 keys={sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}。",
            "科研经验提炼结果已生成或检索完成。",
            "下一步可人工确认 strategy card，再 consolidation 进入长期 memory。",
        )
        if getattr(args, "json", False):
            print_json(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "obsidian-index":
        notes = obsidian_store.index(Path(args.vault))
        print(_txt(translator, "cli.output.indexed_notes", "indexed notes: {count}", count=len(notes)))
        return 0

    if args.command == "obsidian-search":
        notes = obsidian_store.search(args.query, limit=args.limit)
        if args.json:
            print_json({"notes": [note.to_dict() for note in notes]})
        else:
            for note in notes:
                print(f"{note.title}\t{note.path}")
        return 0

    if args.command == "prompt-list":
        templates = prompt_library.list()
        if args.json:
            print_json({"prompts": [template.to_dict() for template in templates]})
        else:
            for template in templates:
                print(f"{template.name}\t{','.join(template.variables)}\t{template.description}")
        return 0

    if args.command == "prompt-render":
        print(prompt_library.render(args.name, parse_vars(args.var)))
        return 0

    if args.command == "materialize-module":
        proposal = ModuleProposal.from_dict(read_json(Path(args.proposal)))
        result = ModuleMaterializer().materialize(
            proposal,
            output_dir=Path(args.output_dir),
            base_plan_path=Path(args.base_plan) if args.base_plan else None,
            ablation_output_dir=Path(args.ablation_output) if args.ablation_output else None,
            force=args.force,
        )
        append_worklog(
            "物化模块建议为模型代码",
            "module_proposal JSON 已生成。",
            f"将 {proposal.name} 物化为模型 factory {result.model_name}。",
            "模块建议必须落到 registry-backed model factory，才能进入训练和消融流程。",
            f"生成模型文件 {result.model_file}，消融配置数量 {len(result.generated_configs)}。",
            "模块代码和消融配置已生成。",
            "下一步可运行 ablation YAML 验证模块效果。",
        )
        print_json(result.to_dict())
        return 0

    if args.command == "plan":
        audit = DatasetAudit.from_dict(read_json(Path(args.audit)))
        base = Path(args.output).parent
        spectral_path = base / "spectral_report.json"
        recommendation_path = base / "model_recommendation.json"
        spectral_report = agent.analyze(audit, spectral_path)
        recommendation = agent.recommend(audit, spectral_report, recommendation_path)
        plan = agent.plan(
            audit,
            spectral_report,
            recommendation,
            Path(args.output),
            Path(args.output_dir) if args.output_dir else None,
            args.seed,
        )
        append_worklog(
            "生成实验计划",
            "数据审计 JSON 已存在。",
            f"生成光谱报告、模型推荐和实验计划 {args.output}。",
            "计划文件统一固定 split、预处理、模型和输出目录，保证实验可复现。",
            f"推荐模型为 {plan.model.name}，输出目录为 {plan.output_dir}。",
            "实验 YAML、光谱报告和推荐报告已生成。",
            "下一步可运行 run-baseline 命令执行训练。",
        )
        print(_txt(translator, "cli.output.wrote_plan", "Wrote plan: {path}", path=args.output))
        return 0

    if args.command == "run-baseline":
        plan = ExperimentPlan.from_dict(read_yaml(Path(args.config)))
        result = agent.run(plan)
        append_worklog(
            "运行 baseline 实验",
            "实验计划 YAML 已生成。",
            f"执行 baseline runner，配置文件为 {args.config}。",
            "训练与评估必须从同一份 ExperimentPlan 启动，避免隐藏参数。",
            f"完成 {result.model_name} 训练，OA={result.evaluation.overall_accuracy:.4f}。",
            f"结果已写入 {result.experiment_dir}/result.json。",
            "下一步可运行 report 命令生成 Markdown 报告。",
        )
        print(_txt(translator, "cli.output.wrote_result", "Wrote result: {path}", path=Path(result.experiment_dir) / "result.json"))
        return 0

    if args.command == "run-suite":
        plan = ExperimentPlan.from_dict(read_yaml(Path(args.config)))
        seeds = parse_int_list(args.seeds)
        suite = ExperimentSuiteRunner().run(
            plan,
            seeds=seeds,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            suite_name=args.suite_name,
        )
        summary = suite.metrics_summary["overall_accuracy"]
        append_worklog(
            "运行多 Seed 实验套件",
            "单次实验计划 YAML 已生成。",
            f"基于 {args.config} 执行 seeds={seeds} 的实验套件。",
            "多 seed 汇总能在换参数或加模块前估计随机性，避免 agent 依据单次结果钻牛角尖。",
            f"完成 {suite.run_count} 次运行，OA mean={summary['mean']:.4f} std={summary['std']:.4f}。",
            f"suite.json 已写入 {Path(suite.output_dir) / 'suite.json'}。",
            "下一步可把 suite 结果交给 experiment-cycle 或模块消融流程作为稳定性依据。",
        )
        print(_txt(translator, "cli.output.wrote_suite", "Wrote suite: {path}", path=Path(suite.output_dir) / "suite.json"))
        print(_kv(translator, "report", "report", Path(suite.output_dir) / "suite_report.md"))
        print(_kv(translator, "oa_mean", "oa_mean", f"{summary['mean']:.4f}"))
        print(_kv(translator, "oa_std", "oa_std", f"{summary['std']:.4f}"))
        return 0

    if args.command == "report":
        result = ExperimentResult.from_dict(read_json(Path(args.experiment) / "result.json"))
        output = Path(args.output) if args.output else Path(args.experiment) / "report.md"
        report_path = agent.write_report(result, output)
        append_worklog(
            "生成实验报告",
            "实验结果 JSON 已生成。",
            f"根据 {args.experiment}/result.json 生成 Markdown 报告。",
            "报告把指标、混淆矩阵和 artifact 汇总为论文实验可读格式。",
            f"报告已写入 {report_path}。",
            "端到端流程已形成可复现实验记录。",
            "下一步可扩展真实数据集和更多模型。",
        )
        print(_txt(translator, "cli.output.wrote_report", "Wrote report: {path}", path=report_path))
        return 0

    if args.command == "demo":
        root = Path(args.root)
        data_root = root / "data"
        write_synthetic_mat(data_root, args.seed)
        audit_path = root / "audit.json"
        spectral_path = root / "spectral_report.json"
        recommendation_path = root / "model_recommendation.json"
        plan_path = root / "experiment.yaml"
        audit = agent.audit(data_root, audit_path)
        spectral_report = agent.analyze(audit, spectral_path)
        recommendation = agent.recommend(audit, spectral_report, recommendation_path)
        plan = agent.plan(audit, spectral_report, recommendation, plan_path, root / "run", args.seed)
        result = agent.run(plan)
        report_path = agent.write_report(result, Path(result.experiment_dir) / "report.md")
        write_json(root / "summary.json", result)
        append_worklog(
            "运行 synthetic demo",
            "CLI、Agent、工具、训练和报告模块已经实现。",
            f"在 {root} 下生成合成 HSI 数据并执行完整流程。",
            "synthetic demo 不依赖外部数据，可以作为最小端到端验收。",
            f"完成训练与报告生成，OA={result.evaluation.overall_accuracy:.4f}，报告为 {report_path}。",
            "audit、spectral report、recommendation、experiment YAML、result 和 report 均已生成。",
            "下一步运行单元测试与解耦性检查。",
        )
        print(_txt(translator, "cli.output.demo_complete", "Demo complete: {path}", path=report_path))
        return 0

    if args.command == "literature":
        result = agent.search_literature(
            args.query,
            Path(args.output),
            provider_name=args.provider,
            max_results=args.max_results,
            year_from=args.year_from,
            sort_by=args.sort_by,
        )
        append_worklog(
            "检索相关最新文献",
            "第二阶段文献 provider 已接入。",
            f"使用 {args.provider} 检索关键词 `{args.query}`，最多 {args.max_results} 篇。",
            "文献检索结果将作为模块设计和实验依据来源之一。",
            f"检索到 {len(result.papers)} 条记录并写入 {args.output}。",
            "文献 JSON 已生成。",
            "下一步可运行 propose-module 结合文献生成模块建议。",
        )
        print(_txt(translator, "cli.output.wrote_literature", "Wrote literature results: {path}", path=args.output))
        return 0

    if args.command == "auto-experiment":
        audit = DatasetAudit.from_dict(read_json(Path(args.audit)))
        spectral = SpectralReport.from_dict(read_json(Path(args.spectral)))
        recommendation = ModelRecommendation.from_dict(read_json(Path(args.recommendation)))
        agenda = agent.design_auto_experiments(
            audit,
            spectral,
            recommendation,
            objective=args.objective,
            max_candidates=args.max_candidates,
        )
        write_json(Path(args.output), agenda)
        append_worklog(
            "生成自动实验议程",
            "审计、光谱报告和模型推荐已经存在。",
            f"为 {audit.dataset_name} 生成 {len(agenda.candidates)} 个带依据的实验候选。",
            "自动实验必须以 objective 和 evidence 为约束，避免盲目网格搜索。",
            f"实验议程已写入 {args.output}。",
            "自动实验候选已生成。",
            "下一步可选择候选并 materialize 为具体 ExperimentPlan。",
        )
        print(_txt(translator, "cli.output.wrote_auto_experiment_agenda", "Wrote auto-experiment agenda: {path}", path=args.output))
        return 0

    if args.command == "tune-next":
        plan = ExperimentPlan.from_dict(read_yaml(Path(args.plan)))
        result = ExperimentResult.from_dict(read_json(Path(args.result)))
        audit = DatasetAudit.from_dict(read_json(Path(args.audit)))
        proposals = agent.propose_parameter_updates(plan, result, audit)
        write_json(Path(args.output), {"proposals": [item.to_dict() for item in proposals]})
        append_worklog(
            "生成目的性调参建议",
            "已有实验计划和实验结果。",
            f"根据 {result.experiment_name} 的结果生成 {len(proposals)} 个参数调整建议。",
            "调参应由结果缺口和数据证据驱动，而不是无目的枚举。",
            f"调参建议已写入 {args.output}。",
            "下一轮实验参数依据已生成。",
            "下一步可将建议应用到新的 ExperimentPlan。",
        )
        print(_txt(translator, "cli.output.wrote_tuning_proposals", "Wrote tuning proposals: {path}", path=args.output))
        return 0

    if args.command == "experiment-cycle":
        if args.resume_paused:
            resume_path = Path(args.resume_paused)
            cycle_path = resume_path / "cycle.json" if resume_path.is_dir() else resume_path
            paused_cycle = ExperimentCycle.from_dict(read_json(cycle_path))
            plan_path = Path(paused_cycle.previous_plan_path)
            result_path = Path(paused_cycle.previous_result_path)
            audit_path = Path(paused_cycle.audit_path)
            output_root = Path(paused_cycle.cycle_dir).parent
        else:
            if not args.plan or not args.result or not args.audit:
                raise ValueError(
                    "experiment-cycle requires --plan, --result, and --audit unless --resume-paused is used"
                )
            plan_path = Path(args.plan)
            result_path = Path(args.result)
            audit_path = Path(args.audit)
            output_root = Path(args.output_root)
        plan = ExperimentPlan.from_dict(read_yaml(plan_path))
        result = ExperimentResult.from_dict(read_json(result_path))
        audit = DatasetAudit.from_dict(read_json(audit_path))
        cycle = ExperimentAutopilotAgent(
            workspace_dir=workspace.workspace_dir,
            llm_store=llm_store,
        ).run_cycle(
            plan,
            result,
            audit,
            previous_plan_path=plan_path,
            previous_result_path=result_path,
            audit_path=audit_path,
            output_root=output_root,
            objective=args.objective,
            target_oa=args.target_oa,
            run_next=args.run_next,
            max_repeated_parameter=args.max_repeated_parameter,
            council_mode=args.council_mode,
            llm_council=args.llm_council,
            council_profile=args.council_profile,
            council_llm_budget=args.council_llm_budget,
            llm_required=args.council_mode == "executable",
            llm_wait_on_failure=args.council_mode == "executable",
            llm_retry_interval_sec=args.llm_gate_retry_sec,
            llm_gate_token_budget=args.llm_gate_token_budget,
        )
        append_worklog(
            "执行自动实验闭环",
            "已有上一轮实验计划、结果和数据审计。",
            f"分析实验 {result.experiment_name} 并生成 cycle {cycle.cycle_id}。",
            "实验闭环必须先诊断结果，再依据证据生成下一轮计划；只有显式 --run-next 才直接运行新实验。",
            f"诊断={cycle.diagnosis_path}，council={cycle.council_path}，下一轮计划={cycle.next_plan_path}，结果={cycle.next_result_path or 'not_run'}。",
            f"cycle 状态为 {cycle.status}。",
            "下一步可查看 cycle.json、运行 next_experiment.yaml，或继续开启下一轮 experiment-cycle。",
        )
        print(_kv(translator, "cycle_id", "cycle_id", cycle.cycle_id))
        print(_kv(translator, "status", "status", cycle.status))
        print(_kv(translator, "diagnosis", "diagnosis", cycle.diagnosis_path))
        print(_kv(translator, "proposals", "proposals", cycle.proposals_path))
        print(_kv(translator, "council", "council", cycle.council_path))
        if cycle.council_run_path:
            print(_kv(translator, "council_run", "council_run", cycle.council_run_path))
        print(_kv(translator, "next_plan", "next_plan", cycle.next_plan_path))
        if cycle.pause_reason:
            print(_kv(translator, "pause_reason", "pause_reason", cycle.pause_reason))
            if cycle.pause_details:
                print(_kv(translator, "pause_details", "pause_details", cycle.pause_details))
        if cycle.next_result_path:
            print(_kv(translator, "next_result", "next_result", cycle.next_result_path))
        if cycle.report_path:
            print(_kv(translator, "report", "report", cycle.report_path))
        for warning in cycle.warnings:
            print(_warning(translator, warning))
        return 0

    if args.command == "propose-module":
        audit = DatasetAudit.from_dict(read_json(Path(args.audit)))
        spectral = SpectralReport.from_dict(read_json(Path(args.spectral)))
        literature = LiteratureSearchResult.from_dict(read_json(Path(args.literature)))
        proposal = agent.propose_module(
            audit,
            spectral,
            literature.papers,
            objective=args.objective,
        )
        write_json(Path(args.output), proposal)
        append_worklog(
            "生成目的性模块建议",
            "已有数据审计、光谱报告和文献检索结果。",
            f"为 {audit.dataset_name} 生成模块建议 {proposal.name}。",
            "模块添加必须说明插入点、预期效果、实现步骤、风险和证据。",
            f"模块建议已写入 {args.output}。",
            "模块设计依据已结构化保存。",
            "下一步可按建议新增模型 factory 并做消融实验。",
        )
        print(_txt(translator, "cli.output.wrote_module_proposal", "Wrote module proposal: {path}", path=args.output))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def parse_keywords(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_env(values: List[str]) -> dict:
    parsed = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE env item, got {item}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed



def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_vars(values: List[str]) -> dict:
    return parse_env(values)


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def add_llm_runtime_args(
    parser: argparse.ArgumentParser,
    translator: Optional[Translator] = None,
) -> None:
    parser.add_argument(
        "--reasonix-profile",
        choices=["reasonix-cheap", "reasonix-balanced", "reasonix-deep"],
        default=None,
        help=_txt(
            translator,
            "cli.arg.reasonix_profile.help",
            "DeepSeek Reasonix-inspired runtime preset.",
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=_txt(translator, "cli.arg.top_p.help", "Nucleus sampling top-p value."),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help=_txt(
            translator,
            "cli.arg.thinking.help",
            "DeepSeek thinking mode switch for supported models.",
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help=_txt(
            translator,
            "cli.arg.reasoning_effort.help",
            "Reasoning strength for providers that support it.",
        ),
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help=_txt(
            translator,
            "cli.arg.json_output.help",
            "Request JSON-object output when the provider supports response_format.",
        ),
    )
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help=_txt(
            translator,
            "cli.arg.extra_body_json.help",
            "Raw JSON object merged into the provider request body.",
        ),
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=_txt(
            translator,
            "cli.arg.user_id.help",
            "Optional provider-side user identifier, separate from the prompt text.",
        ),
    )


def build_llm_runtime_kwargs(args: argparse.Namespace) -> dict:
    extra_body = parse_json_object(args.extra_body_json, "--extra-body-json")
    profile = get_reasonix_profile(getattr(args, "reasonix_profile", None))
    if profile and getattr(args, "provider", "deepseek") != "deepseek":
        raise ValueError("--reasonix-profile can only be used with provider=deepseek")
    thinking_type = args.thinking or (profile.thinking if profile else None)
    reasoning_effort = args.reasoning_effort or (
        profile.reasoning_effort if profile else None
    )
    top_p = args.top_p if args.top_p is not None else (profile.top_p if profile else None)
    thinking = {"type": thinking_type} if thinking_type else None
    response_format = {"type": "json_object"} if args.json_output else None
    return {
        "top_p": top_p,
        "response_format": response_format,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "user": args.user_id,
        "extra_body": extra_body,
    }


def resolve_llm_model(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "model", None):
        return args.model
    profile = get_reasonix_profile(getattr(args, "reasonix_profile", None))
    return profile.model if profile else None


def print_profile(profile: dict, translator: Optional[Translator] = None) -> None:
    print(
        f"{profile['name']}: {_label(translator, 'model', 'model')}={profile['model']} "
        f"{_label(translator, 'thinking', 'thinking')}={profile['thinking']} "
        f"{_label(translator, 'reasoning_effort', 'effort')}={profile['reasoning_effort']}"
    )
    print(f"  {_label(translator, 'intent', 'intent')}: {profile['intent']}")
    if profile.get("use_cases"):
        print(f"  {_label(translator, 'use_cases', 'use_cases')}: " + ", ".join(profile["use_cases"]))
    if profile.get("cache_policy"):
        print(f"  {_label(translator, 'cache_policy', 'cache_policy')}: " + " | ".join(profile["cache_policy"]))


def parse_json_object(value: Optional[str], flag_name: str) -> dict:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must be a JSON object")
    return parsed


def skill_roots(workspace: HyperAgentWorkspace) -> List[Path]:
    codex_home = os.environ.get("CODEX_HOME")
    return [
        PACKAGE_ROOT / "skills",
        Path("skills"),
        workspace.workspace_dir / "skills",
        Path(codex_home) / "skills" if codex_home else Path.home() / ".codex" / "skills",
    ]


def format_skill_block(translator: Optional[Translator], skill) -> str:
    description = " ".join(str(skill.description or "").split())
    if len(description) > 160:
        description = description[:157].rstrip() + "..."
    allowed_tools = ", ".join(str(item) for item in (skill.allowed_tools or []))
    if not allowed_tools:
        allowed_tools = _txt(translator, "cli.value.none", "none")
    return "\n".join(
        [
            f"- {skill.name}",
            f"  {_label(translator, 'run_as', 'run_as')}: {skill.run_as}",
            f"  {_label(translator, 'allowed_tools', 'allowed_tools')}: {allowed_tools}",
            f"  {_label(translator, 'path', 'path')}: {skill.path}",
            f"  {_label(translator, 'description', 'description')}: {description}",
            "  "
            + _txt(
                translator,
                "cli.output.skill_invoke",
                "invoke: /skill {name} <instruction> or /{name} <instruction>",
                name=skill.name,
            ),
        ]
    )


def confirm_tool_permission(request) -> bool:
    translator = _ACTIVE_TRANSLATOR
    print(
        _txt(
            translator,
            "cli.output.permission_requested",
            "permission requested: {tool_name} risk={risk_level} reason={reason}",
            tool_name=request.tool_name,
            risk_level=request.risk_level,
            reason=request.reason,
        )
    )
    answer = input(_txt(translator, "cli.output.allow_prompt", "allow? [y/N] ")).strip().lower()
    return answer in {"y", "yes"}


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
