"""Command line interface for HyperAgent."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from hyperagent.agents import BenchmarkAgent, CoordinatorAgent, ExperimentAutopilotAgent
from hyperagent.core.io import read_json, read_yaml, write_json
from hyperagent.core.worklog import append_worklog
from hyperagent.data.synthetic import write_synthetic_mat
from hyperagent.runtime.agent_loop import AgentLoop
from hyperagent.runtime.action_loop import AgentActionLoop
from hyperagent.runtime.agent_tools import SafeAgentToolExecutor
from hyperagent.runtime.coding_agent import CodingAgent
from hyperagent.runtime.repo_context import RepoContextBuilder
from hyperagent.runtime.workspace import HyperAgentWorkspace
from hyperagent.runtime.conversations import ConversationStore
from hyperagent.runtime.env import load_env_file
from hyperagent.runtime.command_aliases import command_help_text
from hyperagent.runtime.llm import LLMClient, LLMProviderStore, LLMRequestBuilder
from hyperagent.runtime.mcp import MCPServerStore
from hyperagent.runtime.obsidian import ObsidianVaultIndex
from hyperagent.runtime.prompts import PromptLibrary
from hyperagent.runtime.skills import SkillStore
from hyperagent.schemas import (
    DatasetAudit,
    ExperimentPlan,
    ExperimentResult,
    LLMMessage,
    MCPServerSpec,
    LiteratureSearchResult,
    ModuleProposal,
    ModelRecommendation,
    SpectralReport,
)
from hyperagent.tools.module_materializer import ModuleMaterializer
from hyperagent.training.experiment_suite import ExperimentSuiteRunner


DEFAULT_DATASET_ROOT = "/data2/lzj/lab/Mamba_test/dataset"
PACKAGE_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperAgent HSI research workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Inspect an HSI dataset")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--reader", default=None)

    plan = subparsers.add_parser("plan", help="Build an experiment plan from an audit")
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

    status = subparsers.add_parser("status", help="Show HyperAgent workspace status")
    status.add_argument("--json", action="store_true")

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
    experiment_cycle.add_argument("--plan", required=True)
    experiment_cycle.add_argument("--result", required=True)
    experiment_cycle.add_argument("--audit", required=True)
    experiment_cycle.add_argument("--output-root", default="experiments/autopilot")
    experiment_cycle.add_argument("--objective", default="maximize_oa_with_reproducible_baseline")
    experiment_cycle.add_argument("--target-oa", type=float, default=0.9)
    experiment_cycle.add_argument("--run-next", action="store_true")
    experiment_cycle.add_argument("--max-repeated-parameter", type=int, default=2)

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

    llm_dry = subparsers.add_parser("llm-dry-run", help="Build a vendor request payload without sending it")
    llm_dry.add_argument("--provider", required=True)
    llm_dry.add_argument("--model", default=None)
    llm_dry.add_argument("--system", default="You are HyperAgent.")
    llm_dry.add_argument("--user", required=True)
    llm_dry.add_argument("--temperature", type=float, default=0.2)
    llm_dry.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(llm_dry)

    llm_send = subparsers.add_parser("llm-send", help="Send a prompt to a configured LLM provider")
    llm_send.add_argument("--provider", required=True)
    llm_send.add_argument("--model", default=None)
    llm_send.add_argument("--system", default="You are HyperAgent.")
    llm_send.add_argument("--user", required=True)
    llm_send.add_argument("--temperature", type=float, default=0.2)
    llm_send.add_argument("--max-tokens", type=int, default=None)
    add_llm_runtime_args(llm_send)
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
    add_llm_runtime_args(agent_chat)
    agent_chat.add_argument("--max-context-chars", type=int, default=12000)
    agent_chat.add_argument("--no-auto-compress", action="store_true")
    agent_chat.add_argument("--output", default=None)

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
    add_llm_runtime_args(agent_plan)
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
    add_llm_runtime_args(agent_act)
    agent_act.add_argument("--max-files", type=int, default=12)
    agent_act.add_argument("--max-preview-chars", type=int, default=1000)

    agent_tool = subparsers.add_parser(
        "agent-tool",
        help="Run a controlled Claude-Code-like local tool",
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

    tool_check_patch = agent_tool_sub.add_parser("check-patch", help="Validate a unified diff with git apply --check")
    tool_check_patch.add_argument("--patch-file", required=True)
    tool_check_patch.add_argument("--run-id", default=None)
    tool_check_patch.add_argument("--json", action="store_true")

    tool_apply_patch = agent_tool_sub.add_parser("apply-patch", help="Apply a unified diff through git apply")
    tool_apply_patch.add_argument("--patch-file", required=True)
    tool_apply_patch.add_argument("--run-id", default=None)
    tool_apply_patch.add_argument("--json", action="store_true")

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

    subparsers.add_parser(
        "hyperagent-commands",
        help="Show Claude-Code-like HyperAgent command aliases",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    load_env_file(Path(".env"), override=False)
    agent = CoordinatorAgent()
    workspace = HyperAgentWorkspace()
    llm_store = LLMProviderStore(workspace.workspace_dir)
    session_store = ConversationStore(workspace.workspace_dir)
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
        print(f"Initialized HyperAgent workspace: {workspace.workspace_dir}")
        return 0

    if args.command == "hyperagent-commands":
        print(command_help_text())
        return 0

    if args.command == "status":
        status = workspace.status()
        if args.json:
            print_json(status.to_dict())
        else:
            print(f"initialized: {status.initialized}")
            print(f"workspace: {status.workspace_dir}")
            print(f"dataset_root: {status.dataset_root}")
            print(f"tasks: {status.task_count}")
            print(f"tasks_by_status: {status.tasks_by_status}")
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
        print(f"Wrote matrix: {Path(args.reports_root) / 'benchmark_matrix.json'}")
        print(f"report: {Path(args.reports_root) / 'benchmark_matrix.md'}")
        print(f"completed: {completed}")
        print(f"planned: {planned}")
        print(f"failed: {failed}")
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
            print(f"task_id: {task.task_id}")
            print(f"status: {task.status}")
            print(f"dataset: {task.dataset}")
            print(f"objective: {task.objective}")
            print(f"goal: {task.goal}")
            print(f"keywords: {', '.join(task.keywords)}")
            print(f"artifacts: {task.artifacts}")
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
        print(f"Task run complete: {task.task_id}")
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
        print(f"Wrote audit: {args.output}")
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
                configured = "configured" if os.environ.get(provider.api_key_env) else "missing"
                print(
                    f"{provider.name}\t{provider.kind}\t{provider.default_model}\t"
                    f"{provider.api_key_env}\t{configured}"
                )
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
            model=args.model,
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
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **build_llm_runtime_kwargs(args),
        )
        if args.output:
            write_json(Path(args.output), response)
        if response.warnings:
            for warning in response.warnings:
                print(f"warning: {warning}")
        else:
            if response.reasoning_content:
                print("reasoning_content:")
                print(response.reasoning_content)
            if response.tool_calls:
                print("tool_calls:")
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
            model=args.model,
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
        print(f"session_id: {result.session_id}")
        if result.warnings:
            for warning in result.warnings:
                print(f"warning: {warning}")
        if result.response.content:
            print(result.response.content)
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
                print(f"Wrote repo context: {args.output}")
            else:
                print_json(snapshot.to_dict())
        else:
            markdown = builder.to_markdown(snapshot)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(markdown, encoding="utf-8")
                print(f"Wrote repo context: {args.output}")
            else:
                print(markdown)
        return 0

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
            model=args.model,
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
        print(f"run_id: {run.run_id}")
        print(f"session_id: {run.session_id}")
        print(f"plan: {run.plan_path}")
        if run.warnings:
            for warning in run.warnings:
                print(f"warning: {warning}")
        return 0

    if args.command == "agent-act":
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
        ).run(
            session_id=session_id,
            provider=args.provider,
            instruction=args.message,
            model=args.model,
            task_id=args.task_id,
            max_steps=args.max_steps,
            max_files=args.max_files,
            max_preview_chars=args.max_preview_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
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
        print(f"run_id: {run.run_id}")
        print(f"session_id: {run.session_id}")
        print(f"status: {run.status}")
        print(f"action_run: {Path(run.run_dir) / 'action_run.json'}")
        if run.final_response:
            print(run.final_response)
        for warning in run.warnings:
            print(f"warning: {warning}")
        return 0

    if args.command == "agent-tool":
        executor = SafeAgentToolExecutor(workspace.project_root, workspace.workspace_dir)
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
        elif args.tool_command == "check-patch":
            patch_text = Path(args.patch_file).read_text(encoding="utf-8")
            result = executor.check_patch(patch_text, run_id=args.run_id)
        elif args.tool_command == "apply-patch":
            patch_text = Path(args.patch_file).read_text(encoding="utf-8")
            result = executor.apply_patch(patch_text, run_id=args.run_id)
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
            print(f"tool: {result.tool_name}")
            print(f"status: {result.status}")
            print(f"artifact: {result.artifact_path}")
            if result.exit_code is not None:
                print(f"exit_code: {result.exit_code}")
            for warning in result.warnings:
                print(f"warning: {warning}")
            if result.content:
                print(result.content)
        return 0

    if args.command == "session-new":
        session = session_store.new(args.title)
        print(session.session_id)
        return 0

    if args.command == "session-add":
        session = session_store.add_message(args.session_id, args.role, args.content)
        print(f"{session.session_id}\tmessages={len(session.messages)}")
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
            print(f"session_id: {session.session_id}")
            print(f"title: {session.title}")
            print(f"status: {session.status}")
            print(f"summaries: {len(session.summaries)}")
            for message in session.messages:
                print(f"{message.role}: {message.content}")
        return 0

    if args.command == "session-archive":
        session = session_store.archive(args.session_id)
        print(f"archived: {session.session_id}")
        return 0

    if args.command == "session-delete":
        session_store.delete(args.session_id, hard=args.hard)
        print(f"deleted: {args.session_id}")
        return 0

    if args.command == "session-compress":
        session = session_store.compress(
            args.session_id,
            keep_last=args.keep_last,
            max_chars=args.max_chars,
        )
        print(f"{session.session_id}\tmessages={len(session.messages)}\tsummaries={len(session.summaries)}")
        return 0

    if args.command == "skill-list":
        roots = [
            Path("skills"),
            workspace.workspace_dir / "skills",
        ]
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            roots.append(Path(codex_home) / "skills")
        skills = SkillStore(roots).list()
        if args.json:
            print_json({"skills": [skill.to_dict() for skill in skills]})
        else:
            for skill in skills:
                print(f"{skill.name}\t{skill.path}\t{skill.description}")
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
        print(f"registered MCP server: {server.name}")
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
            print(f"Wrote MCP config: {args.output}")
        else:
            print_json(payload)
        return 0

    if args.command == "obsidian-index":
        notes = obsidian_store.index(Path(args.vault))
        print(f"indexed notes: {len(notes)}")
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
        print(f"Wrote plan: {args.output}")
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
        print(f"Wrote result: {Path(result.experiment_dir) / 'result.json'}")
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
        print(f"Wrote suite: {Path(suite.output_dir) / 'suite.json'}")
        print(f"report: {Path(suite.output_dir) / 'suite_report.md'}")
        print(f"oa_mean: {summary['mean']:.4f}")
        print(f"oa_std: {summary['std']:.4f}")
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
        print(f"Wrote report: {report_path}")
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
        print(f"Demo complete: {report_path}")
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
        print(f"Wrote literature results: {args.output}")
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
        print(f"Wrote auto-experiment agenda: {args.output}")
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
        print(f"Wrote tuning proposals: {args.output}")
        return 0

    if args.command == "experiment-cycle":
        plan = ExperimentPlan.from_dict(read_yaml(Path(args.plan)))
        result = ExperimentResult.from_dict(read_json(Path(args.result)))
        audit = DatasetAudit.from_dict(read_json(Path(args.audit)))
        cycle = ExperimentAutopilotAgent().run_cycle(
            plan,
            result,
            audit,
            previous_plan_path=Path(args.plan),
            previous_result_path=Path(args.result),
            audit_path=Path(args.audit),
            output_root=Path(args.output_root),
            objective=args.objective,
            target_oa=args.target_oa,
            run_next=args.run_next,
            max_repeated_parameter=args.max_repeated_parameter,
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
        print(f"cycle_id: {cycle.cycle_id}")
        print(f"status: {cycle.status}")
        print(f"diagnosis: {cycle.diagnosis_path}")
        print(f"proposals: {cycle.proposals_path}")
        print(f"council: {cycle.council_path}")
        print(f"next_plan: {cycle.next_plan_path}")
        if cycle.next_result_path:
            print(f"next_result: {cycle.next_result_path}")
        if cycle.report_path:
            print(f"report: {cycle.report_path}")
        for warning in cycle.warnings:
            print(f"warning: {warning}")
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
        print(f"Wrote module proposal: {args.output}")
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


def parse_vars(values: List[str]) -> dict:
    return parse_env(values)


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def add_llm_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="DeepSeek thinking mode switch for supported models.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Reasoning strength for providers that support it.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Request JSON-object output when the provider supports response_format.",
    )
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="Raw JSON object merged into the provider request body.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional provider-side user identifier, separate from the prompt text.",
    )


def build_llm_runtime_kwargs(args: argparse.Namespace) -> dict:
    extra_body = parse_json_object(args.extra_body_json, "--extra-body-json")
    thinking = {"type": args.thinking} if args.thinking else None
    response_format = {"type": "json_object"} if args.json_output else None
    return {
        "top_p": args.top_p,
        "response_format": response_format,
        "thinking": thinking,
        "reasoning_effort": args.reasoning_effort,
        "user": args.user_id,
        "extra_body": extra_body,
    }


def parse_json_object(value: Optional[str], flag_name: str) -> dict:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must be a JSON object")
    return parsed


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
