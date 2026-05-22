"""Research-experience extraction agent.

The agent extracts transferable research strategy from papers and notes. It is
deliberately different from a paper summarizer: every lesson must explain how
authors frame, package, validate, hide weaknesses, or persuade reviewers.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hyperagent.runtime.hypervault import HyperVaultClient, HyperVaultHit, parse_markdown_metadata
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import LLMMessage
from hyperagent.schemas.research_experience import (
    EvidenceSpan,
    ExperimentStrategy,
    PaperStrategyCard,
    RESEARCH_EXPERIENCE_DIMENSIONS,
    ResearchPattern,
    ResearchTaste,
    ScientificStorytelling,
    StrategyLesson,
    all_research_experience_dimensions,
)


SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
NON_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


DIMENSION_QUERIES: Dict[str, List[str]] = {
    "novelty_construction": ["novel", "first", "new", "unlike", "不同", "首次", "创新"],
    "problem_framing": ["challenge", "problem", "difficult", "however", "面临", "问题", "挑战"],
    "gap_definition": ["gap", "limited", "remain", "lack", "insufficient", "不足", "缺少"],
    "contribution_packaging": ["contribution", "we make", "we summarize", "贡献", "主要贡献"],
    "reviewer_expectation": ["state-of-the-art", "effective", "efficient", "reproduc", "fair", "审稿", "可信"],
    "baseline_selection": ["baseline", "compare", "state-of-the-art", "SOTA", "对比", "基线"],
    "ablation_logic": ["ablation", "variant", "without", "component", "消融", "模块"],
    "control_variable": ["same", "fixed", "setting", "protocol", "split", "controlled", "相同", "固定"],
    "robustness_validation": ["robust", "sensitivity", "different", "variance", "seed", "noise", "泛化", "鲁棒"],
    "visualization_strategy": ["visual", "figure", "t-sne", "map", "attention", "图", "可视化"],
    "narrative_pacing": ["first", "then", "finally", "motivation", "overall", "首先", "随后"],
    "figure_order": ["Figure 1", "Fig. 1", "Figure 2", "Table 1", "图1", "表1"],
    "motivation_progression": ["motivat", "therefore", "thus", "to address", "因此", "动机"],
    "claim_scaffolding": ["demonstrate", "show", "indicate", "suggest", "证明", "表明"],
    "failure_hiding": ["limitation", "future work", "although", "while", "scope", "局限", "未来"],
    "reviewer_persuasion": ["significant", "consistent", "fair", "comprehensive", "convinc", "充分", "显著"],
    "trend_sensing": ["recent", "increasing", "emerging", "trend", "近年来", "趋势"],
    "sota_evolution": ["state-of-the-art", "SOTA", "previous", "latest", "advanced", "最先进"],
    "field_hotspot": ["attention", "transformer", "mamba", "diffusion", "graph", "热点"],
    "benchmark_lifecycle": ["benchmark", "dataset", "protocol", "split", "leaderboard", "数据集"],
    "innovation_density": ["simple", "lightweight", "combine", "hybrid", "dense", "轻量", "组合"],
}


STRATEGY_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    "novelty_construction": (
        "作者把 novelty 构造成一个可命名、可比较、可实验验证的研究动作，而不是只声称方法新。",
        "这种包装降低 reviewer 判断成本，让创新点能被基线、消融和图表同时支撑。",
        "将 novelty 写成：旧范式的限制 -> 一个清晰机制 -> 与现有方法的最小关键差异 -> 可验证收益。",
    ),
    "problem_framing": (
        "作者先把问题定义成领域共同痛点，再把自己的方法放在解决该痛点的自然路径上。",
        "problem framing 越贴近 reviewer 已接受的任务瓶颈，后续贡献越不容易显得任意。",
        "用一段话完成：任务重要性 -> 现有方案卡点 -> 为什么这个卡点现在值得解决。",
    ),
    "gap_definition": (
        "作者把 gap 定义成现有工作无法同时满足的约束组合，而不是泛泛说效果不足。",
        "约束型 gap 更容易导向实验设计，也能防止 novelty 被认为只是工程堆叠。",
        "把 gap 写成 A 能做到但牺牲 B，B 能做到但缺少 C，因此需要同时满足 A/B/C 的方案。",
    ),
    "contribution_packaging": (
        "作者把 contribution 包装为问题、方法机制、证据闭环三件事，而不是罗列模块。",
        "贡献列表如果对应 introduction 的 gap 和 experiment 的 evidence，会让 reviewer 感到论文结构可信。",
        "贡献 bullet 采用：我们识别/提出/验证，每条都能映射到一个实验或图表。",
    ),
    "reviewer_expectation": (
        "作者预判 reviewer 会关心公平性、SOTA 对比、稳定性和解释性，并在叙事中提前补证。",
        "提前回应 reviewer expectation 可以减少审稿人把问题写成 major concern 的机会。",
        "在实验前显式交代公平协议、强基线、消融边界和计算成本。",
    ),
    "baseline_selection": (
        "作者选择 baseline 时覆盖经典方法、当前强模型和相邻技术路线，形成 reviewer 难以质疑的比较面。",
        "baseline 不是越多越好，而是要覆盖 reviewer 心里的替代方案。",
        "baseline 表设计为 classic / recent SOTA / closest mechanism / efficient variant 四组。",
    ),
    "ablation_logic": (
        "作者用 ablation 证明每个 claim 对应的机制都有独立贡献，而不是只证明最终结果高。",
        "消融把 novelty 从黑箱结果拆成 reviewer 能接受的因果链。",
        "每个模块消融都回答：去掉它损失什么、替换它说明什么、组合它是否互补。",
    ),
    "control_variable": (
        "作者通过固定训练协议、数据划分和实现条件，把性能差异归因到方法本身。",
        "控制变量是实验可信度的地基，能防止 reviewer 质疑 comparison unfair。",
        "显式写清 same split、same backbone、same budget、same augmentation、same metric。",
    ),
    "robustness_validation": (
        "作者增加跨数据集、不同设置或扰动实验，让结论从单点性能升级为稳定趋势。",
        "robustness 让 reviewer 相信方法不是 benchmark 偶然适配。",
        "至少加入跨数据集、不同 seed、参数敏感性或噪声/缺失条件之一。",
    ),
    "visualization_strategy": (
        "作者用图表把机制解释、定性结果和失败边界可视化，降低复杂方法的理解成本。",
        "好的可视化不是装饰，而是帮助 reviewer 看见方法为什么有效。",
        "图表顺序采用 framework overview -> quantitative table -> ablation -> qualitative/diagnostic figure。",
    ),
    "narrative_pacing": (
        "作者按痛点、gap、机制、证据的节奏推进，避免一开始堆技术细节。",
        "叙事节奏稳定会让 reviewer 在读方法前已经接受问题的重要性。",
        "Introduction 每段只推进一个逻辑层级：field need、existing gap、our insight、contributions。",
    ),
    "figure_order": (
        "作者用图表顺序控制理解路径，先给整体框架，再给结果证据，再解释机制。",
        "图表顺序就是 reviewer 的阅读路线图，顺序错会让证据显得碎。",
        "Figure 1 讲方法全貌，Table 1 讲主结果，后续图表分别支撑机制、稳定性和案例。",
    ),
    "motivation_progression": (
        "作者让 motivation 从领域痛点逐步收窄到一个可操作设计原则。",
        "递进式动机能把方法选择塑造成必然，而不是作者偏好。",
        "用 therefore / however / to address 串联每个设计决策，不让模块突然出现。",
    ),
    "claim_scaffolding": (
        "作者先铺设小 claim，再用主结果和消融托起大 claim，避免超出证据。",
        "claim 分层能减少 reviewer 对夸大贡献的反感。",
        "把 claim 分为 observation、mechanism、performance、generalization 四级，逐级给证据。",
    ),
    "failure_hiding": (
        "作者通过 scope boundary、future work 和补充实验管理弱点，而不是正面放大失败。",
        "弱点不消失，但可以被定位为适用边界，避免破坏主 claim。",
        "对弱点使用：承认边界 -> 给出原因 -> 说明不影响核心 claim -> 放入 future work。",
    ),
    "reviewer_persuasion": (
        "作者用公平设置、强对比、机制解释和限制讨论组合说服 reviewer。",
        "reviewer persuasion 依赖证据矩阵，而不是单个高分表格。",
        "准备一张 claim-evidence 矩阵，确保每个可能 concern 都有对应实验或说明。",
    ),
    "trend_sensing": (
        "作者选择的问题贴近近期技术趋势，但落点仍是一个明确任务瓶颈。",
        "跟趋势能提高关注度，落到瓶颈能避免被认为只是追热点。",
        "判断趋势题：近期方法密集出现、benchmark 仍有争议、现有路线存在共同短板。",
    ),
    "sota_evolution": (
        "作者把 SOTA 演化写成路线竞争，借此说明自己的设计位于下一步自然位置。",
        "SOTA 叙事如果能解释路线为何演化，novelty 会更像研究判断。",
        "按 paradigm timeline 写 related work：传统 -> 深度 -> attention/sequence -> 当前缺口。",
    ),
    "field_hotspot": (
        "作者利用领域热点提供入口，但把贡献落在可迁移机制而非热点名词本身。",
        "热点只能带来关注，机制和证据才决定能否发表。",
        "热点包装要回答：为什么这个热点适合该任务、相比已有热点用法差异在哪里。",
    ),
    "benchmark_lifecycle": (
        "作者选择仍有解释空间的 benchmark，而不是只在饱和榜单上追小数点提升。",
        "benchmark 生命周期决定实验说服力，过饱和数据集需要更多跨集和机制证据。",
        "优先选择：社区熟悉、协议稳定、仍暴露真实缺陷、能支撑定性分析的 benchmark。",
    ),
    "innovation_density": (
        "作者控制 innovation density，让论文有一个主创新和少量支撑设计，避免堆砌。",
        "创新密度过高会增加 reviewer 理解负担，过低又显得 incremental。",
        "一篇 paper 只押一个核心 insight，其余模块服务于证据闭环和实现完整性。",
    ),
}


class ResearchExperienceAgent:
    def __init__(
        self,
        project_root: Path,
        workspace_dir: Path,
        *,
        vault_client: Optional[HyperVaultClient] = None,
        llm_store: Optional[LLMProviderStore] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.workspace_dir = workspace_dir.resolve()
        self.vault = vault_client or HyperVaultClient()
        self.llm_store = llm_store or LLMProviderStore(self.workspace_dir)
        self.llm_client = llm_client or LLMClient(timeout_sec=90)

    def extract(
        self,
        paper: str,
        *,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
        title: str = "",
        venue: str = "",
        year: Optional[int] = None,
        write: bool = True,
    ) -> PaperStrategyCard:
        source_path, text, metadata = self._load_paper_context(paper)
        card = self._extract_with_llm(
            paper,
            source_path,
            text,
            metadata,
            provider=provider,
            model=model,
            field=field,
            title=title,
            venue=venue,
            year=year,
        )
        if card is None:
            card = self._heuristic_card(
                paper,
                source_path,
                text,
                metadata,
                field=field,
                title=title,
                venue=venue,
                year=year,
            )
        card.warnings.extend(card.validation_warnings())
        card.warnings = sorted(set(card.warnings))
        if write:
            path = self.write_strategy_card(card)
            card.warnings.append("strategy card written: %s" % path)
        return card

    def extract_section(
        self,
        section_type: str,
        paper: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        card = self.extract(paper, **kwargs)
        section = getattr(card, section_type)
        return {
            "paper_id": card.paper_id,
            "title": card.title,
            "section_type": section_type,
            "section": section.to_dict(),
            "warnings": card.warnings,
        }

    def search_dimension(
        self,
        dimension: str,
        query: str,
        *,
        top_k: int = 8,
        field: str = "",
        verified: Optional[bool] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "type": "paper_strategy",
            "dimension": dimension,
        }
        if field:
            filters["field"] = field
        if verified is not None:
            filters["verified"] = verified
        hits = self.vault.search(query or dimension, top_k=top_k, filters=filters)
        return {
            "dimension": dimension,
            "query": query,
            "hits": [hit.to_dict() for hit in hits],
            "source": "hypervault",
        }

    def compare_paper_strategies(
        self,
        papers: Iterable[str],
        *,
        provider: str = "",
        model: Optional[str] = None,
        field: str = "",
    ) -> Dict[str, Any]:
        cards = [
            self.extract(
                paper,
                provider=provider,
                model=model,
                field=field,
                write=False,
            )
            for paper in papers
        ]
        dimension_counts: Dict[str, int] = {dimension: 0 for dimension in all_research_experience_dimensions()}
        for card in cards:
            for lesson in card.all_lessons():
                dimension_counts[lesson.dimension] = dimension_counts.get(lesson.dimension, 0) + 1
        common = [
            {"dimension": name, "paper_count": count}
            for name, count in sorted(dimension_counts.items(), key=lambda item: item[1], reverse=True)
            if count > 0
        ]
        return {
            "paper_count": len(cards),
            "common_dimensions": common,
            "cards": [card.to_dict() for card in cards],
            "warning": "Research Taste is more reliable when comparing multiple papers." if len(cards) < 2 else "",
        }

    def consolidate_research_experience(
        self,
        topic: str,
        *,
        papers: Optional[Iterable[str]] = None,
        field: str = "",
        provider: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        cards: List[PaperStrategyCard] = []
        if papers:
            cards = [
                self.extract(
                    paper,
                    provider=provider,
                    model=model,
                    field=field,
                    write=True,
                )
                for paper in papers
            ]
        hits = self.vault.search(topic, top_k=12, filters={"type": "paper_strategy", **({"field": field} if field else {})})
        lessons: List[StrategyLesson] = []
        for card in cards:
            lessons.extend(card.all_lessons())
        if not lessons:
            lessons.extend(self._lessons_from_hits(hits))
        slug = slugify(topic) or "research-experience"
        relative = "memory/research-experience/%s.md" % slug
        content = self._memory_markdown(topic, lessons, hits, field=field)
        path = self.vault.write_markdown(relative, content)
        return {
            "topic": topic,
            "path": str(path),
            "lesson_count": len(lessons),
            "source_hits": [hit.to_dict() for hit in hits],
        }

    def write_strategy_card(self, card: PaperStrategyCard) -> Path:
        relative = "summaries/research-strategies/%s.md" % (slugify(card.paper_id) or "paper-strategy")
        return self.vault.write_markdown(relative, self._card_markdown(card))

    def _load_paper_context(self, paper: str) -> Tuple[str, str, Dict[str, Any]]:
        local = self._read_local_paper(paper)
        if local is not None:
            return local
        hits = self.vault.search(paper, top_k=12, filters={"type": "paper"})
        if not hits:
            hits = self.vault.search(paper, top_k=12)
        if not hits:
            raise ValueError("paper could not be found locally or in HyperVault: %s" % paper)
        text = "\n\n".join(hit.text for hit in hits)
        metadata = dict(hits[0].metadata) if hits else {}
        return hits[0].file_path, text, metadata

    def _read_local_paper(self, paper: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        raw = Path(paper)
        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((self.project_root / raw).resolve())
            candidates.append((Path.cwd() / raw).resolve())
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".md", ".txt"}:
                text = candidate.read_text(encoding="utf-8", errors="replace")
                metadata, body = parse_markdown_metadata(text)
                return str(candidate), body, metadata
        vault_note = self.vault.read_markdown(paper)
        if vault_note is not None:
            relative, text = vault_note
            metadata, body = parse_markdown_metadata(text)
            return relative, body, metadata
        return None

    def _extract_with_llm(
        self,
        paper: str,
        source_path: str,
        text: str,
        metadata: Dict[str, Any],
        *,
        provider: str,
        model: Optional[str],
        field: str,
        title: str,
        venue: str,
        year: Optional[int],
    ) -> Optional[PaperStrategyCard]:
        provider_name = provider.strip()
        if not provider_name:
            return None
        try:
            self.llm_store.ensure_defaults()
            spec = self.llm_store.get(provider_name)
        except Exception:
            return None
        prompt = strategy_extraction_prompt(text[:24000], source_path=source_path)
        response = self.llm_client.send(
            spec,
            [
                LLMMessage(role="system", content=RESEARCH_EXPERIENCE_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        if response.warnings or not response.content.strip():
            return None
        try:
            parsed = json.loads(strip_json_fence(response.content))
            if "paper_strategy_card" in parsed:
                parsed = parsed["paper_strategy_card"]
            card = PaperStrategyCard.from_dict(parsed)
        except Exception:
            return None
        card.paper_id = card.paper_id or paper_id_from(source_path, metadata)
        card.title = card.title or title or str(metadata.get("title") or infer_title(text, source_path))
        card.field = field or card.field or str(metadata.get("field", ""))
        card.venue = venue or card.venue or str(metadata.get("venue", ""))
        card.year = year if year is not None else card.year
        card.source_paths = card.source_paths or [source_path]
        card.extraction_mode = "llm"
        card.created_at = card.created_at or utc_now()
        if not card.research_taste.lessons():
            card.warnings.append("Research Taste is underdetermined from one paper; consolidate across multiple papers.")
        return card

    def _heuristic_card(
        self,
        paper: str,
        source_path: str,
        text: str,
        metadata: Dict[str, Any],
        *,
        field: str,
        title: str,
        venue: str,
        year: Optional[int],
    ) -> PaperStrategyCard:
        paper_id = paper_id_from(source_path, metadata) or slugify(paper)
        resolved_title = title or str(metadata.get("title") or infer_title(text, source_path))
        resolved_venue = venue or str(metadata.get("venue", ""))
        resolved_field = field or str(metadata.get("field", ""))
        resolved_year = year
        if resolved_year is None and metadata.get("year") not in {None, ""}:
            try:
                resolved_year = int(metadata.get("year"))
            except (TypeError, ValueError):
                resolved_year = None
        lessons = {dimension: self._lesson_for_dimension(dimension, text, source_path) for dimension in all_research_experience_dimensions()}
        research_pattern = ResearchPattern(
            novelty_construction=[lessons["novelty_construction"]],
            problem_framing=[lessons["problem_framing"]],
            gap_definition=[lessons["gap_definition"]],
            contribution_packaging=[lessons["contribution_packaging"]],
            reviewer_expectation=[lessons["reviewer_expectation"]],
        )
        experiment_strategy = ExperimentStrategy(
            baseline_selection=[lessons["baseline_selection"]],
            ablation_logic=[lessons["ablation_logic"]],
            control_variable=[lessons["control_variable"]],
            robustness_validation=[lessons["robustness_validation"]],
            visualization_strategy=[lessons["visualization_strategy"]],
        )
        storytelling = ScientificStorytelling(
            narrative_pacing=[lessons["narrative_pacing"]],
            figure_order=[lessons["figure_order"]],
            motivation_progression=[lessons["motivation_progression"]],
            claim_scaffolding=[lessons["claim_scaffolding"]],
            failure_hiding=[lessons["failure_hiding"]],
            reviewer_persuasion=[lessons["reviewer_persuasion"]],
        )
        taste = ResearchTaste()
        return PaperStrategyCard(
            paper_id=paper_id,
            title=resolved_title,
            venue=resolved_venue,
            year=resolved_year,
            field=resolved_field,
            research_pattern=research_pattern,
            experiment_strategy=experiment_strategy,
            scientific_storytelling=storytelling,
            research_taste=taste,
            verified=False,
            evidence_level="chunk",
            source_paths=[source_path],
            extraction_mode="heuristic",
            created_at=utc_now(),
            warnings=["Research Taste requires multiple papers or a field timeline; single-paper taste extraction was skipped."],
        )

    def _lesson_for_dimension(self, dimension: str, text: str, source_path: str) -> StrategyLesson:
        claim, why, template = STRATEGY_TEMPLATES[dimension]
        quote = best_evidence_sentence(text, DIMENSION_QUERIES.get(dimension, [])) or first_nonempty_sentence(text)
        return StrategyLesson(
            dimension=dimension,
            strategy_claim=claim,
            why_it_works=why,
            evidence_span=EvidenceSpan(
                source_path=source_path,
                quote=quote[:900],
                role=dimension,
                confidence=0.62 if quote else 0.35,
            ),
            transferable_template=template,
            risk_or_limit="该经验来自自动提炼，需要人工确认；如果证据只来自单篇论文，应避免泛化成领域规律。",
            confidence=0.62 if quote else 0.35,
            tags=[dimension],
        )

    def _lessons_from_hits(self, hits: Iterable[HyperVaultHit]) -> List[StrategyLesson]:
        lessons: List[StrategyLesson] = []
        for hit in hits:
            for dimension in all_research_experience_dimensions():
                if dimension in hit.text or dimension in str(hit.metadata.get("strategy_dimensions", [])):
                    lessons.append(self._lesson_for_dimension(dimension, hit.text, hit.file_path))
                    break
        return lessons[:20]

    def _card_markdown(self, card: PaperStrategyCard) -> str:
        dimensions = sorted({lesson.dimension for lesson in card.all_lessons()})
        frontmatter = {
            "type": "paper_strategy",
            "paper_id": card.paper_id,
            "title": card.title,
            "venue": card.venue,
            "year": card.year,
            "field": card.field,
            "verified": card.verified,
            "evidence_level": card.evidence_level,
            "strategy_dimensions": dimensions,
            "source_paths": card.source_paths,
            "tags": ["research-experience", "paper-strategy"],
        }
        lines = ["---", to_yaml_like(frontmatter), "---", "", "# Research Strategy - %s" % card.title, ""]
        lines.append("> This note captures transferable research experience, not paper-content summary.")
        lines.append("")
        for section_name, section in [
            ("Research Pattern", card.research_pattern),
            ("Experiment Strategy", card.experiment_strategy),
            ("Scientific Storytelling", card.scientific_storytelling),
            ("Research Taste", card.research_taste),
        ]:
            lines.append("## %s" % section_name)
            section_lessons = section.lessons()
            if not section_lessons:
                lines.append("- insufficient evidence; consolidate across multiple papers.")
                lines.append("")
                continue
            for lesson in section_lessons:
                lines.extend(lesson_markdown(lesson))
            lines.append("")
        lines.append("## Machine-Readable Card")
        lines.append("```json")
        lines.append(json.dumps(card.to_dict(), ensure_ascii=False, indent=2))
        lines.append("```")
        return "\n".join(lines).rstrip() + "\n"

    def _memory_markdown(
        self,
        topic: str,
        lessons: List[StrategyLesson],
        hits: List[HyperVaultHit],
        *,
        field: str,
    ) -> str:
        dimensions = sorted({lesson.dimension for lesson in lessons})
        frontmatter = {
            "type": "research_experience_memory",
            "topic": topic,
            "field": field,
            "verified": False,
            "evidence_level": "strategy_card",
            "strategy_dimensions": dimensions,
            "tags": ["memory", "research-experience", "strategy"],
        }
        lines = ["---", to_yaml_like(frontmatter), "---", "", "# Research Experience - %s" % topic, ""]
        lines.append("## Transferable Patterns")
        if not lessons:
            lines.append("- No strategy lessons found yet. Add or extract paper strategy cards first.")
        for lesson in lessons[:30]:
            lines.extend(lesson_markdown(lesson))
        lines.append("")
        lines.append("## Sources")
        for hit in hits[:12]:
            lines.append("- [[%s]] score=%.3f" % (hit.file_path, hit.score))
        return "\n".join(lines).rstrip() + "\n"


RESEARCH_EXPERIENCE_SYSTEM_PROMPT = """You extract research experience, not paper summaries.
Return one JSON object that matches PaperStrategyCard. Every lesson must include:
strategy_claim, why_it_works, evidence_span, transferable_template, risk_or_limit, confidence.
Focus on author strategy: novelty construction, problem framing, baseline logic,
ablation logic, storytelling, reviewer persuasion, weak-point management, and research taste.
Do not write sentences like 'the paper proposes method X' unless you explain how the authors package or justify it.
If evidence is insufficient, return an empty list for that dimension and add a warning.
Research Taste requires multiple papers; for one paper, mark it as underdetermined.
"""


def strategy_extraction_prompt(text: str, *, source_path: str) -> str:
    return (
        "Source: %s\n\n"
        "Extract transferable research strategy from the paper text below. "
        "Do not summarize method content. Bind every claim to evidence.\n\n%s"
    ) % (source_path, text)


def strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def best_evidence_sentence(text: str, keywords: Iterable[str]) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    lowered_keywords = [keyword.lower() for keyword in keywords]
    best = ""
    best_score = 0
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(1 for keyword in lowered_keywords if keyword in lowered)
        if score > best_score:
            best = sentence
            best_score = score
    return best


def first_nonempty_sentence(text: str) -> str:
    for sentence in SENTENCE_RE.split(text):
        stripped = sentence.strip()
        if stripped:
            return stripped[:900]
    return ""


def infer_title(text: str, source_path: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return Path(source_path).stem.replace("-", " ").replace("_", " ").title()


def paper_id_from(source_path: str, metadata: Dict[str, Any]) -> str:
    return slugify(str(metadata.get("paper_id") or metadata.get("id") or Path(source_path).stem))


def slugify(value: str) -> str:
    slug = NON_SLUG_RE.sub("-", str(value).strip().lower()).strip("-._")
    return slug[:120]


def to_yaml_like(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, value in data.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            lines.append("%s: %s" % (key, "true" if value else "false"))
        elif isinstance(value, list):
            lines.append("%s:" % key)
            for item in value:
                lines.append("  - %s" % item)
        else:
            lines.append("%s: %s" % (key, json.dumps(value, ensure_ascii=False) if isinstance(value, str) and (":" in value or "#" in value) else value))
    return "\n".join(lines)


def lesson_markdown(lesson: StrategyLesson) -> List[str]:
    return [
        "### %s" % lesson.dimension,
        "- Strategy claim: %s" % lesson.strategy_claim,
        "- Why it works: %s" % lesson.why_it_works,
        "- Transferable template: %s" % lesson.transferable_template,
        "- Risk or limit: %s" % lesson.risk_or_limit,
        "- Evidence: `%s`" % lesson.evidence_span.quote.replace("`", "'")[:500],
        "- Source: %s" % lesson.evidence_span.source_path,
        "- Confidence: %.2f" % lesson.confidence,
    ]
