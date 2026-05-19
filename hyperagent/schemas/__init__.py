"""Schema dataclasses exchanged between decoupled HyperAgent modules."""

from hyperagent.schemas.agent import (
    AgentToolCall,
    AgentToolResult,
    CodingAgentRun,
    RepoFileContext,
    RepoSnapshot,
)
from hyperagent.schemas.dataset import DatasetAudit
from hyperagent.schemas.conversation import (
    ConversationMessage,
    ConversationSession,
    ConversationSummary,
)
from hyperagent.schemas.evaluation import EvaluationReport
from hyperagent.schemas.experiment import (
    ExperimentPlan,
    ExperimentResult,
    ModelConfig,
    PreprocessingConfig,
    SplitConfig,
)
from hyperagent.schemas.literature import LiteraturePaper, LiteratureSearchResult
from hyperagent.schemas.llm import (
    AgentTurnResult,
    LLMMessage,
    LLMProviderSpec,
    LLMRequest,
    LLMResponse,
)
from hyperagent.schemas.recommendation import ModelCandidate, ModelRecommendation
from hyperagent.schemas.integrations import (
    MaterializationResult,
    MCPServerSpec,
    ObsidianNote,
    PromptTemplate,
    SkillSpec,
)
from hyperagent.schemas.research import (
    AblationStudy,
    AblationVariant,
    AutoExperimentAgenda,
    DecisionRecord,
    EvidenceItem,
    ExperimentCandidate,
    ModuleProposal,
    ParameterProposal,
)
from hyperagent.schemas.runtime import ProjectConfig, ResearchTask, WorkspaceStatus
from hyperagent.schemas.spectral import SpectralReport

__all__ = [
    "DatasetAudit",
    "AblationStudy",
    "AblationVariant",
    "AgentTurnResult",
    "AgentToolCall",
    "AgentToolResult",
    "AutoExperimentAgenda",
    "CodingAgentRun",
    "ConversationMessage",
    "ConversationSession",
    "ConversationSummary",
    "DecisionRecord",
    "EvaluationReport",
    "EvidenceItem",
    "ExperimentCandidate",
    "ExperimentPlan",
    "ExperimentResult",
    "LiteraturePaper",
    "LiteratureSearchResult",
    "LLMMessage",
    "LLMProviderSpec",
    "LLMRequest",
    "LLMResponse",
    "MaterializationResult",
    "MCPServerSpec",
    "ModelCandidate",
    "ModelConfig",
    "ModuleProposal",
    "ModelRecommendation",
    "ObsidianNote",
    "ParameterProposal",
    "PreprocessingConfig",
    "PromptTemplate",
    "ProjectConfig",
    "RepoFileContext",
    "RepoSnapshot",
    "ResearchTask",
    "SkillSpec",
    "SpectralReport",
    "SplitConfig",
    "WorkspaceStatus",
]
