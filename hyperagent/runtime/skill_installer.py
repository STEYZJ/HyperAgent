"""Third-party SKILL.md installer with preflight safety checks."""

import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hyperagent.runtime.repo_context import TEXT_SUFFIXES
from hyperagent.runtime.skills import SkillStore
from hyperagent.schemas import SkillSpec


SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
HIGH_RISK_TOOL_MARKERS = {
    "bash",
    "shell",
    "run_command",
    "execute",
    "write",
    "edit",
    "apply_patch",
    "network",
    "web",
    "web_fetch",
    "web_search",
    "download",
    "install",
    "training",
    "run_experiment",
}


@dataclass
class GitHubSkillSource:
    owner: str
    repo: str
    ref: str
    skill_path: str

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class SkillInstallPlan:
    source_type: str
    source: str
    source_path: str
    target_dir: str
    install_root: str
    skill_name: str
    description: str = ""
    run_as: str = "inline"
    allowed_tools: List[str] = field(default_factory=list)
    model: str = ""
    profile: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    existing: bool = False
    force: bool = False

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reasons)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["blocked"] = self.blocked
        return payload


@dataclass
class SkillInstallResult:
    status: str
    installed: bool
    plan: SkillInstallPlan
    skill: Optional[SkillSpec] = None
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "installed": self.installed,
            "message": self.message,
            "plan": self.plan.to_dict(),
            "skill": self.skill.to_dict() if self.skill is not None else None,
        }


class SkillInstaller:
    """Install local or GitHub-hosted SKILL.md directories without executing them."""

    def __init__(
        self,
        *,
        install_root: Optional[Path] = None,
        github_token: Optional[str] = None,
    ) -> None:
        self.install_root = Path(install_root) if install_root else default_user_skill_root()
        self.github_token = github_token if github_token is not None else (
            os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        )

    def install_from_path(
        self,
        path: Path,
        *,
        name: str = "",
        force: bool = False,
        dry_run: bool = False,
    ) -> SkillInstallResult:
        source = self._normalize_local_source(Path(path))
        plan = self._build_plan(
            source,
            source_type="local",
            source=str(Path(path)),
            name=name,
            force=force,
        )
        return self._install_from_plan(plan, dry_run=dry_run)

    def install_from_repo(
        self,
        repo: str,
        skill_path: str,
        *,
        ref: str = "main",
        name: str = "",
        force: bool = False,
        dry_run: bool = False,
    ) -> SkillInstallResult:
        owner, repo_name = parse_repo_slug(repo)
        source = GitHubSkillSource(
            owner=owner,
            repo=repo_name,
            ref=ref or "main",
            skill_path=skill_path.strip("/"),
        )
        with tempfile.TemporaryDirectory(prefix="hyperagent-skill-") as tmp:
            local_source = self._download_github_source(source, Path(tmp))
            plan = self._build_plan(
                local_source,
                source_type="github",
                source=f"{source.repo_slug}@{source.ref}:{source.skill_path}",
                name=name,
                force=force,
            )
            return self._install_from_plan(plan, dry_run=dry_run)

    def install_from_url(
        self,
        url: str,
        *,
        name: str = "",
        force: bool = False,
        dry_run: bool = False,
    ) -> SkillInstallResult:
        source = parse_github_skill_url(url)
        return self.install_from_repo(
            source.repo_slug,
            source.skill_path,
            ref=source.ref,
            name=name,
            force=force,
            dry_run=dry_run,
        )

    def plan_from_request(
        self,
        *,
        path: str = "",
        repo: str = "",
        skill_path: str = "",
        url: str = "",
        ref: str = "main",
        name: str = "",
        force: bool = False,
    ) -> SkillInstallResult:
        if path:
            return self.install_from_path(Path(path), name=name, force=force, dry_run=True)
        if url:
            return self.install_from_url(url, name=name, force=force, dry_run=True)
        if repo:
            return self.install_from_repo(
                repo,
                skill_path,
                ref=ref,
                name=name,
                force=force,
                dry_run=True,
            )
        raise ValueError("skill source is required: provide --path, --repo/--skill-path, or --url")

    def install_from_request(
        self,
        *,
        path: str = "",
        repo: str = "",
        skill_path: str = "",
        url: str = "",
        ref: str = "main",
        name: str = "",
        force: bool = False,
        dry_run: bool = False,
    ) -> SkillInstallResult:
        if path:
            return self.install_from_path(Path(path), name=name, force=force, dry_run=dry_run)
        if url:
            return self.install_from_url(url, name=name, force=force, dry_run=dry_run)
        if repo:
            return self.install_from_repo(
                repo,
                skill_path,
                ref=ref,
                name=name,
                force=force,
                dry_run=dry_run,
            )
        raise ValueError("skill source is required: provide --path, --repo/--skill-path, or --url")

    def _normalize_local_source(self, source: Path) -> Path:
        source = source.expanduser().resolve()
        if source.name == "SKILL.md" and source.is_file():
            return source
        if (source / "SKILL.md").is_file():
            return source
        raise FileNotFoundError(f"SKILL.md not found: {source}")

    def _build_plan(
        self,
        source_path: Path,
        *,
        source_type: str,
        source: str,
        name: str,
        force: bool,
    ) -> SkillInstallPlan:
        skill_file = source_path if source_path.name == "SKILL.md" else source_path / "SKILL.md"
        skill = SkillStore([skill_file.parent.parent])._load(  # local parse only; no discovery side effects
            skill_file,
            source=str(skill_file.parent.parent),
        )
        target_name = sanitize_skill_name(name.strip() or skill.name or skill_file.parent.name)
        target_dir = self.install_root / target_name
        scan_source = skill_file if source_path.name == "SKILL.md" else skill_file.parent
        warnings, blocked_reasons = self._scan_source(scan_source, skill)
        existing = target_dir.exists()
        if existing and not force:
            blocked_reasons.append(
                f"target skill already exists: {target_dir}; pass --force to overwrite"
            )
        return SkillInstallPlan(
            source_type=source_type,
            source=source,
            source_path=str(source_path),
            target_dir=str(target_dir),
            install_root=str(self.install_root),
            skill_name=target_name,
            description=skill.description,
            run_as=skill.run_as,
            allowed_tools=list(skill.allowed_tools),
            model=skill.model,
            profile=skill.profile,
            metadata=dict(skill.metadata),
            warnings=warnings,
            blocked_reasons=blocked_reasons,
            existing=existing,
            force=force,
        )

    def _install_from_plan(
        self,
        plan: SkillInstallPlan,
        *,
        dry_run: bool,
    ) -> SkillInstallResult:
        if dry_run:
            status = "blocked" if plan.blocked else "planned"
            return SkillInstallResult(
                status=status,
                installed=False,
                plan=plan,
                message="dry-run only; no files were written",
            )
        if plan.blocked:
            return SkillInstallResult(
                status="blocked",
                installed=False,
                plan=plan,
                message="preflight blocked installation",
            )

        source_path = Path(plan.source_path)
        target_dir = Path(plan.target_dir)
        self.install_root.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            if not plan.force:
                return SkillInstallResult(
                    status="blocked",
                    installed=False,
                    plan=plan,
                    message="target exists; pass --force to overwrite",
                )
            shutil.rmtree(target_dir)
        if source_path.name == "SKILL.md":
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_dir / "SKILL.md")
        else:
            shutil.copytree(source_path, target_dir)
        skill = SkillStore([self.install_root]).get(plan.skill_name)
        return SkillInstallResult(
            status="installed",
            installed=True,
            plan=plan,
            skill=skill,
            message=f"installed skill: {plan.skill_name}",
        )

    def _scan_source(self, source: Path, skill: SkillSpec) -> Tuple[List[str], List[str]]:
        warnings: List[str] = []
        blocked: List[str] = []
        high_risk = [
            tool for tool in skill.allowed_tools
            if any(marker in str(tool).lower() for marker in HIGH_RISK_TOOL_MARKERS)
        ]
        if high_risk:
            warnings.append("high-risk allowed-tools declared: " + ", ".join(high_risk))
        source_dir = source.parent if source.is_file() else source
        scripts_dir = source_dir / "scripts"
        if source.is_dir() and scripts_dir.exists():
            warnings.append(f"scripts directory present: {scripts_dir}")

        candidates = [source] if source.is_file() else list(source_dir.rglob("*"))
        for path in candidates:
            if path.is_dir():
                continue
            try:
                mode = path.stat().st_mode
            except OSError:
                continue
            if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                warnings.append(f"executable file present: {path.relative_to(source_dir)}")
            if self._looks_text(path):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if SECRET_RE.search(text):
                    blocked.append(f"possible API key found in {path.relative_to(source_dir)}")
        return warnings, blocked

    def _looks_text(self, path: Path) -> bool:
        return path.name == "SKILL.md" or path.suffix.lower() in TEXT_SUFFIXES

    def _download_github_source(self, source: GitHubSkillSource, tmp_root: Path) -> Path:
        try:
            return self._download_github_archive(source, tmp_root)
        except Exception:
            return self._download_github_sparse_checkout(source, tmp_root)

    def _download_github_archive(self, source: GitHubSkillSource, tmp_root: Path) -> Path:
        archive = tmp_root / "repo.zip"
        url = f"https://github.com/{source.repo_slug}/archive/{urllib.parse.quote(source.ref, safe='')}.zip"
        request = urllib.request.Request(url, headers=self._github_headers())
        with urllib.request.urlopen(request, timeout=30) as response:
            archive.write_bytes(response.read())
        extract_dir = tmp_root / "archive"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        roots = [item for item in extract_dir.iterdir() if item.is_dir()]
        if not roots:
            raise FileNotFoundError("GitHub archive did not contain a repository directory")
        candidate = roots[0] / source.skill_path
        return self._normalize_local_source(candidate)

    def _download_github_sparse_checkout(self, source: GitHubSkillSource, tmp_root: Path) -> Path:
        repo_dir = tmp_root / "repo"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{source.repo_slug}.git", str(repo_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", source.skill_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", source.ref],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        return self._normalize_local_source(repo_dir / source.skill_path)

    def _github_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/zip, application/vnd.github+json",
            "User-Agent": "HyperAgent-SkillInstaller",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers


def default_user_skill_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    return (Path(codex_home) / "skills") if codex_home else (Path.home() / ".codex" / "skills")


def sanitize_skill_name(name: str) -> str:
    clean = SAFE_NAME_RE.sub("-", str(name or "").strip()).strip("-")
    if not clean:
        raise ValueError("skill name is empty")
    return clean


def parse_repo_slug(repo: str) -> Tuple[str, str]:
    text = str(repo or "").strip().strip("/")
    parts = text.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("repo must be in owner/repo form")
    return parts[0], parts[1]


def parse_github_skill_url(url: str) -> GitHubSkillSource:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError("only github.com skill URLs are supported")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in {"tree", "blob"}:
        raise ValueError("GitHub URL must look like https://github.com/owner/repo/tree/ref/path/to/skill")
    owner, repo, _, ref = parts[:4]
    skill_path = "/".join(parts[4:])
    if not skill_path:
        raise ValueError("GitHub skill URL is missing a skill path")
    return GitHubSkillSource(owner=owner, repo=repo, ref=ref, skill_path=skill_path)


def format_install_plan(plan: SkillInstallPlan) -> str:
    lines = [
        f"skill: {plan.skill_name}",
        f"description: {plan.description or '-'}",
        f"run_as: {plan.run_as}",
        f"allowed_tools: {', '.join(plan.allowed_tools) if plan.allowed_tools else '-'}",
        f"source: {plan.source_type} {plan.source}",
        f"target: {plan.target_dir}",
        f"existing: {plan.existing}",
    ]
    if plan.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in plan.warnings)
    if plan.blocked_reasons:
        lines.append("blocked:")
        lines.extend(f"- {reason}" for reason in plan.blocked_reasons)
    return "\n".join(lines)


def format_install_result(result: SkillInstallResult) -> str:
    return (
        f"status: {result.status}\n"
        f"installed: {result.installed}\n"
        f"{format_install_plan(result.plan)}\n"
        f"message: {result.message}"
    )
