"""Small JSON language-pack runtime for HyperAgent UI text."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hyperagent.core.io import read_yaml, write_json, write_yaml


DEFAULT_LOCALE = "zh-CN"
FALLBACK_LOCALE = "en"


@dataclass
class LanguagePack:
    locale: str
    translations: Dict[str, str]
    path: str
    source: str


class Translator:
    def __init__(
        self,
        locale: str,
        translations: Dict[str, str],
        fallback: Optional[Dict[str, str]] = None,
    ) -> None:
        self.locale = locale
        self.translations = dict(translations)
        self.fallback = dict(fallback or {})

    def t(self, key: str, default: Optional[str] = None, **kwargs: Any) -> str:
        template = self.translations.get(key)
        if template is None:
            template = self.fallback.get(key, default if default is not None else key)
        try:
            return str(template).format(**kwargs)
        except (KeyError, ValueError):
            return str(template)


class I18nStore:
    def __init__(self, project_root: Path = Path("."), workspace_dir: Optional[Path] = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.workspace_dir = (
            Path(workspace_dir)
            if workspace_dir is not None
            else self.project_root / ".hyperagent"
        )
        self.builtin_dir = Path(__file__).resolve().parents[1] / "i18n"
        self.user_dir = self.workspace_dir / "language_packs"
        self.config_path = self.workspace_dir / "config.yaml"

    def resolve_locale(self, argv: Optional[Iterable[str]] = None) -> str:
        cli_locale = extract_lang_arg(list(argv or []))
        if cli_locale:
            return cli_locale
        env_locale = os.environ.get("HYPERAGENT_LANG", "").strip()
        if env_locale:
            return env_locale
        config_locale = self._config_locale()
        if config_locale:
            return config_locale
        return DEFAULT_LOCALE

    def translator(self, locale: Optional[str] = None) -> Translator:
        selected = locale or DEFAULT_LOCALE
        fallback = self._load_translations(FALLBACK_LOCALE)
        translations = dict(fallback)
        if selected != FALLBACK_LOCALE:
            translations.update(self._load_translations(selected))
        return Translator(selected, translations, fallback=fallback)

    def list_packs(self) -> List[LanguagePack]:
        packs: Dict[str, LanguagePack] = {}
        for source, directory in (("builtin", self.builtin_dir), ("user", self.user_dir)):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                locale, translations = self._read_pack(path)
                packs[locale] = LanguagePack(
                    locale=locale,
                    translations=translations,
                    path=str(path),
                    source=source,
                )
        return [packs[key] for key in sorted(packs)]

    def install(self, path: Path) -> LanguagePack:
        locale, translations = self._read_pack(Path(path))
        self.user_dir.mkdir(parents=True, exist_ok=True)
        target = self.user_dir / f"{locale}.json"
        shutil.copyfile(path, target)
        return LanguagePack(
            locale=locale,
            translations=translations,
            path=str(target),
            source="user",
        )

    def export(self, locale: str, output: Path) -> Path:
        payload = {
            "locale": locale,
            "translations": self._load_translations(locale),
        }
        return write_json(Path(output), payload)

    def set_workspace_locale(self, locale: str) -> Path:
        if not self.config_path.exists():
            raise FileNotFoundError(
                "HyperAgent workspace is not initialized; run `HyperAgent init` first."
            )
        config = read_yaml(self.config_path)
        metadata = dict(config.get("metadata", {}))
        metadata["locale"] = locale
        config["metadata"] = metadata
        return write_yaml(self.config_path, config)

    def _config_locale(self) -> Optional[str]:
        if not self.config_path.exists():
            return None
        try:
            config = read_yaml(self.config_path)
        except OSError:
            return None
        metadata = config.get("metadata", {})
        if not isinstance(metadata, dict):
            return None
        value = str(metadata.get("locale", "")).strip()
        return value or None

    def _load_translations(self, locale: str) -> Dict[str, str]:
        translations: Dict[str, str] = {}
        for directory in (self.builtin_dir, self.user_dir):
            path = directory / f"{locale}.json"
            if path.exists():
                _, data = self._read_pack(path)
                translations.update(data)
        return translations

    def _read_pack(self, path: Path) -> tuple[str, Dict[str, str]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Language pack root must be an object: {path}")
        locale = str(data.get("locale") or Path(path).stem)
        raw = data.get("translations", data)
        if not isinstance(raw, dict):
            raise ValueError(f"Language pack translations must be an object: {path}")
        translations = {
            str(key): str(value)
            for key, value in raw.items()
            if key != "locale" and key != "translations"
        }
        return locale, translations


def extract_lang_arg(argv: List[str]) -> Optional[str]:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return None
        if token == "--lang" and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith("--lang="):
            return token.split("=", 1)[1]
        index += 1
    return None
