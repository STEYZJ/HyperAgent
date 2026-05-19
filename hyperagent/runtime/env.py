"""Local environment loading for secrets such as API keys."""

import os
from pathlib import Path
from typing import Dict


def load_env_file(path: Path = Path(".env"), override: bool = False) -> Dict[str, str]:
    """Load KEY=VALUE lines from a local .env file without logging secrets."""

    loaded: Dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded

