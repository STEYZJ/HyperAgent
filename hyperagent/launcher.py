"""Console entry point for the Claude-Code-like HyperAgent command."""

import sys
from typing import Optional, Sequence

from hyperagent.cli import main as cli_main
from hyperagent.runtime.command_aliases import normalize_hyperagent_args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return cli_main(normalize_hyperagent_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
