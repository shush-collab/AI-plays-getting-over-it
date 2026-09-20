#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run the canonical IL2CPP ptrace probe."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from aiget.game.probing.ptrace_il2cpp import main as implementation

    implementation()

if __name__ == "__main__":
    main()
