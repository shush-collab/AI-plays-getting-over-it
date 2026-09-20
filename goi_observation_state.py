#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run the canonical observation-state probe."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from aiget.game.probing.observation_state import main as implementation

    implementation()

if __name__ == "__main__":
    main()
