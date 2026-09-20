"""Compatibility import for :mod:`aiget.game.probing.ptrace_il2cpp`."""
from ._compat import reexport

reexport(globals(), "aiget.game.probing.ptrace_il2cpp")
if __name__ == "__main__":
    globals()["main"]()
