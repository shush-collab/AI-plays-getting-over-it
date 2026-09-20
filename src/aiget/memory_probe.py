"""Compatibility import for :mod:`aiget.game.probing.memory_probe`."""
from ._compat import reexport

reexport(globals(), "aiget.game.probing.memory_probe")
if __name__ == "__main__":
    globals()["main"]()
