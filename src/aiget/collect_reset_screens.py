"""Compatibility import for :mod:`aiget.game.tools.collect_reset_screens`."""
from ._compat import reexport

reexport(globals(), "aiget.game.tools.collect_reset_screens")
if __name__ == "__main__":
    globals()["main"]()
