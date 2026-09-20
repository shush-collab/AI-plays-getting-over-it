"""Compatibility import for :mod:`aiget.rl.tools.test_reset`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.test_reset")
if __name__ == "__main__":
    globals()["main"]()
