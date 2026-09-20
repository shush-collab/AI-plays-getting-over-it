"""Compatibility import for :mod:`aiget.rl.tools.train_sac`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.train_sac")
if __name__ == "__main__":
    globals()["main"]()
