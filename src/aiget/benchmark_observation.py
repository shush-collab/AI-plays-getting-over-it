"""Compatibility import for :mod:`aiget.rl.tools.benchmark_observation`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.benchmark_observation")
if __name__ == "__main__":
    globals()["main"]()
