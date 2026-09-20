"""Compatibility import for :mod:`aiget.rl.tools.random_rollout`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.random_rollout")
if __name__ == "__main__":
    globals()["main"]()
