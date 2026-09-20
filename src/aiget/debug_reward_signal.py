"""Compatibility import for :mod:`aiget.rl.tools.debug_reward_signal`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.debug_reward_signal")
if __name__ == "__main__":
    globals()["main"]()
