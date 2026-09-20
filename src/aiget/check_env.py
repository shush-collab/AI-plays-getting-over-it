"""Compatibility import for :mod:`aiget.rl.tools.check_env`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.tools.check_env")
if __name__ == "__main__":
    globals()["main"]()
