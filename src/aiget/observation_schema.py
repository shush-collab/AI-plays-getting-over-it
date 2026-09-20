"""Compatibility import for :mod:`aiget.rl.observation_schema`."""
from ._compat import reexport

reexport(globals(), "aiget.rl.observation_schema")
if __name__ == "__main__":
    globals()["main"]()
