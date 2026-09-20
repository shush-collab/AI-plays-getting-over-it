"""Compatibility import for :mod:`aiget.game.worker.verify_physics_worker`."""
from ._compat import reexport

reexport(globals(), "aiget.game.worker.verify_physics_worker")
if __name__ == "__main__":
    globals()["main"]()
