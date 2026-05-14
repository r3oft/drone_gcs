"""
Compatibility shim for the vendored DroneKit source tree.

The repository keeps the upstream package under ``dronekit/dronekit``. When
commands run from the project root, Python sees the outer ``dronekit`` directory
first and would otherwise treat it as an empty namespace package. Execute the
inner package in this module namespace so ``import dronekit`` behaves like the
installed package.
"""

from pathlib import Path

_INNER_PACKAGE = Path(__file__).resolve().parent / "dronekit"
__path__ = [str(_INNER_PACKAGE)]

_INNER_INIT = _INNER_PACKAGE / "__init__.py"
with _INNER_INIT.open("rb") as _f:
    exec(compile(_f.read(), str(_INNER_INIT), "exec"), globals(), globals())
