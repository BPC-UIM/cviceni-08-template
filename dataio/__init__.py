"""Verejne API balicku ``dataio`` pro cviceni 08.

Pravdivotni tabulky logickych hradel (AND, OR, XOR) jako male 2D datasety,
nacteni jednoho priznaku datasetu Breast Cancer Wisconsin, typovana sprava
konfigurace a vykreslovani rozhodovaci hranice neuronu.

Verejne API
-----------
- ``make_gate`` -- ``(X, y)`` pro logicke hradlo z
  ``{"and", "or", "imply", "xor", "xnor"}``
- ``load_1d_feature`` -- ``(x, y)`` jednoho priznaku datasetu breast cancer
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``GateParams`` / ``DataConfig`` / ``SigmoidConfig`` / ``ExplorationConfig`` /
  ``ExperimentConfig`` -- dataclassy
- ``DecisionBoundaryPlotter`` -- opakovane vykresleni 2D rozhodovaci hranice neuronu
- ``plot_sigmoid_1d`` -- pravdepodobnostni krivka 1D sigmoidoveho neuronu s prahem

Cely balicek ``dataio/`` je v tomto cviceni **predvyplneny** -- zadny
studentsky ukol, zadny ``NotImplementedError``. Studentske ukoly cviceni 08
zijou pouze v ``src/activations.py`` a ``src/neuron.py``.

**Tento soubor (__init__.py) neupravujte** -- re-exporty verejneho API
zustavaji beze zmeny.
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    ExperimentConfig,
    ExplorationConfig,
    GateParams,
    SigmoidConfig,
    load_config,
    validate_config,
)
from dataio.gates import make_gate
from dataio.loader import load_1d_feature
from dataio.plotting import DecisionBoundaryPlotter, plot_sigmoid_1d

__all__ = [
    "make_gate",
    "load_1d_feature",
    "load_config",
    "validate_config",
    "GateParams",
    "DataConfig",
    "SigmoidConfig",
    "ExplorationConfig",
    "ExperimentConfig",
    "DecisionBoundaryPlotter",
    "plot_sigmoid_1d",
]
