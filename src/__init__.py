"""Verejne API balicku ``src`` pro cviceni 08.

Obsahuje:

* ``Activation`` (ABC) -- spolecne rozhrani rodiny aktivacnich funkci
  (``forward``, ``output_range``).
* ``Step`` / ``Sigmoid`` / ``Tanh`` / ``ReLU`` / ``ReLU6`` -- konkretni
  aktivace (Strategy vzor), kazda s vlastnim tvarem a oborem hodnot.
* ``make_activation`` -- tovarni funkce (Strategy + Factory): jmeno aktivace
  (retezec) -> instance ``Activation``. Pouziva ji ``Neuron.load``.
* ``Linear`` -- samostatna afinni vrstva (``X @ weights + bias``), bez
  aktivace.
* ``Neuron`` -- slozeni ``Linear`` a injektovane ``Activation``. Vahy i bias
  se nastavuji rucne, zadne uceni. Umi ulozit/nacist sam sebe (``save``/``load``).

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.activations import Activation, ReLU, ReLU6, Sigmoid, Step, Tanh, make_activation
from src.linear import Linear
from src.neuron import Neuron

__all__ = [
    "Activation",
    "Step",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "ReLU6",
    "make_activation",
    "Linear",
    "Neuron",
]
