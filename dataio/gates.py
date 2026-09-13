"""Pravdivostni tabulky logickych hradel jako male 2D datasety.

Pipeline pracuje s peti hradly, rozdelenymi do dvou skupin podle toho, jestli
je jeden neuron dokaze oddelit primkou:

======  ================  ==========================
hradlo  vystup ``y``       linearne separovatelne?
======  ================  ==========================
AND     ``[0, 0, 0, 1]``  ano
OR      ``[0, 1, 1, 1]``  ano
IMPLY   ``[1, 1, 0, 1]``  ano (izoluje jediny roh)
XOR     ``[0, 1, 1, 0]``  **ne**
XNOR    ``[1, 0, 0, 1]``  **ne** (negace XOR, stejny dukaz sporem)
======  ================  ==========================

AND, OR a IMPLY jsou linearne separovatelne -- existuje primka/nadrovina,
ktera oddeli tridu 0 od tridy 1, a jeden neuron s vhodne navrzenymi vahami
a biasem (viz `config.yaml` sekce `gates`, doplnuje student) ji dokaze
realizovat presne. XOR a XNOR linearne separovatelne NEJSOU -- zadna primka
body nerozdeli, coz je zamerny (nikoli chybovy) vysledek pipeline a motivace
pro vicevrstve site (cviceni 09+).
"""

from __future__ import annotations

import numpy as np

# Ctyri kombinace vstupu v pevnem poradi; sdilene vsemi hradly.
_INPUTS: np.ndarray = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
)

# Pravdivostni tabulky (vystup pro poradi radku v ``_INPUTS``).
_TRUTH_TABLES: dict[str, list[int]] = {
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "xor": [0, 1, 1, 0],
    "xnor": [1, 0, 0, 1],
    "imply": [1, 1, 0, 1],
}


def make_gate(gate: str) -> tuple[np.ndarray, np.ndarray]:
    """Vrati ``(X, y)`` pro zadane logicke hradlo.

    Parametry
    ---------
    gate:
        Jmeno hradla z mnoziny ``{"and", "or", "imply", "xor", "xnor"}``.
        Velikost pismen ani okrajove mezery nevadi -- hodnota se normalizuje
        pres ``gate.lower().strip()``.

    Navratova hodnota
    -----------------
    X:
        ``np.ndarray`` tvaru ``(4, 2)`` typu ``float64`` se ctyrmi radky
        ``[0, 0]``, ``[0, 1]``, ``[1, 0]``, ``[1, 1]`` v tomto poradi.
    y:
        ``np.ndarray`` tvaru ``(4,)`` typu ``int64`` s vystupem hradla pro
        odpovidajici radky ``X``:

        - AND   -> ``[0, 0, 0, 1]``
        - OR    -> ``[0, 1, 1, 1]``
        - IMPLY -> ``[1, 1, 0, 1]``
        - XOR   -> ``[0, 1, 1, 0]``
        - XNOR  -> ``[1, 0, 0, 1]``

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``gate`` po normalizaci neni ``"and"``, ``"or"``, ``"imply"``,
        ``"xor"`` ani ``"xnor"``.
    """
    key = gate.lower().strip()
    if key not in _TRUTH_TABLES:
        povolene = ", ".join(sorted(_TRUTH_TABLES))
        raise ValueError(
            f"Nezname hradlo: {gate!r}. Povolene hodnoty jsou: {povolene}."
        )

    x = _INPUTS.copy()
    y = np.array(_TRUTH_TABLES[key], dtype=np.int64)
    return x, y
