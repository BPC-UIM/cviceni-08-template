"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 08.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a dve
funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje a
vrati) a ``validate_config`` (rozsahove kontroly s ceskymi chybovymi
hlaskami).

K hodnotam se pristupuje pres atributy (napr. ``cfg.data.feature_index``),
nikdy ne pres klice slovniku -- preklep v atributu odhali editor/typovy
kontroler staticky, zatimco ``cfg["data"]["feature_index"]`` je jen
dynamicke vyhledani v ``dict`` a preklep spadne az za behu.

Poznamka k ``ExplorationConfig.weight_sweep``: v ``config.yaml`` je to
mapovani s presne trojici klicu ``start``/``stop``/``step`` (vsechny
``float``). Zamerne se neobaluje do vlastni dataclass -- struktura zustava
1:1 s YAML a pristupuje se k ni jako ``cfg.exploration.weight_sweep["start"]``
apod.

Poznamka k ``ExperimentConfig.gates``: mapovani ``dict[str, GateParams]``
klicovane jmenem hradla (``"and"``, ``"or"``, ``"imply"``, ``"xor"``,
``"xnor"``). Dataclassa se stejnou logikou jako u ``weight_sweep`` vyse
nejde pouzit primo -- ``and``/``or`` jsou v Pythonu vyhrazena slova a nemohou
byt jmena atributu. Hodnoty ``weights``/``bias`` jsou v cerstvem ``config.yaml``
``null`` (Python ``None``) -- to je platny, ocekavany stav "student je jeste
nedoplnil", NE chyba; ``validate_config`` kontroluje jen rozsahy JIZ zadanych
hodnot, ne jejich uplnost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class GateParams:
    """Rucne navrzene vahy a bias jednoho hradla (jedna polozka sekce ``gates``).

    ``weights``/``bias`` jsou v cerstvem ``config.yaml`` ``None`` -- student
    je doplni sam podle vlastniho odvozeni (viz ``priklady_08.md``). Pipeline
    tento stav rozezna a jen vypise, ze konfiguraci je potreba doplnit, misto
    aby spadla.
    """

    weights: list[float] | None
    bias: float | None


@dataclass
class DataConfig:
    """Nastaveni vstupnich dat (sekce ``data``)."""

    feature_index: int
    random_state: int


@dataclass
class SigmoidConfig:
    """Nastaveni sigmoidove aktivace (sekce ``sigmoid``)."""

    temperature: float


@dataclass
class ExplorationConfig:
    """Nastaveni exploracniho sweepu (sekce ``exploration``).

    Atributy
    --------
    weight_sweep:
        Slovnik s klici ``"start"``, ``"stop"``, ``"step"`` (vsechny
        ``float``) -- rozsah hodnoty jedne vahy pro demonstraci rotace
        rozhodovaci hranice.
    temperature_sweep:
        Seznam kladnych teplot sigmoidy pro demonstraci prechodu
        tvrda -> hladka hranice.
    """

    weight_sweep: dict[str, float]
    temperature_sweep: list[float]


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    gates: dict[str, GateParams]
    data: DataConfig
    sigmoid: SigmoidConfig
    exploration: ExplorationConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``GateParams`` (slovnik),
    ``DataConfig``, ``SigmoidConfig`` a ``ExplorationConfig``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud nektera hodnota nesplnuje rozsahove kontroly ve
        ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    weight_sweep_raw = raw["exploration"]["weight_sweep"]
    gates_raw: dict[str, Any] = raw["gates"]
    cfg = ExperimentConfig(
        gates={
            name: GateParams(
                weights=(
                    [float(w) for w in params["weights"]]
                    if params["weights"] is not None
                    else None
                ),
                bias=(float(params["bias"]) if params["bias"] is not None else None),
            )
            for name, params in gates_raw.items()
        },
        data=DataConfig(
            feature_index=int(raw["data"]["feature_index"]),
            random_state=int(raw["data"]["random_state"]),
        ),
        sigmoid=SigmoidConfig(
            temperature=float(raw["sigmoid"]["temperature"]),
        ),
        exploration=ExplorationConfig(
            weight_sweep={
                "start": float(weight_sweep_raw["start"]),
                "stop": float(weight_sweep_raw["stop"]),
                "step": float(weight_sweep_raw["step"]),
            },
            temperature_sweep=[
                float(t) for t in raw["exploration"]["temperature_sweep"]
            ],
        ),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou obsahujici zadanou hodnotu. Kontroluji se:

    - kazda polozka ``gates`` s vyplnenymi ``weights`` ma prave 2 hodnoty
      (vsechna hradla v tomto cviceni maji 2 vstupy); ``weights=None`` a/nebo
      ``bias=None`` je platny "jeste nevyplneno" stav a NEKONTROLUJE se
    - ``data.feature_index`` je v rozsahu ``0..29`` (30 priznaku datasetu
      breast cancer)
    - ``data.random_state`` je typu ``int``
    - ``sigmoid.temperature > 0``
    - ``exploration.weight_sweep["start"] < exploration.weight_sweep["stop"]``
    - ``exploration.weight_sweep["step"] > 0``
    - ``exploration.temperature_sweep`` je neprazdny seznam a vsechny
      hodnoty jsou ``> 0``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    data = cfg.data
    sigmoid = cfg.sigmoid
    exploration = cfg.exploration

    for name, params in cfg.gates.items():
        if params.weights is not None and len(params.weights) != 2:
            raise ValueError(
                f"gates.{name}.weights musi mit presne 2 hodnoty (2 vstupy hradla), "
                f"zadano: {params.weights}"
            )

    if not 0 <= data.feature_index <= 29:
        raise ValueError(
            f"data.feature_index musi byt v rozsahu 0..29, zadano: {data.feature_index}"
        )
    if not isinstance(data.random_state, int):
        raise ValueError(
            f"data.random_state musi byt typu int, zadano: {data.random_state!r}"
        )
    if sigmoid.temperature <= 0:
        raise ValueError(
            f"sigmoid.temperature musi byt > 0, zadano: {sigmoid.temperature}"
        )

    weight_sweep = exploration.weight_sweep
    start = weight_sweep["start"]
    stop = weight_sweep["stop"]
    step = weight_sweep["step"]
    if not start < stop:
        raise ValueError(
            "exploration.weight_sweep['start'] musi byt mensi nez "
            f"weight_sweep['stop'], zadano: start={start}, stop={stop}"
        )
    if step <= 0:
        raise ValueError(
            f"exploration.weight_sweep['step'] musi byt > 0, zadano: {step}"
        )

    temperature_sweep = exploration.temperature_sweep
    if len(temperature_sweep) == 0:
        raise ValueError("exploration.temperature_sweep nesmi byt prazdny seznam")
    for t in temperature_sweep:
        if t <= 0:
            raise ValueError(
                f"exploration.temperature_sweep musi obsahovat jen kladne hodnoty, "
                f"zadano: {t}"
            )
