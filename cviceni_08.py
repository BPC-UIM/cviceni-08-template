# -*- coding: utf-8 -*-

"""
Created on 13. 09. 2026 at 21:10:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Vstupni bod cviceni 08 (neuron jako linearni klasifikator). Pipeline
    projde sest fazi:

      1. hradla AND, OR, IMPLY (linearne separovatelna): Neuron slozeny
         z Linear + Step reprodukuje pravdivostni tabulku podle vah a biasu
         z config.yaml (sekce gates -- DOPLNUJE STUDENT), DecisionBoundaryPlotter
         ukaze rozhodovaci primku,
      1b. perzistence: Neuron.save/load pro hradla AND a OR -- model jako
          ulozene parametry (.npz),
      2. hradla XOR, XNOR (linearne NEseparovatelna): stejny druh neuronu
         na nich selze -- zamerny vysledek, ne chyba (dukaz sporem viz
         README, odd. 6),
      3. explorace: sweep jedne vahy ukazuje rotaci hranice, sweep teploty
         sigmoidy ukazuje prechod tvrda -> hladka hranice,
      4. 1D logisticka regrese: jednovstupy sigmoidovy neuron na jednom
         priznaku datasetu Breast Cancer Wisconsin, rucne zvoleny prah x0.

    Repozitar bezi v kazde fazi. Dokud nejsou ukoly hotove (src/activations.py,
    src/linear.py, src/neuron.py), faze se zastavi jen hlaskou o nedokoncenem
    ukolu (prefix "Ukol:") a pipeline pokracuje dal -- nikdy nezpracovanym
    tracebackem. Dokud student nedoplni vahy a bias hradel do config.yaml,
    faze 1/1b/2 se zastavi hlaskou [NENI VYPLNENO] a take pokracuji dal.
================================================================================
"""

from __future__ import annotations

import sys

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio import (
        DecisionBoundaryPlotter,
        ExperimentConfig,
        load_1d_feature,
        load_config,
        make_gate,
        plot_sigmoid_1d,
    )
    from src import Linear, Neuron, Sigmoid, Step
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"          # vystupni grafy (.png)
MODELS_DIR = "models"          # vystupni ulozene neurony (.npz)

# Linearne separovatelna hradla (faze 1) vs. linearne NEseparovatelna (faze 2).
SEPARABLE_GATES = ("and", "or", "imply")
INSEPARABLE_GATES = ("xor", "xnor")

# Hradla, u kterych se navic demonstruje ulozeni/nacteni modelu (faze 1b).
PERSISTENCE_GATES = ("and", "or")


def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def _config_neni_vyplnena(gate: str) -> None:
    """Vypise pratelskou hlasku, kdyz config.yaml jeste nema vahy hradla."""
    print(f"  [NENI VYPLNENO] gates.{gate}: weights/bias jsou v config.yaml "
          f"jeste null.")
    print("  -> Doplnte je podle vlastniho odvozeni (viz priklady_08.md); "
          "pipeline pokracuje dal.")


def faze_hradla(cfg: ExperimentConfig) -> dict[str, Neuron]:
    """Faze 1 -- linearne separovatelna hradla AND, OR, IMPLY.

    Vahy a bias pro kazde hradlo cte z ``cfg.gates`` (sekce ``gates`` v
    config.yaml) -- pipeline zadne "spravne" hodnoty sama nezna, doplnuje je
    student rucnim odvozenim (viz priklady_08.md, priklad 2).

    Navratova hodnota
    -----------------
    dict[str, Neuron]
        Neurony uspesne postavene z vyplnene konfigurace, klicovane jmenem
        hradla -- pouziva je faze 1b (perzistence), aby hradla nemusela
        stavet znovu.
    """
    _banner("Faze 1: Hradla AND, OR, IMPLY -- linearne separovatelna")

    neurony: dict[str, Neuron] = {}
    for gate in SEPARABLE_GATES:
        params = cfg.gates[gate]
        if params.weights is None:
            _config_neni_vyplnena(gate)
            continue

        x, y = make_gate(gate)
        neuron = Neuron(Linear(np.array(params.weights), params.bias), Step())
        neurony[gate] = neuron

        try:
            y_pred = neuron(x)
            acc = float(np.mean(np.asarray(y_pred).astype(int) == y))
            print(f"  {gate.upper():>5}: weights={params.weights}, bias={params.bias} "
                  f"-> presnost = {acc:.2f}")
        except NotImplementedError as exc:
            _faze_neni_hotova(exc)

        plotter = DecisionBoundaryPlotter(x, y)
        plotter.set_params(neuron, title=f"Hradlo {gate.upper()} (Step)")
        save_path = f"{GRAPHS_DIR}/hradlo_{gate}.png"
        plotter.show(save_path=save_path, mode="hard")
        print(f"         graf ulozen: {save_path}")

    return neurony


def faze_perzistence(neurony: dict[str, Neuron]) -> None:
    """Faze 1b -- model jako ulozene parametry: Neuron.save -> Neuron.load.

    Pro hradla AND a OR (pokud maji v konfiguraci vyplnene vahy) ulozi
    natvrdo navrzeny neuron do ``models/<hradlo>.npz`` a znovu ho nacte,
    aby overila, ze nacteny neuron dava stejne predikce jako puvodni.
    """
    _banner("Faze 1b: Perzistence -- Neuron.save / Neuron.load (.npz)")

    for gate in PERSISTENCE_GATES:
        neuron = neurony.get(gate)
        if neuron is None:
            print(f"  {gate.upper()}: preskakuji, config.yaml jeste nema vyplnene vahy.")
            continue

        x, _ = make_gate(gate)
        model_path = f"{MODELS_DIR}/{gate}.npz"
        try:
            neuron.save(model_path)
            obnoveny = Neuron.load(model_path)
            y_puvodni = np.asarray(neuron(x)).astype(int)
            y_obnoveny = np.asarray(obnoveny(x)).astype(int)
            shoda = bool(np.array_equal(y_puvodni, y_obnoveny))
            print(f"  {gate.upper()}: ulozeno do {model_path}, znovu nacteno.")
            print(f"        load(...).forward(x) == puvodni predikce:  {shoda}")
        except NotImplementedError as exc:
            _faze_neni_hotova(exc)


def faze_xor(cfg: ExperimentConfig) -> None:
    """Faze 2 -- linearne NEseparovatelna hradla XOR, XNOR (zamerny vysledek)."""
    _banner("Faze 2: Hradla XOR, XNOR -- mez jednoho neuronu")

    for gate in INSEPARABLE_GATES:
        params = cfg.gates[gate]
        if params.weights is None:
            _config_neni_vyplnena(gate)
            continue

        x, y = make_gate(gate)
        neuron = Neuron(Linear(np.array(params.weights), params.bias), Step())

        try:
            y_pred = neuron(x)
            acc = float(np.mean(np.asarray(y_pred).astype(int) == y))
            print(f"  {gate.upper():>4}: weights={params.weights}, bias={params.bias} "
                  f"-> presnost = {acc:.2f} (pod 1.00 pro KAZDOU volbu -- ocekavane)")
        except NotImplementedError as exc:
            _faze_neni_hotova(exc)

        plotter = DecisionBoundaryPlotter(x, y)
        plotter.set_params(neuron, title=f"Hradlo {gate.upper()} (jeden neuron nestaci)")
        save_path = f"{GRAPHS_DIR}/hradlo_{gate}.png"
        plotter.show(save_path=save_path, mode="hard")
        print(f"        graf ulozen: {save_path}")

    print("  Zadna volba (weights, bias) neda na XOR ani XNOR 100% presnost -- ")
    print("  oba jsou linearne neseparovatelne (dukaz sporem, viz README odd. 6).")
    print("  Toto NENI chyba k opravovani, je to motivace pro vicevrstve site (cv9+).")


def faze_sweep_vahy(cfg: ExperimentConfig) -> None:
    """Faze 3a -- sweep vahy w1 na hradle AND: rotace rozhodovaci hranice."""
    _banner("Faze 3a: Sweep vahy w1 (AND) -- rotace hranice")

    x, y = make_gate("and")
    plotter = DecisionBoundaryPlotter(x, y)
    sweep = cfg.exploration.weight_sweep
    hodnoty = np.arange(sweep["start"], sweep["stop"], sweep["step"])

    try:
        for i, w1 in enumerate(hodnoty):
            weights = np.array([float(w1), 1.0])
            neuron = Neuron(Linear(weights, -1.5), Step())
            plotter.set_params(
                neuron, title=f"AND: w1={float(w1):.2f}, w2=1.00, bias=-1.5"
            )
            save_path = f"{GRAPHS_DIR}/sweep_vaha_{i:02d}.png"
            plotter.show(save_path=save_path, mode="hard")
            print(f"  w1={float(w1):.2f} -> graf ulozen: {save_path}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        return


def faze_sweep_teploty(cfg: ExperimentConfig) -> None:
    """Faze 3b -- sweep teploty sigmoidy na hradle AND: tvrda -> hladka hranice."""
    _banner("Faze 3b: Sweep teploty sigmoidy (AND) -- tvrda -> hladka hranice")

    x, y = make_gate("and")
    plotter = DecisionBoundaryPlotter(x, y)

    try:
        for i, t in enumerate(cfg.exploration.temperature_sweep):
            neuron = Neuron(Linear(np.array([1.0, 1.0]), -1.5), Sigmoid(temperature=t))
            plotter.set_params(neuron, title=f"AND: Sigmoid, temperature={t}")
            save_path = f"{GRAPHS_DIR}/sweep_teplota_{i:02d}.png"
            plotter.show(save_path=save_path, mode="soft")
            print(f"  temperature={t} -> graf ulozen: {save_path}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        return


def faze_logisticka_regrese(cfg: ExperimentConfig) -> None:
    """Faze 4 -- 1D logisticka regrese jako specialni pripad sigmoidoveho neuronu."""
    _banner("Faze 4: 1D logisticka regrese (sigmoidovy neuron)")

    x, y = load_1d_feature(cfg.data.feature_index)

    # Deterministicka heuristika prahu a smeru/strmosti vahy (totez rucne
    # dela student na malem vzorku v priklady_08.md).
    x0 = 0.5 * (x[y == 0].mean() + x[y == 1].mean())          # stred mezi tridnimi prumery
    smer = 1.0 if x[y == 1].mean() > x[y == 0].mean() else -1.0
    strmost = 4.0 / (x.std() + 1e-9)                            # heuristicke meritko strmosti
    weight = np.array([smer * strmost])
    bias = -weight[0] * x0

    neuron = Neuron(Linear(weight, bias), Sigmoid(temperature=cfg.sigmoid.temperature))

    print(f"  Priznak index={cfg.data.feature_index}, x0={x0:.3f}, "
          f"weight={weight[0]:.4f}, bias={bias:.4f}")

    try:
        proba = neuron(x.reshape(-1, 1))
        pred = (np.asarray(proba) >= 0.5).astype(int)
        acc = float(np.mean(pred == y))
        print(f"  Presnost = {acc:.3f}, chybovost = {1.0 - acc:.3f}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    try:
        save_path = f"{GRAPHS_DIR}/logisticka_regrese_1d.png"
        plot_sigmoid_1d(x, y, neuron, threshold=x0, save_path=save_path)
        print(f"  graf ulozen: {save_path}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 08 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 08 -- Neuron jako linearni klasifikator -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    neurony_hradel = faze_hradla(cfg)
    faze_perzistence(neurony_hradel)
    faze_xor(cfg)
    faze_sweep_vahy(cfg)
    faze_sweep_teploty(cfg)
    faze_logisticka_regrese(cfg)

    _banner("CVICENI 08 -- konec")


if __name__ == "__main__":
    main()
