"""Vykreslovani vysledku cviceni 08 -- vse PREDVYPLNENE.

Dve veci se kresli: rozhodovaci hranice neuronu ve 2D (hradla AND/OR/XOR a
exploracni sweep vah/teploty, trida ``DecisionBoundaryPlotter``) a
pravdepodobnostni krivka 1D sigmoidoveho neuronu s vyznacenym prahem
(``plot_sigmoid_1d``, blok logisticke regrese).

Hranice se kresli z **nove** konvence biasu -- ``Neuron.weights`` (vektor) a
volitelneho skalaru ``Neuron.bias`` -- nikoli ze stare tri-slozkove konvence
``[bias, w1, w2]`` pouzivane v drivejsich cvicenich. Aby modul sel importovat
i pred dokoncenim ``src/neuron.py``, typ ``Neuron`` je referencovan jen jako
retezcovy type hint pod ``TYPE_CHECKING`` -- ``dataio`` nesmi za behu
importovat ``src`` (zavislost jde jen ``src -> dataio``, nikdy obracene).

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy figuru zavrou. Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    # Jen pro typovou napovedu -- za behu se "src" z "dataio" NEIMPORTUJE
    # (cirkularni zavislost by navic nemela existovat: src -> dataio ano,
    # dataio -> src ne).
    from src.neuron import Neuron


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


class DecisionBoundaryPlotter:
    """Drzi 2D trenovaci data a opakovane kresli rozhodovaci hranici neuronu.

    Trida drzi priznakovou matici ``x`` a stitky ``y`` predane v
    konstruktoru; ``set_params`` pak nastavi (nebo aktualizuje) neuron a
    titulek, ktery se pouzije pri nasledujicim volani ``show``. Tento vzor
    (drz data jednou, opakovane men parametry a kresli) je vhodny pro
    exploracni sweep -- v cyklu se pro kazdou hodnotu vahy/teploty zavola
    ``set_params`` a hned ``show``.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Ulozi trenovaci data pro opakovane vykreslovani.

        Parametry
        ---------
        x:
            Priznakova matice tvaru ``(n, 2)`` -- presne dva priznaky,
            aby slo hranici zobrazit v rovine.
        y:
            Binarni stitky tvaru ``(n,)``.
        """
        self.x: np.ndarray = np.asarray(x, dtype=np.float64)
        self.y: np.ndarray = np.asarray(y)
        self.neuron: "Neuron | None" = None
        self.title: str = ""

    def set_params(self, neuron: "Neuron", title: str = "") -> None:
        """Nastavi (nebo aktualizuje) neuron a titulek pro nasledujici ``show``.

        Parametry
        ---------
        neuron:
            Instance ``Neuron``, jejiz ``weights``/``bias``/``activation``
            se pouziji pri vykresleni hranice.
        title:
            Titulek grafu. Prazdny retezec ponecha vychozi titulek podle
            zvoleneho rezimu v ``show``.
        """
        self.neuron = neuron
        self.title = title

    def show(self, save_path: str | None = None, mode: str = "hard") -> None:
        """Vykresli rozhodovaci hranici aktualne nastaveneho neuronu.

        Parametry
        ---------
        save_path:
            Cesta k vystupnimu PNG, nebo ``None`` (pak se figura jen
            zavre).
        mode:
            ``"hard"`` -- ostra hranice: dvoubarevne pozadi podle
            predikovane tridy (``z >= 0``) a kontura presne na ``z = 0``;
            vhodne pro ``Step``. ``"soft"`` -- spojity odstin podle
            vystupu aktivace (pravdepodobnostni gradient); vhodne pro
            ``Sigmoid``.

        Vyjimky
        -------
        ``ValueError``:
            Pokud nikdo predtim nezavolal ``set_params`` (``self.neuron``
            je ``None``), nebo pokud ``mode`` neni ``"hard"`` ani
            ``"soft"``.
        """
        if self.neuron is None:
            raise ValueError(
                "DecisionBoundaryPlotter.show() zavolan bez neuronu -- "
                "nejdrive zavolejte set_params(neuron, ...)."
            )
        if mode not in ("hard", "soft"):
            raise ValueError(
                f"Neznamy mode: {mode!r}. Povolene hodnoty jsou: 'hard', 'soft'."
            )

        neuron = self.neuron
        x, y = self.x, self.y

        x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        bias = neuron.bias if neuron.bias is not None else 0.0
        z = (grid @ neuron.weights + bias).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        if mode == "hard":
            predicted = (z >= 0.0).astype(np.float64)
            ax.contourf(xx, yy, predicted, alpha=0.25, cmap="coolwarm",
                        levels=[-0.5, 0.5, 1.5])
            ax.contour(xx, yy, z, colors="#555555", linewidths=1.2, levels=[0.0])
            default_title = "Rozhodovaci hranice (tvrda, z = 0)"
        else:  # mode == "soft"
            proba = neuron.activation.forward(z)
            contour = ax.contourf(xx, yy, proba, levels=20, cmap="coolwarm", alpha=0.8)
            ax.contour(xx, yy, z, colors="#555555", linewidths=1.0,
                       linestyles="--", levels=[0.0])
            fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04,
                         label="vystup aktivace")
            default_title = "Rozhodovaci hranice (hladka, vystup aktivace)"

        colors = {0: "#2ca02c", 1: "#d62728"}
        for cls in np.unique(y):
            mask = y == cls
            ax.scatter(x[mask, 0], x[mask, 1], s=40, alpha=0.9,
                       color=colors.get(int(cls), "#7f7f7f"), edgecolor="white",
                       linewidth=0.6, label=f"trida {int(cls)}", zorder=3)

        ax.set_xlabel("priznak 1")
        ax.set_ylabel("priznak 2")
        ax.set_title(self.title if self.title else default_title)
        ax.legend(loc="best")

        _save_and_close(fig, save_path)


def plot_sigmoid_1d(
    x: np.ndarray,
    y: np.ndarray,
    neuron: "Neuron",
    threshold: float | None = None,
    save_path: str | None = None,
) -> None:
    """Vykresli 1D pravdepodobnostni krivku sigmoidoveho neuronu.

    Parametry
    ---------
    x:
        1D priznak tvaru ``(n,)``.
    y:
        Binarni stitky tvaru ``(n,)`` (skutecna trida).
    neuron:
        Jednovstupovy ``Neuron`` (typicky se ``Sigmoid`` aktivaci), jehoz
        vystup pro rozsah hodnot ``x`` se vykresli jako spojita krivka.
    threshold:
        Volitelny rozhodovaci prah ``x0`` na ose x. Je-li zadan, vykresli
        se jako svisla carkovana cara s popiskem ``x0``.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None`` (pak se figura jen zavre).

    Krivka pokryva rozsah ``x`` o neco sirsi (5 % okraj na kazdou stranu),
    aby bylo videt i chovani mimo pozorovana data. Trenovaci body se
    vykresli na skutecnou hodnotu tridy (0 nebo 1) na ose y, obarvene
    podle tridy -- jejich poloha vuci krivce ukazuje, jak dobre neuron
    pravdepodobnost odhaduje.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)

    x_min, x_max = x.min(), x.max()
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_linspace = np.linspace(x_min - pad, x_max + pad, 300)
    proba = neuron(x_linspace.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(x_linspace, proba, color="#1f77b4", lw=2,
            label="vystup neuronu (pravdepodobnost)")

    colors = {0: "#2ca02c", 1: "#d62728"}
    labels = {0: "benigni (0)", 1: "maligni (1)"}
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(x[mask], np.full(int(mask.sum()), float(cls)), s=24, alpha=0.6,
                   color=colors.get(int(cls), "#7f7f7f"), edgecolor="white",
                   linewidth=0.3, label=labels.get(int(cls), f"trida {cls}"), zorder=3)

    if threshold is not None:
        ax.axvline(threshold, color="#555555", linestyle="--", linewidth=1.2,
                   label=f"x0 = {threshold:.4g}")

    ax.set_xlabel("priznak x")
    ax.set_ylabel("pravdepodobnost / trida")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("1D logisticka regrese: vystup sigmoidoveho neuronu")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    _save_and_close(fig, save_path)
