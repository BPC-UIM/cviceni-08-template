"""Nacitani jednoho priznaku datasetu Breast Cancer Wisconsin pro cviceni 08.

Modul poskytuje jedinou funkci ``load_1d_feature``, ktera z datasetu vezme
**jediny** sloupec jako 1D priznak -- na rozdil od cviceni 05-07 (kde se
pouzivala cela priznakova matice), tady jde o demonstraci 1D logisticke
regrese jako specialniho pripadu sigmoidoveho neuronu (blok 4 pipeline):
jeden vstup, jedna vaha, jeden bias, prah ``x0 = -bias / weight`` na ose x.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_1d_feature(feature_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Nacte dataset Breast Cancer Wisconsin a vrati jeden priznak ``(x, y)``.

    Parametry
    ---------
    feature_index:
        Index sloupce (0 az 29) v ``dataset.data``, ktery se pouzije jako
        1D priznak ``x``. Vychozi ``0`` odpovida priznaku ``mean radius``,
        ktery tridy v 1D rozumne oddeluje (velke nadory maji vetsi stredni
        polomer) -- pro ucely demonstrace prahu logisticke regrese v bloku
        4 pipeline je to rozumna vychozi volba, i kdyz zdaleka ne dokonale
        separujici (v 1D se tridy vzdy trochu prekryvaji).

    Navratova hodnota
    -----------------
    x:
        ``np.ndarray`` tvaru ``(569,)`` typu ``float64`` -- hodnoty
        vybraneho priznaku, priznaky se **nestandardizuji**.
    y:
        ``np.ndarray`` tvaru ``(569,)`` typu ``int64`` s hodnotami
        ``{0, 1}``, kde **1 = maligni (zhoubny)** a **0 = benigni
        (nezhoubny)**.

    Poznamka ke kodovani cilove promenne
    ------------------------------------
    ``sklearn.datasets.load_breast_cancer`` koduje ``target`` opacne, nez
    potrebujeme: 0 = malignant (zhoubny), 1 = benign (nezhoubny). Kurz
    vsak pracuje s konvenci "vystup 1 -> maligni" (shoda s cvicenim
    05-07), proto se stitky prohazuji vztahem ``y = 1 - dataset.target``.

    Vyjimky
    -------
    ``IndexError``:
        Pokud ``feature_index`` neni platny index sloupce datasetu (mimo
        rozsah ``0..29``) -- vyhozena primo numpy pri indexaci.
    """
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)[:, feature_index]
    # Prohozeni stitku: sklearn ma 0 = malignant, 1 = benign; kurz chce 1 = maligni.
    y = 1 - np.asarray(dataset.target, dtype=np.int64)

    return x, y
