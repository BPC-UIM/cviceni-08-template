# -*- coding: utf-8 -*-

"""
Created on 13. 09. 2026 at 21:40:00

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
    Testy pro cviceni 08 -- neuron jako linearni klasifikator.

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a jednotlive testy, ktere volaji nedokoncene
    ukoly (Activation.forward/output_range, Linear.forward, Neuron.save/load),
    se oznaci jako xfail (ocekavane selhani s NotImplementedError) -- sada
    nikdy neskonci holym tracebackem. Po dokonceni ukolu se z nich stanou
    xpass a nasledne plne prochazejici testy.

    Toto cviceni NEMA branu Distance -- neuron nepocita parove vzdalenosti,
    takze v testech neni zadna DummyDistance (neni co odpojovat).

    Pet trid testu:
      TestActivations    -- kazda aktivace na znamych vstupech + output_range
      TestLinear           -- afinni transformace Linear.forward, bias=None cesta, tvar
      TestNeuron           -- kompozice Linear+Activation, AND/OR pravdivostni tabulky
      TestSigmoidNeuron   -- jednovstupy sigmoidovy neuron: rozsah (0,1), monotonie
      TestPersistence      -- Neuron.save/load round-trip pres .npz
================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from dataio import make_gate
from src import Linear, Neuron, ReLU, ReLU6, Sigmoid, Step, Tanh

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")


# --------------------------------------------------------------------------- #
#  Aktivacni funkce                                                          #
# --------------------------------------------------------------------------- #
class TestActivations:
    """Kazda aktivace na znamych vstupech + spravnost output_range.

    Vsechny testy jsou STUB -- jak forward, tak output_range jsou u kazdeho
    potomka Activation jeste nedokoncene metody.
    """

    @STUB
    def test_step_zaporny_vstup_je_nula(self) -> None:
        vystup = Step()(np.array([-1.0]))
        assert np.array_equal(vystup, np.array([0.0]))

    @STUB
    def test_step_nula_je_jedna(self) -> None:
        vystup = Step()(np.array([0.0]))
        assert np.array_equal(vystup, np.array([1.0]))

    @STUB
    def test_step_kladny_vstup_je_jedna(self) -> None:
        vystup = Step()(np.array([5.0]))
        assert np.array_equal(vystup, np.array([1.0]))

    @STUB
    def test_sigmoid_v_nule_je_pul(self) -> None:
        proba = Sigmoid()(np.array([0.0]))
        assert proba[0] == pytest.approx(0.5)

    @STUB
    def test_sigmoid_nizka_teplota_se_blizi_step(self) -> None:
        """Male temperature ostri prechod -- Sigmoid se limitne blizi Step."""
        sigmoid = Sigmoid(temperature=0.001)
        vysoky_vstup = sigmoid(np.array([10.0]))
        nizky_vstup = sigmoid(np.array([-10.0]))
        assert vysoky_vstup[0] == pytest.approx(1.0, abs=1e-6)
        assert nizky_vstup[0] == pytest.approx(0.0, abs=1e-6)

    @STUB
    def test_tanh_v_nule_je_nula(self) -> None:
        assert Tanh()(np.array([0.0]))[0] == pytest.approx(0.0)

    @STUB
    def test_relu_zaporny_vstup_je_nula(self) -> None:
        assert ReLU()(np.array([-2.0]))[0] == pytest.approx(0.0)

    @STUB
    def test_relu_kladny_vstup_je_bezezmeny(self) -> None:
        assert ReLU()(np.array([3.0]))[0] == pytest.approx(3.0)

    @STUB
    def test_relu6_zaporny_vstup_je_nula(self) -> None:
        assert ReLU6()(np.array([-1.0]))[0] == pytest.approx(0.0)

    @STUB
    def test_relu6_stredni_vstup_je_bezezmeny(self) -> None:
        assert ReLU6()(np.array([3.0]))[0] == pytest.approx(3.0)

    @STUB
    def test_relu6_nad_stropem_je_orezano_na_sest(self) -> None:
        assert ReLU6()(np.array([10.0]))[0] == pytest.approx(6.0)

    @STUB
    def test_output_range_vsech_peti_aktivaci(self) -> None:
        assert Step().output_range == (0.0, 1.0)
        assert Sigmoid().output_range == (0.0, 1.0)
        assert Tanh().output_range == (-1.0, 1.0)
        assert ReLU().output_range == (0.0, np.inf)
        assert ReLU6().output_range == (0.0, 6.0)


# --------------------------------------------------------------------------- #
#  Linear (afinni vrstva, bez aktivace)                                      #
# --------------------------------------------------------------------------- #
class TestLinear:
    """Linear.forward pocita z = X @ weights + bias, nezavisle na aktivaci."""

    @STUB
    def test_z_je_vazeny_soucet_s_biasem(self) -> None:
        linear = Linear(np.array([1.0, 1.0]), bias=-0.5)
        z = np.asarray(linear(np.array([[0.0, 0.0]])))
        assert z[0] == pytest.approx(-0.5)

    @STUB
    def test_bias_none_neprida_zadny_posun(self) -> None:
        """bias=None znamena BEZ biasu (nepricita se), ne nejakou defaultni hodnotu.

        Bod [0, 0] s weights=[1, 1]: bez biasu je z=0, zatimco se stejnymi
        vahami a bias=-0.5 je z=-0.5. Rozdil ve vysledku overuje, ze
        bias=None skutecne nic nepricita.
        """
        bod = np.array([[0.0, 0.0]])
        bez_biasu = Linear(np.array([1.0, 1.0]), bias=None)
        s_biasem = Linear(np.array([1.0, 1.0]), bias=-0.5)
        assert np.asarray(bez_biasu(bod))[0] == pytest.approx(0.0)
        assert np.asarray(s_biasem(bod))[0] == pytest.approx(-0.5)

    @STUB
    def test_tvar_vystupu_odpovida_poctu_vzorku(self) -> None:
        X, _ = make_gate("and")
        linear = Linear(np.array([1.0, 1.0]), bias=-1.5)
        z = np.asarray(linear(X))
        assert z.shape == (4,)


# --------------------------------------------------------------------------- #
#  Neuron (kompozice Linear + Activation)                                    #
# --------------------------------------------------------------------------- #
class TestNeuron:
    """Neuron sklada Linear a injektovanou Activation; reprodukuje hradla."""

    def test_weights_bias_jsou_passthrough_na_linear(self) -> None:
        """__init__ a passthrough vlastnosti jsou PŘEDVYPLNĚNÉ, nejsou STUB."""
        linear = Linear(np.array([1.0, 2.0]), bias=-0.5)
        neuron = Neuron(linear, Step())
        assert neuron.weights is linear.weights
        assert neuron.bias == linear.bias

    @STUB
    def test_and_reprodukuje_pravdivostni_tabulku(self) -> None:
        X, y = make_gate("and")
        neuron = Neuron(Linear(np.array([1.0, 1.0]), -1.5), Step())
        pred = np.asarray(neuron(X)).astype(int)
        assert np.array_equal(pred, y)

    @STUB
    def test_or_reprodukuje_pravdivostni_tabulku(self) -> None:
        X, y = make_gate("or")
        neuron = Neuron(Linear(np.array([1.0, 1.0]), -0.5), Step())
        pred = np.asarray(neuron(X)).astype(int)
        assert np.array_equal(pred, y)

    @STUB
    def test_tvar_vystupu_odpovida_poctu_vzorku(self) -> None:
        X, y = make_gate("and")
        assert X.shape == (4, 2)
        neuron = Neuron(Linear(np.array([1.0, 1.0]), -1.5), Step())
        pred = np.asarray(neuron(X))
        assert pred.shape == (4,)


# --------------------------------------------------------------------------- #
#  Sigmoidovy neuron (pravdepodobnostni vystup)                              #
# --------------------------------------------------------------------------- #
class TestSigmoidNeuron:
    """Jednovstupy sigmoidovy neuron: vystup striktne v (0,1) a monotonie."""

    @STUB
    def test_vystup_je_ostre_mezi_nulou_a_jednickou(self) -> None:
        neuron = Neuron(Linear(np.array([2.0]), bias=-1.0), Sigmoid())
        x = np.array([[-5.0], [-1.0], [0.0], [1.0], [5.0]])
        proba = np.asarray(neuron(x))
        assert np.all(proba > 0.0)
        assert np.all(proba < 1.0)

    @STUB
    def test_monotonie_pro_rostouci_vstup(self) -> None:
        neuron = Neuron(Linear(np.array([2.0]), bias=-1.0), Sigmoid())
        x = np.array([[-5.0], [0.0], [5.0]])
        proba = np.asarray(neuron(x))
        assert np.all(np.diff(proba) >= 0.0)


# --------------------------------------------------------------------------- #
#  Perzistence (Neuron.save / Neuron.load, .npz)                             #
# --------------------------------------------------------------------------- #
class TestPersistence:
    """save -> load pres .npz (kontrakt: shodne predikce, spravna metadata)."""

    @STUB
    def test_round_trip_stejne_predikce(self, tmp_path) -> None:
        X, _ = make_gate("and")
        neuron = Neuron(Linear(np.array([1.0, 1.0]), -1.5), Step())
        p = tmp_path / "and.npz"
        neuron.save(str(p))
        obnoveny = Neuron.load(str(p))
        assert np.array_equal(
            np.asarray(neuron(X)).astype(int),
            np.asarray(obnoveny(X)).astype(int),
        )

    @STUB
    def test_ulozeny_soubor_je_platny_npz(self, tmp_path) -> None:
        """save vytvori soubor, ktery jde nacist jako .npz (ne pickle)."""
        neuron = Neuron(Linear(np.array([1.0, 1.0]), -0.5), Step())
        p = tmp_path / "or.npz"
        neuron.save(str(p))
        data = np.load(str(p))
        assert "weights" in data
        assert "activation_name" in data

    @STUB
    def test_bias_none_se_neuklada_jako_sentinel(self, tmp_path) -> None:
        """Nepritomnost klice 'bias' v archivu je signal bias=None, ne NaN."""
        neuron = Neuron(Linear(np.array([1.0]), bias=None), Step())
        p = tmp_path / "bez_biasu.npz"
        neuron.save(str(p))
        obnoveny = Neuron.load(str(p))
        assert obnoveny.bias is None

    @STUB
    def test_sigmoid_temperature_se_uklada_a_nacita(self, tmp_path) -> None:
        neuron = Neuron(Linear(np.array([2.0]), bias=-1.0), Sigmoid(temperature=0.1))
        p = tmp_path / "sigmoid.npz"
        neuron.save(str(p))
        obnoveny = Neuron.load(str(p))
        assert isinstance(obnoveny.activation, Sigmoid)
        assert obnoveny.activation.temperature == pytest.approx(0.1)
