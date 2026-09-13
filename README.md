# Cvičení 8: Neuron jako lineární klasifikátor

Osmé praktické cvičení předmětu **Umělá inteligence v medicíně** navazuje na
blok modelů z Cvičení 05–07 tématem klasifikace, ale mění přístup k
parametrům: váhy a bias jednoho **umělého neuronu** se v tomto cvičení
**navrhují ručně**, z geometrie úlohy, nikoli se neučí z dat. Cílem je
pochopit, co neuron počítá a proč má rozhodovací hranice tvar nadroviny,
než se v Cvičení 09 přejde k iterativnímu dolaďování týchž parametrů.

Neuron je zde složen ze dvou nezávislých, samostatně testovatelných částí:
afinní vrstvy `Linear` (`z = X @ weights + bias`) a injektované aktivační
funkce `Activation`, kterou `Neuron` na výstup `Linear` aplikuje. Na
hradlech AND, OR a IMPLY jeden neuron s vhodně navrženými parametry
pravdivostní tabulku přesně reprodukuje; na hradlech XOR a XNOR to **nejde**
— a jde o matematický fakt, který motivuje vícevrstvé sítě, ne o chybu
k opravení. Na konci se ukáže, že **logistická regrese** je přesně tentýž
neuron se sigmoidovou aktivací.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Chápe neuron jako nadrovinu.** `z = w·x + bias` je lineární funkce
   vstupu; množina bodů `z = 0` je nadrovina, která prostor rozděluje na dvě
   poloviny, a **znaménko** `z` určuje, do které z nich bod padne. Váhy
   nadrovinu **natáčejí**, bias ji **posouvá** rovnoběžně se sebou samou.
2. **Umí odvodit rovnici hranice ve 2D ve směrnicovém tvaru** a rozumí
   vztahu mezi vektorem vah a normálovým vektorem hranice — ne jen
   kvalitativně, ale i jako přímý algebraický důsledek `z = w·x + bias = 0`.
3. **Umí navrhnout váhy a bias ručně, z geometrie.** Pro AND, OR a IMPLY
   odvodí z pravdivostní tabulky přímku, která třídy odděluje, a přímku
   zapíše jako dvojici `(weights, bias)` — žádné učení, jen geometrický
   návrh; hodnoty vyplní přímo do `config.yaml`.
4. **Rozumí znaménkové konvenci biasu.** Bias je **záporně vzatý práh**:
   `bias = -threshold`. Umí to odvodit z nerovnosti `w·x >= threshold`
   přepsané na `w·x - threshold >= 0`.
5. **Rozliší tvrdé a měkké rozhodnutí.** `Step` dá ostrou hranici 0/1;
   `Sigmoid` dá hladkou pravděpodobnost; parametr `temperature` mezi nimi
   plynule interpoluje. Zná i `Tanh`, `ReLU` a `ReLU6` jako další členy téže
   rodiny (Strategy vzor), včetně jejich oborů hodnot — s výhledem na to, že
   právě tyto funkce tvoří vnitřní vrstvy sítí v pozdějších cvičeních.
6. **Vidí mez jednoho neuronu na hradlech XOR a XNOR.** AND, OR a IMPLY jsou
   lineárně separovatelné, XOR a XNOR **nejsou** — žádná přímka je
   nerozdělí. Rozumí, proč je to přesně ten výsledek, který historicky
   (Minsky–Papert) motivoval přechod od jednoho neuronu k vícevrstvým sítím
   (Cvičení 09+).
7. **Rozpozná logistickou regresi jako speciální případ.** Jednovstupý neuron
   se sigmoidou **je** logistická regrese: výstup je pravděpodobnost
   příslušnosti ke třídě, ne tvrdá třída, a rozhodovací práh leží v bodě, kde
   sigmoida protíná 0,5, tedy `x₀ = -bias/weight`. Zde je práh navržen ručně.
8. **Skládá nezávislé vrstvy.** `Linear` (afinní transformace) a
   `Activation` jsou samostatné objekty, ne jeden monolitický neuron — stejný
   princip jako skládání `nn.Linear` a `nn.ReLU()` do sítě v PyTorch.
   Aktivace i afinní vrstva jsou **injektované** (dependency injection,
   stejný vzor jako `Distance`/`Initializer`/`Validator`/`Kernel` v
   předchozích cvičeních).
9. **Ukládá a načítá model.** `Neuron.save`/`Neuron.load` uloží váhy, bias
   a jméno použité aktivace do `.npz` a zpět je zrekonstruují — stejný návyk
   jako u induktivních modelů v Cvičení 05–07, i když se zde model nenaučí,
   ale navrhne.
10. **Pracuje s typovanou konfigurací** — čte hyperparametry z `config.yaml`
    přes dataclassy (`cfg.sigmoid.temperature`, `cfg.gates["and"]`), stejný
    vzor jako v Cvičení 03–07.

---

## Struktura repozitáře

```
cviceni-08-template/
├── cviceni_08.py            # Hlavní pipeline — spusťte pro průběžné ověření (PŘEDVYPLNĚNO)
├── config.yaml               # Konfigurace experimentu (YAML) — sekci gates DOPLŇTE
├── priklady_08.md             # Papírové (teoretické) příklady — BEZ řešení v repozitáři
├── requirements.txt           # Python závislosti (zamčené verze)
├── .gitignore
├── src/
│   ├── __init__.py           # Re-exporty balíčku (neupravujte)
│   ├── activations.py        # Activation (ABC) + Step/Sigmoid/Tanh/ReLU/ReLU6 — ÚKOL
│   ├── linear.py               # Linear — ÚKOL: forward(); __init__ a __call__ předvyplněny
│   └── neuron.py              # Neuron — ÚKOL: save()/load(); zbytek předvyplněn
├── dataio/
│   ├── __init__.py           # Re-exporty balíčku (neupravujte)
│   ├── gates.py                # make_gate() — hradla AND/OR/IMPLY/XOR/XNOR (předvyplněno)
│   ├── loader.py               # load_1d_feature() — jeden příznak breast-cancer (předvyplněno)
│   ├── config_manager.py     # Dataclassy + load_config + validate_config (předvyplněno)
│   └── plotting.py             # DecisionBoundaryPlotter + plot_sigmoid_1d (předvyplněno)
├── models/                    # Uložené neurony (.npz) — generuje se automaticky
│   └── .gitkeep
├── graphs/                    # Výstupní složka pro grafy (generuje se automaticky)
│   └── .gitkeep
└── test_cviceni_08.py         # Automatické testy (pytest)
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`,
> `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje
> veřejné API. Díky tomu lze psát `from src import Neuron` místo
> `from src.neuron import Neuron`. **Tyto soubory neupravujte.**

> **Balíček `dataio/` je v tomto cvičení předvyplněn celý** — generování
> hradel, načítání dat, konfigurace i vykreslování. Žádný `NotImplementedError`
> se v `dataio/` nevyskytuje. Veškerá vaše práce je ve dvou souborech:
> `src/activations.py` a `src/linear.py`, plus dvě metody `Neuron.save`/
> `Neuron.load`.

> **Model se ukládá do `models/`.** `Neuron.save`/`Neuron.load` uloží a
> zpět načtou váhy, bias a jméno aktivace do/z `.npz` — přesně tři čísla
> a trocha textových metadat, ale i tady platí stejný návyk jako
> u induktivních modelů z Cvičení 05–07: **model = parametry, které přežijí
> běh programu.** Rozdíl je jen v tom, odkud parametry pocházejí (ruční návrh,
> ne `fit`). Formát je `.npz`, nikdy `pickle` (kap. 9b konfigurátoru).

> **Žádný `src/distance.py`, žádná `DummyDistance`.** Neuron nepočítá párové
> vzdálenosti mezi body — počítá jediný vážený součet. Zavádět sem nepoužitou
> třídu `Distance` by bylo umělé (stejné pravidlo jako v Cvičení 05/07).

> **Jen jeden druh `NotImplementedError`.** Každá zpráva začíná `Úkol:` a je
> potřeba ji doplnit. Selhání jednoho neuronu na hradlech XOR/XNOR **není**
> tenhle druhý (trvalý) typ chyby — je to matematický fakt, který pipeline
> sama demonstruje a vysvětluje; nejde o nedodělanou metodu. Hláška
> `[NENI VYPLNENO]` u hradel je zase třetí, od kódu zcela nezávislý případ:
> znamená jen to, že `config.yaml` ještě nemá vyplněnou sekci `gates`.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění

```bash
python cviceni_08.py
```

Pipeline je rozdělena do fází a **každá fáze má vlastní ošetření chyb**. Dokud
nejsou úkoly hotové, fáze, která na nedokončenou metodu narazí, se ukončí
hláškou `[NENI HOTOVO] Úkol: …` a pipeline **pokračuje další fází**. Fáze,
které čtou vahy hradel, se navíc bez vyplněné sekce `gates` v `config.yaml`
ukončí hláškou `[NENI VYPLNENO]` — to je jiný, na kódu nezávislý stav (viz
[Pokyny k vypracování](#pokyny-k-vypracování)). Z jednoho spuštění tak
vidíte, co všechno ještě chybí; nikdy nedostanete holý traceback.

> **Jediná výjimka — načtení konfigurace.** `load_config()` volá
> `validate_config()`; kdyby v `config.yaml` byla nesmyslná hodnota, pipeline
> se korektně ukončí hláškou `[CHYBA KONFIGURACE]` hned na začátku. Obě funkce
> jsou předvyplněné, takže při nezměněné konfiguraci tento stav nenastane.

Jednotlivé metody lze mezitím ověřovat přes `pytest`, viz
[Lokální testování](#lokální-testování).

---

## Teoretický základ

### 1. Neuron jako nadrovina

Neuron spočítá vážený součet vstupů a volitelně přičte bias:

$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i} w_i x_i + b$$

a pak na `z` aplikuje aktivační funkci. Množina bodů, pro které platí
`z = 0`, je **nadrovina** (v 2D přímka, v 3D rovina) — **znaménko** `z`
říká, na které straně nadroviny bod leží:

- `z > 0` → bod leží „nad" nadrovinou (typicky třída 1),
- `z < 0` → bod leží „pod" nadrovinou (typicky třída 0),
- `z = 0` → bod leží přesně na hranici.

**Váhy `w` nadrovinu natáčejí** (mění směr normály), **bias `b` ji posouvá**
rovnoběžně se sebou samou, aniž by měnil její sklon. Návrh neuronu pro daný
klasifikační problém je proto **čistě geometrická úloha**: najít přímku
(nadrovinu), která oddělí třídy, a přečíst z ní `w` a `b`.

### 2. Rovnice hranice ve 2D a normálový vektor

Pro dva vstupní příznaky se hranice `z = 0` dá zapsat explicitně jako
přímka. Z

$$w_1 x_1 + w_2 x_2 + b = 0$$

vyjádříme (pro $w_2 \ne 0$) $x_2$:

$$x_2 = -\frac{w_1}{w_2}\, x_1 - \frac{b}{w_2}.$$

To je **směrnicový tvar přímky** $x_2 = a x_1 + c$ se **směrnicí**
$a = -w_1/w_2$ a **úsekem na ose** $c = -b/w_2$. Obě veličiny hranice jsou
tedy přímým algebraickým důsledkem vah a biasu — žádná nová informace, jen
jiný zápis téže rovnice.

> **Váhy jako normálový vektor.** Vektor $\mathbf{w} = (w_1, w_2)$ je **kolmý**
> na hranici $z = 0$ a směřuje k rostoucímu $z$ (tedy k třídě 1). Plyne to
> přímo z toho, že $z(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$ je lineární
> funkce a její gradient $\nabla z = \mathbf{w}$ je **konstantní** — směr
> nejrychlejšího růstu $z$ je všude stejný a kolmý na vrstevnice (množiny
> $z = \text{konst.}$, mezi nimi i $z=0$). Proto věta „váhy nadrovinu
> natáčejí" z předchozího oddílu není jen kvalitativní pozorování: natočení
> normály **je** to, co znamená změnit $\mathbf{w}$.

### 3. Bias jako záporně vzatý práh

Klasický zápis prahové podmínky je `w·x >= threshold` — „vážený součet
dosáhne alespoň prahu". Převedením prahu na levou stranu dostaneme

$$\mathbf{w}\cdot\mathbf{x} - \text{threshold} \ge 0 \quad\Longleftrightarrow\quad \mathbf{w}\cdot\mathbf{x} + b \ge 0, \qquad b = -\text{threshold}.$$

Bias je tedy **záporně vzatý práh**. Prakticky: chcete-li, aby neuron
„vystřelil" při prahu 1,5, nastavíte `bias = -1.5`; při prahu 0,5
`bias = -0.5`.

> **Odlišnost od staré „−1 augmentace".** Jedna z možných alternativních
> konvencí modeluje bias jako fiktivní vstupní příznak `x₀ = -1` s vlastní
> vahou `w₀`, čímž se `z = w·x + b` zapíše jako jediný skalární součin nad
> rozšířeným vektorem. Tady se bias nemodeluje jako sloupec dat, ale jako
> **samostatný, volitelný parametr** `bias: float | None`, stejně jako
> v PyTorch (`bias=False` = žádný bias). Je to čistší rozhraní: `weights`
> má vždy přesně tolik prvků, kolik má vstup příznaků, žádný umělý sloupec
> navíc.

### 4. `Linear` a `Activation` jako nezávislé vrstvy

`Neuron` sám nepočítá nic — skládá dvě samostatné, injektované součásti:

$$\text{Neuron}(\mathbf{x}) = \text{Activation}\big(\text{Linear}(\mathbf{x})\big), \qquad \text{Linear}(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b.$$

`Linear` počítá afinní transformaci (odd. 1–3 výše), `Activation` na jejím
výstupu aplikuje jednu z pěti funkcí z tabulky níže. Obě části jsou
samostatně testovatelné a **injektované** zvenku (dependency injection,
stejný vzor jako `Distance`/`Initializer`/`Validator`/`Kernel` v předchozích
cvičeních) — `Neuron` si žádnou z nich sám nevytváří. Je to obdoba skládání
`nn.Linear` a `nn.ReLU()` do sítě v PyTorch: vrstvy jsou nezávislé stavební
bloky, ne jeden monolitický objekt.

Díky tomu, že `Linear.forward` počítá `X @ weights`, funguje beze změny kódu
i pro `weights` tvaru `(n_příznaků, n_jednotek)` — tedy pro víc výstupních
jednotek najednou. Toto cvičení `Linear` používá vždy jen s jedním sloupcem
vah (jeden neuron), ale třída je tak architektonicky připravená na
zobecnění ve vícevrstvých sítích.

### 5. Aktivace: tvrdé vs. měkké rozhodnutí

| Aktivace | Vzorec | Obor hodnot | Charakter |
|:---|:---|:---|:---|
| `Step` | $1$ pokud $z \ge 0$, jinak $0$ | $\{0, 1\}$ | tvrdé rozhodnutí — ostrý skok |
| `Sigmoid` | $\dfrac{1}{1+e^{-z/T}}$ | $(0, 1)$ | měkké rozhodnutí — hladká pravděpodobnost |
| `Tanh` | $\tanh(z)$ | $(-1, 1)$ | měkké, symetrické kolem nuly |
| `ReLU` | $\max(0, z)$ | $[0, \infty)$ | neomezené shora, nulové pro záporné `z` |
| `ReLU6` | $\min(\max(0, z), 6)$ | $[0, 6]$ | `ReLU` s horním stropem |

`Sigmoid` má navíc parametr **`temperature`** ($T$), který řídí, jak ostrý
je přechod:

$$\sigma_T(z) = \frac{1}{1 + e^{-z/T}}.$$

> **Teplota jako most mezi tvrdým a měkkým.** Pro velké `T` je sigmoida
> téměř plochá (výstup blízko 0,5 skoro všude); pro `T → 0` se přechod
> zostřuje a `Sigmoid` se **limitně blíží** funkci `Step`. Pipeline to
> demonstruje sweepem přes několik hodnot `temperature` — sledujte, jak se
> hladká hranice mění v ostrou.

`Tanh`, `ReLU` a `ReLU6` nejsou v tomto cvičení použity ke klasifikaci na
hradlech (na to slouží `Step` a `Sigmoid`), ale jsou součástí stejné rodiny
a cvičení je testuje samostatně — s výhledem na to, že právě tyto funkce
tvoří vnitřní vrstvy neuronových sítí v navazujících cvičeních.

### 6. Hradla AND, OR, IMPLY a mez jednoho neuronu: XOR, XNOR

Logická hradla AND, OR a IMPLY nad vstupy $\{0,1\}^2$ jsou **lineárně
separovatelná** — existuje přímka, která odděluje výstup 1 od výstupu 0:

| Hradlo | Výstupy $(0,0),(0,1),(1,0),(1,1)$ | Práh / poznámka |
|:---|:---|:---|
| AND | $0, 0, 0, 1$ | práh $1{,}5$ |
| OR  | $0, 1, 1, 1$ | práh $0{,}5$ |
| IMPLY | $1, 1, 0, 1$ | izoluje jediný roh $(1,0)$ |

Konkrétní $(w, b)$ pro každé hradlo si student odvodí sám (viz
`priklady_08.md`, příklad 2) a vyplní do `config.yaml`.

**XOR** ($0, 1, 1, 0$) a **XNOR** ($1, 0, 0, 1$, negace XOR) **lineárně
separovatelné nejsou** — a to je záměrný, klíčový výsledek cvičení, ne
chyba k opravení.

> **Proč žádná přímka nefunguje (důkaz sporem, pro XOR).** Předpokládejme,
> že existuje $(w_1, w_2, b)$ takové, že `Step(w·x+b)` dá XOR. Z bodů
> $(0,0)\to 0$ a $(1,1)\to 0$ plyne $b < 0$ a $w_1+w_2+b < 0$. Z bodů
> $(0,1)\to 1$ a $(1,0)\to 1$ plyne $w_2+b \ge 0$ a $w_1+b \ge 0$, tedy
> jejich součet $w_1+w_2+2b \ge 0$. Sečtením první dvojice nerovností
> $w_1+w_2+b<0$ a $b<0$ dostaneme $w_1+w_2+2b < w_1+w_2+b < 0$ — což je
> ve sporu s $w_1+w_2+2b \ge 0$. Spor, tedy takové $(w,b)$ neexistuje.
> Pro XNOR (negaci XOR) platí tentýž argument se všemi nerovnostmi
> obrácenými.
>
> **Historický význam (Minsky–Papert, 1969).** Právě tento důkaz — že jeden
> perceptron nedokáže vyjádřit XOR — na dlouhou dobu utlumil výzkum
> neuronových sítí. Řešením není „lepší" jeden neuron, ale **více vrstev**:
> dva neurony ve skryté vrstvě dokážou XOR rozložit na kombinaci dvou
> lineárně separovatelných problémů (např. OR a NAND), které pak třetí
> neuron zkombinuje. To je přesně to, co otevírá Cvičení 09 — vícevrstvé
> sítě, ve kterých se parametry **učí** místo aby se navrhovaly ručně.

### 7. Logistická regrese jako speciální případ

Jednovstupý neuron (`weights` má jeden prvek) se sigmoidovou aktivací
**je** logistická regrese, kterou znáte z biostatistiky:

$$P(y=1 \mid x) = \sigma\!\big(w x + b\big) = \frac{1}{1+e^{-(wx+b)}}.$$

Klíčový rozdíl oproti `Step` na hradlech: výstup **není** tvrdá třída 0/1,
je to **pravděpodobnost příslušnosti ke třídě 1**. Rozhodnutí (klasifikace)
vzniká až dodatečným prahováním této pravděpodobnosti, typicky na 0,5.
Bod, kde sigmoida protíná 0,5, se dá spočítat přímo z parametrů:

$$\sigma(wx_0+b) = 0{,}5 \;\Longleftrightarrow\; wx_0+b = 0 \;\Longleftrightarrow\; x_0 = -\frac{b}{w}.$$

V tomto cvičení se `x₀` (a tedy i `(w,b)`) volí **ručně** — student se podívá
na rozložení jednoho příznaku datasetu Breast Cancer Wisconsin a odhadne
práh, který třídy rozumně odděluje (viz `priklady_08.md`). Cvičení 09 nahradí
tento ruční odhad iterativním dolaďováním — stejný model, jiný způsob, jak
se dostat k parametrům.

> **Proč právě sigmoida, a ne `Step`, pro pravděpodobnost.** `Step` dává jen
> dvě hodnoty, nedá se z něj číst „jak moc si je model jistý". `Sigmoid` je
> spojitá a monotónní, takže její hodnota nese informaci o vzdálenosti od
> hranice — čím dál od `x₀`, tím blíž k 0 nebo k 1. To je přesně vlastnost,
> kterou logistická regrese v biostatistice využívá (např. jako odhad
> rizika), a proč se pro klasifikaci s potřebou pravděpodobnosti nepoužívá
> `Step`.

---

## Konfigurace projektu

### Soubor `config.yaml`

```yaml
gates:                     # vahy a bias hradel -- DOPLNTE sami podle vlastniho odvozeni
  and:   {weights: null, bias: null}
  or:    {weights: null, bias: null}
  imply: {weights: null, bias: null}
  xor:   {weights: null, bias: null}
  xnor:  {weights: null, bias: null}

data:
  feature_index: 0          # ktery priznak breast-cancer datasetu pro 1D logistickou regresi (0..29)
  random_state: 42          # seed pro reprodukovatelne deleni/vyber dat

sigmoid:
  temperature: 1.0          # teplota sigmoidy; male cislo -> blizi se tvrdemu stupni (Step)

exploration:
  weight_sweep: {start: 0.5, stop: 3.5, step: 0.25}       # demonstrace rotace hranice pri zmene vahy
  temperature_sweep: [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]    # prechod tvrda -> hladka hranice
```

### Typovaná konfigurace (dataclassy)

```
ExperimentConfig
├── gates:       dict[str, GateParams(weights, bias)]   # klice: and, or, imply, xor, xnor
├── data:        DataConfig(feature_index, random_state)
├── sigmoid:     SigmoidConfig(temperature)
└── exploration: ExplorationConfig(weight_sweep, temperature_sweep)
```

K hodnotám se přistupuje **přes atributy, nikdy přes klíče slovníku**:

```python
# Místo:   cfg["sigmoid"]["temperature"]   ← runtime chyba při překlepu
# Správně: cfg.sigmoid.temperature          ← editor odhalí překlep okamžitě
```

`cfg.gates` je výjimka z „vždy atribut" — je to `dict[str, GateParams]`
klíčovaný jménem hradla (`cfg.gates["and"]`), protože `and`/`or` jsou
v Pythonu vyhrazená slova a nemohou být jména atributů dataclassy. Uvnitř
každé položky se ale opět přistupuje přes atributy (`cfg.gates["and"].weights`).

`validate_config()` ověří rozsahy hodnot (`0 <= feature_index <= 29`,
`temperature > 0`, `weight_sweep.start < weight_sweep.stop`,
`weight_sweep.step > 0`, `temperature_sweep` neprázdný a kladný,
`gates.*.weights` má délku 2 pokud je vyplněné) a při porušení vyhodí
`ValueError` se srozumitelnou hláškou. Hodnoty `gates.*.weights`/`bias`
rovné `null` jsou platný, očekávaný „ještě nevyplněno" stav — validace je
nekontroluje jako chybu.

---

## Pokyny k vypracování

Vaše práce je ve **dvou souborech**: `src/activations.py` (Blok I) a
`src/linear.py` (Blok II), plus dvě metody v `src/neuron.py` (Blok III).
Bloky vypracujte **v tomto pořadí** — `Neuron.forward` je předvyplněné a
volá `Linear.forward` i injektovanou aktivaci, takže bez dokončených Bloků
I a II nejde `Neuron` smysluplně otestovat.

Kromě kódu ještě **doplňte sekci `gates` v `config.yaml`** — váhy a bias
pěti hradel, které odvodíte na papíře (`priklady_08.md`, příklad 2 a 4).
Bez toho pipeline u fází 1, 1b a 2 jen vypíše, že konfigurace ještě chybí,
a pokračuje dál.

### Blok I: Aktivační funkce — `src/activations.py`

`Activation` je abstraktní báze (Strategy vzor — plochá ABC hierarchie,
implementace sdílejí jen rozhraní, ne kód) s abstraktní metodou `forward`
**a** abstraktní property `output_range`. `__call__` je předvyplněné a jen
deleguje na `forward`. Doplňte **u každého z pěti potomků** `forward` a
`output_range`:

```
# Step.forward(x):
#   Vraťte 1.0 tam, kde x >= 0, jinak 0.0 — vektorizovaně (np.where), ne smyčkou.
# Step.output_range:
#   Vraťte (0.0, 1.0).

# Sigmoid.forward(x):
#   Vraťte 1 / (1 + exp(-x / self.temperature)).
# Sigmoid.output_range:
#   Vraťte (0.0, 1.0).

# Tanh.forward(x):
#   Vraťte np.tanh(x).
# Tanh.output_range:
#   Vraťte (-1.0, 1.0).

# ReLU.forward(x):
#   Vraťte np.maximum(0.0, x).
# ReLU.output_range:
#   Vraťte (0.0, nekonečno).

# ReLU6.forward(x):
#   Vraťte np.minimum(np.maximum(0.0, x), 6.0).
# ReLU6.output_range:
#   Vraťte (0.0, 6.0).
```

### Blok II: Afinní vrstva — `src/linear.py`

`__init__` a `__call__` jsou předvyplněné. Doplňte jedinou metodu:

```
# Linear.forward(X):
#   Ověřte (assert), že X je 2D pole a že X.shape[1] odpovídá prvnímu rozměru self.weights.
#   Spočítejte a vraťte z = X @ self.weights + (self.bias, pokud není None, jinak 0.0).
#   Žádnou aktivaci zde neaplikujte — o tu se stará Neuron.
```

Připomínka ke znaménkové konvenci biasu (viz teorie, odd. 3): `bias` je
**záporně vzatý práh** — pro práh `t` nastavte `bias = -t`.

### Blok III: Perzistence — `Neuron.save` / `Neuron.load` v `src/neuron.py`

`Neuron.__init__`, `forward`, `__call__` a passthrough vlastnosti
`weights`/`bias` jsou předvyplněné (jen skládají `Linear` a `Activation`,
viz teorie odd. 4). Doplňte dvě metody:

```
# Neuron.save(path):
#   Slozte slovnik: 'weights' = self.linear.weights, 'activation_name' = type(self.activation).__name__,
#   pripadne 'bias' (jen pokud self.linear.bias is not None) a 'temperature' (jen pokud
#   self.activation je Sigmoid). Ulozte pomoci np.savez(path, **slovnik).

# Neuron.load(path) [classmethod]:
#   Nactete np.load(path). Precte 'weights'; 'bias' je None, pokud klic chybi, jinak float(...).
#   Aktivaci sestavte pres make_activation(jmeno, temperature=...) ze src.activations.
#   Vratte cls(Linear(weights, bias), activation).
```

Nepřítomnost klíče `bias` v archivu je signál `bias=None` — žádný sentinel
jako `NaN`.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_08.py -v
```

| Třída testů | Co ověřuje |
|:---|:---|
| `TestActivations` | Každá aktivace na známých vstupech (`Step(-1)=0`, `Step(0)=1`; `Sigmoid(0)=0.5`; `Tanh(0)=0`; `ReLU(-2)=0`; `ReLU6(10)=6`, …) a správnost `output_range`. |
| `TestLinear` | `Linear.forward` počítá `X @ weights + bias`; cesta `bias=None`; tvar výstupu. |
| `TestNeuron` | Skládá `Linear`+`Activation` a reprodukuje pravdivostní tabulky AND/OR; `weights`/`bias` passthrough (projde i bez dokončeného `Linear.forward`). |
| `TestSigmoidNeuron` | Výstup jednovstupého sigmoidového neuronu leží v `(0,1)` a je monotónní ve vstupu. |
| `TestPersistence` | `Neuron.save`→`Neuron.load` round-trip: shodné predikce, platný `.npz`, `bias=None` a `Sigmoid.temperature` přežijí uložení. |

Dokud nejsou příslušné metody hotové, testy, které je volají, se hlásí jako
**`xfail`** (očekávané selhání na `NotImplementedError`) a celá sada skončí
s návratovým kódem 0 — jde o záměrné chování, nikoli o chybu. Jakmile metodu
doplníte, stejný test začne procházet (`xpass` → `pass`).

Průběžně ověřujte i celou pipeline:

```bash
python cviceni_08.py
```

Kroky s neimplementovanými metodami se přeskočí s hláškou
`[NENI HOTOVO] Úkol: …`; ostatní proběhnou normálně a uloží grafy do
`graphs/`.

---

## Doplňkové (papírové) příklady

Soubor `priklady_08.md` obsahuje číselné příklady k ručnímu výpočtu, navázané
přímo na metody, které programujete: odvození vah a biasu pro AND, OR
a IMPLY z geometrie (hodnoty pak patří přímo do `config.yaml`), znaménková
konvence biasu, argument proč žádné `(w,b)` neoddělí XOR ani XNOR, ruční
vyhodnocení neuronu pro daný bod a aktivaci, volba prahu `x₀` pro 1D
logistickou regresi z malého vzorku dat, a na závěr sada cvičných
labelingů na mřížce bodů k procvičení bez nápovědy.

Čísla jsou volena tak, aby se dala spočítat na papíře. **Řešení nejsou
součástí repozitáře** — výsledky si ověřte u vyučujícího nebo výpočtem
v `numpy`.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení
implementace proveďte:

```bash
git add src/activations.py src/linear.py src/neuron.py config.yaml
git commit -m "Implementace cvičení 8"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří
správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho
repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **`config.yaml` se odevzdává také — ale jen sekce `gates`.** Je to jediná
> část konfigurace, kterou máte upravovat; `data`/`sigmoid`/`exploration`
> nechte beze změny.

> **Soubory, které se neodevzdávají:** `src/__init__.py`, `dataio/__init__.py`,
> `dataio/gates.py`, `dataio/loader.py`, `dataio/config_manager.py`,
> `dataio/plotting.py`, `cviceni_08.py` a `test_cviceni_08.py` jsou
> předvyplněny nebo se nemají měnit. Systém hodnotí soubory
> `src/activations.py`, `src/linear.py`, `src/neuron.py` a sekci `gates`
> v `config.yaml`.
