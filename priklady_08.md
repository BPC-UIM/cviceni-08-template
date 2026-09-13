# Papírové příklady — Cvičení 8

Neuron jako lineární klasifikátor, znaménková konvence biasu, aktivační funkce
a logistická regrese jako speciální případ sigmoidového neuronu. Příklady
procvičují **přesně tu notaci a ty výpočty**, které implementujete ve
`src/activations.py`, `src/linear.py` a `src/neuron.py` — žádné učení, váhy
a bias navrhujete ručně z geometrie a vyplňujete přímo do `config.yaml`.
Čísla jsou volena tak, aby se dala spočítat na papíře.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem v `numpy`.

Značení je stejné jako ve stub docstringech:

- `weights` — vektor vah tvaru `(n_priznaku,)`, nastavený ručně.
- `bias` — samostatný, volitelný skalár (`float | None`). `bias=None` znamená
  bez biasu, stejně jako PyTorch `bias=False`.
- `X` — matice vstupů tvaru `(n_vzorku, n_priznaku)`; jeden vzorek je řádek.
- `z` — vážený součet před aktivací, počítaný samostatnou vrstvou `Linear`:
  $z = $ `X @ weights + bias`, tedy pro jeden vzorek
  $z = w_1 x_1 + w_2 x_2 + \dots + \mathtt{bias}$. `Neuron` na `z` teprve
  aplikuje injektovanou `Activation` — `Neuron` = `Linear` + `Activation`.
- `x0` — rozhodovací práh na ose jednoho příznaku u 1D úlohy; platí
  $x_0 = -\,\mathtt{bias}/\mathtt{weight}$.
- Aktivace: `Step`, `Sigmoid(temperature)`, `Tanh`, `ReLU`, `ReLU6`.
  Pozor na neostrou nerovnost u `Step`: $z \ge 0 \Rightarrow 1$, jinak $0$.

Pravdivostní tabulky hradel (pořadí řádků `X` je v celém cvičení stejné jako
ve `dataio/gates.py`):

| `X` | AND | OR | IMPLY | XOR | XNOR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $(0, 0)$ | 0 | 0 | 1 | 0 | 1 |
| $(0, 1)$ | 0 | 1 | 1 | 1 | 0 |
| $(1, 0)$ | 0 | 1 | 0 | 1 | 0 |
| $(1, 1)$ | 1 | 1 | 1 | 0 | 1 |

---

## Příklad 1 — Znaménková konvence biasu

Klasický zápis prahové podmínky je „vážený součet dosáhne alespoň prahu",
tedy $w \cdot x \ge t$. V kódu ale pracujeme s aditivním tvarem
$z = w \cdot x + \mathtt{bias}$ a rozhodujeme podle znaménka `z`.

**Úkoly:**

- **a)** Přepište nerovnost $w \cdot x \ge t$ do tvaru $(\dots) \ge 0$ a z toho
  odvoďte obecný vztah mezi prahem $t$ a hodnotou `bias`.
- **b)** Doplňte tabulku (pro každý řádek uveďte chybějící hodnotu):

  | práh $t$ | `bias` |
  |:---:|:---:|
  | $2{,}5$ | ? |
  | $-0{,}75$ | ? |
  | $0$ | ? |
  | ? | `-3.0` |
  | ? | `+1.25` |

- **c)** Neuron má `weights = [1.0, 1.0]` a `bias = -0.5`. Jakému prahu $t$ to
  odpovídá? Zformulujte slovně podmínku, kterou tento neuron testuje.
- **d)** U kterého jediného prahu je `bias=None` a `bias=0.0` **naprosto
  zaměnitelné**? Vysvětlete, proč se přesto v kódu rozlišují (co dělá
  `forward`, když je `bias=None`).
- **e)** Proč **nelze** roli biasu zastoupit tím, že jen zvětšíte některou
  z vah? Argumentujte geometricky pomocí věty „váhy nadrovinu natáčejí, bias ji
  posouvá".

---

## Příklad 2 — Návrh vah a biasu pro AND a OR z geometrie

Uvažujte dvouvstupý neuron s aktivací `Step` a pevně zvolenými vahami
`weights = [1.0, 1.0]`.

**Úkoly:**

- **a)** Spočítejte `z` **bez biasu** (tj. `X @ weights`) pro všechny čtyři
  řádky `X`. Kolik různých hodnot vyjde a kterým bodům odpovídají?
- **b)** Aby neuron realizoval **AND**, musí `Step` dát 1 pouze pro $(1,1)$.
  Zapište to jako dvě nerovnosti pro `bias` (jednu z bodu $(1,1)$, druhou
  z bodů $(0,1)$ a $(1,0)$) a vyřešte je. Výsledkem je **interval** přípustných
  hodnot `bias` — uveďte ho včetně informace, který konec je uzavřený a proč
  (souvisí s neostrou nerovností v `Step`).
- **c)** Totéž pro **OR**: zapište nerovnosti a určete interval přípustných
  `bias`.
- **d)** Z každého intervalu vyberte **střed** a vysvětlete, proč je to lepší
  volba než kterýkoli jiný bod intervalu (pojem *odstup od hranice*, margin).
  Vámi zvolené dvojice `(weights, bias)` pro AND a OR jsou přesně to, co
  vyplníte do `config.yaml` (sekce `gates.and`, `gates.or`).
- **e)** Pro vámi zvolený AND-neuron dopočítejte všechny čtyři hodnoty `z`
  a výsledek `Step(z)`; ověřte, že se shoduje s pravdivostní tabulkou. Totéž
  pro OR.
- **f)** Napište rovnici rozhodovací hranice ($z = 0$) pro oba neurony ve tvaru
  $x_2 = a\,x_1 + b$ a nakreslete obě přímky do jednoho grafu se čtyřmi body
  hradla. Které body leží v polorovině $z \ge 0$?
- **g)** Navrhněte stejným postupem neuron pro **NAND** (negace AND, tj.
  výstupy 1, 1, 1, 0). *Nápověda:* začněte od `weights = [-1.0, -1.0]`. Určete
  opět celý přípustný interval `bias` a jeho střed.
- **h)** Navrhněte stejným postupem neuron pro **IMPLY** (implikace
  $x_1 \Rightarrow x_2$, výstupy 1, 1, 0, 1 — nepravda jen pro $(1,0)$).
  *Nápověda:* zkuste `weights = [-1.0, 1.0]` a najděte přípustný interval
  `bias` i jeho střed. Výsledek je hodnota, kterou vyplníte do
  `config.yaml` (sekce `gates.imply`).

---

## Příklad 3 — Ruční vyhodnocení neuronu pro daný bod a aktivaci

Je dán neuron s `weights = [2.0, -1.0]` a `bias = -0.5`.

**Úkoly:**

- **a)** Spočítejte `z` pro každý z bodů

  | bod | $x_1$ | $x_2$ |
  |:---:|:---:|:---:|
  | A | $1{,}0$ | $1{,}0$ |
  | B | $0{,}0$ | $0{,}0$ |
  | C | $0{,}25$ | $0{,}0$ |
  | D | $4{,}0$ | $1{,}0$ |

- **b)** Doplňte pro každý bod výstup aktivací `Step`, `ReLU` a `ReLU6`.
  Tyto tři zvládnete přesně, bez kalkulačky.
- **c)** Bod C leží **přesně na rozhodovací hranici**. Jakou hodnotu dá `Step`
  a proč? Které místo stub docstringu to určuje? Co by se změnilo, kdyby byla
  podmínka $z > 0$ místo $z \ge 0$?
- **d)** U bodu D se výstupy `ReLU` a `ReLU6` **liší**. Uveďte oba a vysvětlete
  rozdíl odkazem na `output_range` obou aktivací.
- **e)** Určete znaménko `Tanh(z)` pro každý bod bez počítání a vysvětlete,
  proč stačí znát znaménko `z`. Pro který bod vyjde `Tanh` přesně 0?
- **f)** Pro body A a B spočítejte `Sigmoid` s `temperature = 1.0`
  (na 3 desetinná místa; použijte $e^{0{,}5} \approx 1{,}6487$). Ověřte, že
  součet obou výsledků je 1, a vysvětlete, čím je to způsobeno (symetrie
  logistické funkce kolem $z = 0$).
- **g)** Neuron dostane celou matici `X` se všemi čtyřmi body naráz. Jaký
  **tvar** má `z` a jaký tvar má návratová hodnota `forward`? Proč se nesmí
  psát smyčka přes řádky `X`?

---

## Příklad 4 — XOR a hranice jednoho neuronu

Hradlo XOR má výstupy 0, 1, 1, 0 (viz tabulka v úvodu). Uvažujte **jediný**
neuron s aktivací `Step`, tedy libovolné `weights = [w1, w2]` a `bias`.

**Úkoly:**

- **a)** Zapište `z` pro všechny čtyři body symbolicky pomocí $w_1$, $w_2$
  a `bias`.
- **b)** Z požadovaných výstupů XOR odvoďte čtyři nerovnosti (dvě s $\ge 0$,
  dvě s $< 0$).
- **c)** Sečtěte dvě nerovnosti odpovídající bodům s výstupem **1** a porovnejte
  je se součtem nerovností pro body s výstupem **0**. Dojděte ke sporu a
  zformulujte závěr jednou větou. *(Toto je Minského–Papertův argument.)*
- **d)** Ukaž, že součet `z` přes body $\{(0,0), (1,1)\}$ je **vždy** stejný
  jako součet `z` přes body $\{(0,1), (1,0)\}$, ať jsou `weights` a `bias`
  jakékoli. Proč tento invariant sám o sobě XOR vylučuje?
- **e)** Nakreslete čtyři body XOR do roviny. Zkuste je oddělit jednou přímkou
  a slovně popište, na čem to selže. Kolik ze čtyř bodů lze jednou přímkou
  klasifikovat správně nejvíc?
- **f)** Selhání XOR v pipeline **není chyba, kterou byste měli opravovat**.
  Vysvětlete jednou větou, co v pipeline pozorujete a proč je to záměrný
  výsledek.
- **g)** **XNOR** (negace XOR, výstupy 1, 0, 0, 1) je stejně
  neseparovatelný. Ukažte, že stejný důkaz sporem z bodu c) platí i pro
  XNOR — stačí přeformulovat, které body mají mít výstup 1 a které 0, zbytek
  argumentu je identický. Do `config.yaml` (sekce `gates.xor`, `gates.xnor`)
  přesto vyplňte libovolnou dvojici `(weights, bias)` — pipeline tím ukáže
  přesnost pod 1.00 bez ohledu na volbu, což je přesně bod cvičení.
- **h)** *(výhled na cv9)* XOR lze zapsat jako
  $\mathrm{XOR}(x) = \mathrm{AND}\big(\mathrm{OR}(x),\ \mathrm{NAND}(x)\big)$.
  Použijte své neurony z příkladu 2 b)–d) a 2 g): spočítejte pro všechny čtyři
  body výstupy OR-neuronu a NAND-neuronu, sestavte z nich nový dvouprvkový
  vstupní vektor a proženěte ho AND-neuronem. Vyjde XOR? Kolik neuronů a kolik
  **vrstev** jste použili, a proč to neodporuje závěru bodu c)?

---

## Příklad 5 — Volba prahu `x0` pro 1D logistickou regresi

Máme jeden číselný příznak (např. průměrný radius buněčného jádra) a malý
vzorek sedmi pacientů. Popisek `1` znamená **maligní** (stejné kódování jako
v `dataio/loader.py`).

| `x` | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `y` | 0 | 0 | 1 | 0 | 0 | 1 | 1 |

Klasifikujeme pravidlem: predikuj 1, právě když `x` $\ge$ `x0`.

**Úkoly:**

- **a)** Za kandidáty na `x0` stačí brát **středy mezer** mezi sousedními
  hodnotami `x` (plus jeden kandidát pod nejmenší a jeden nad největší
  hodnotou). Vysvětlete, proč se tím nic neztratí — proč nemá smysl zkoušet
  např. $x_0 = 14{,}2$ zvlášť.
- **b)** Pro každého kandidáta z bodu a) vypište vektor predikcí a **počet
  chybně klasifikovaných** vzorků. Doporučuji tabulku se sloupci
  `x0` | predikce | počet chyb.
- **c)** Který `x0` minimalizuje počet chyb? Je optimum **jednoznačné**?
  Kolik chyb v optimu zbývá a který pacient je klasifikován špatně?
- **d)** Tento vzorek **není** 1D lineárně separovatelný. Poznáte to přímo
  z tabulky — jak? Jakou nejmenší změnu jedné hodnoty `y` byste museli udělat,
  aby separovatelný byl?
- **e)** Ze vztahu $x_0 = -\,\mathtt{bias}/\mathtt{weight}$ dopočítejte `bias`
  pro svůj optimální `x0`, jestliže zvolíte `weight = 1.0`. Pak totéž pro
  `weight = 4.0`. Porovnejte oba výsledky: liší se `bias`, liší se `x0`?
- **f)** Pro obě volby z bodu e) spočítejte `z` v bodě `x = 16` a určete, která
  dá **vyšší** hodnotu `Sigmoid`. Co znamená `weight` geometricky pro tvar
  sigmoidy a jak se to pozná na grafu z `plot_sigmoid_1d`?
- **g)** Jakou hodnotu dá `Sigmoid` přesně v bodě `x0`, a to pro **libovolnou**
  kladnou `weight`? Zdůvodněte a vysvětlete, proč je právě 0,5 přirozená
  hranice pro převod pravděpodobnosti na tvrdou třídu.
- **h)** Ukažte, že prahování $\mathtt{Sigmoid}(z) \ge 0{,}5$ dává **tytéž**
  predikce jako $\mathtt{Step}(z)$. Co tedy sigmoidový neuron přidává, když
  je výsledná tvrdá klasifikace stejná?
- **i)** Změníte-li u `Sigmoid` parametr `temperature` (např. z `1.0` na `0.2`),
  posune se rozhodovací práh `x0`? Rozhodněte a zdůvodněte ze vztahu
  $\mathtt{Sigmoid}(z) = 1/(1 + e^{-z/T})$.

---

## Příklad 6 — Aktivační funkce, obory hodnot a teplota

**Úkoly:**

- **a)** Doplňte tabulku `output_range` — přesně ty n-tice, které vracejí
  property v `src/activations.py`:

  | aktivace | `output_range` |
  |:---|:---|
  | `Step` | ? |
  | `Sigmoid` | ? |
  | `Tanh` | ? |
  | `ReLU` | ? |
  | `ReLU6` | ? |

- **b)** U kterých aktivací jsou obě **meze skutečně dosažené** (existuje `z`,
  pro které výstup rovná mezi) a u kterých jsou pouze **asymptotické**?
  Podložte to jedním konkrétním `z` u každé aktivace.
- **c)** Je dán vektor `z = [-2.0, -0.5, 0.0, 0.5, 3.0, 7.0]`. Vypište přesně
  (bez kalkulačky) výstup `Step`, `ReLU` a `ReLU6`. U kterého prvku se `ReLU`
  a `ReLU6` liší a proč?
- **d)** Pro `z = 0.5` spočítejte `Sigmoid` s `temperature` po řadě `1.0`,
  `0.1` a `0.01` (pomůcky: $e^{-0{,}5} \approx 0{,}6065$,
  $e^{-5} \approx 6{,}738\cdot 10^{-3}$, $e^{-50} \approx 1{,}93\cdot10^{-22}$).
  Zopakujte pro `z = -0.5`. K čemu se posloupnost výsledků blíží?
- **e)** Zformulujte závěr bodu d) jednou větou o vztahu mezi `Sigmoid`
  s malou `temperature` a aktivací `Step`. Které místo v pipeline (který blok)
  tohle ukazuje na obrázku?
- **f)** Proč by **`ReLU` byla nevhodná** jako výstupní aktivace tohoto
  binárního klasifikátoru? Uveďte dva důvody a oba opřete o `output_range`.
- **g)** `Tanh` a `Sigmoid` dávají pro totéž `z` **stejné rozhodnutí**, pokud
  `Tanh` prahujeme na 0 a `Sigmoid` na 0,5. Ukažte to na prvcích vektoru
  z bodu c) a doplňte: jaký je obecný vztah mezi oběma funkcemi?
- **h)** Všechny stuby mají implementovat výpočet **vektorizovaně**
  (`np.where`, `np.exp`, `np.tanh`, `np.maximum`, `np.clip`). Vysvětlete, co
  přesně by bylo špatně na řešení, které iteruje `for` cyklem přes prvky `x`
  a vrací `list` — uveďte důvod věcný (co se rozbije dál v kódu) i praktický.
- **i)** `Neuron.save` ukládá `weights`, volitelně `bias`, a
  `type(self.activation).__name__` (jméno třídy jako text) — pro `Sigmoid`
  navíc `temperature`. Vysvětlete, proč **nestačí** uložit jen `weights`
  a `bias`: co přesně by `Neuron.load` nemohl zjistit, kdyby v archivu
  jméno aktivace chybělo? Uveďte konkrétní příklad dvou neuronů se stejnými
  `weights`/`bias`, ale různým chováním.

---

## Příklady k procvičení

Devět bodů na mřížce $\{0;\,0{,}5;\,1\} \times \{0;\,0{,}5;\,1\}$ a deset
sloupců štítků $y_1$–$y_{10}$. **Každý sloupec je lineárně separovatelný
jedním neuronem** — pro každé $y_i$ existuje dvojice `(weights, bias)`
taková, že `Step(X @ weights + bias)` dá přesně tento sloupec. Najděte `weights` a `bias` výpočtem (např. sestavením a řešením soustavy nerovností jako v Příkladu 2, nebo odhadem směru dělicí přímky z rozložení bodů a
dopočtením prahu).

| $x_1$ | $x_2$ | $y_1$ | $y_2$ | $y_3$ | $y_4$ | $y_5$ | $y_6$ | $y_7$ | $y_8$ | $y_9$ | $y_{10}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 1 |
| 0.0 | 0.5 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| 0.0 | 1.0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| 0.5 | 0.0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| 0.5 | 0.5 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| 0.5 | 1.0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| 1.0 | 0.0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| 1.0 | 0.5 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 |
| 1.0 | 1.0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |

Pro každý sloupec najděte alespoň jednu dvojici `(weights, bias)`. U alespoň
jednoho z nich navíc rozhodněte, zda je nalezené řešení **jediné možné**
(až na násobek), nebo existuje celý interval/oblast přípustných hodnot —
stejná otázka jako v Příkladu 2 b)–c), tady ale bez vedení.
