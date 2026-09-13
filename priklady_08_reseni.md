# Řešení papírových příkladů — Cvičení 8

> **Tento soubor NENÍ součástí gitu** (viz `.gitignore`, blok „Vedlejsi
> (negitovane) reseni papirovych prikladu"). Slouží vyučujícímu; studentům se
> nedodává. Trackovaný `priklady_08.md` žádná řešení neobsahuje.
>
> **Všechna čísla níže jsou numericky ověřena v `numpy`** (Reviewer A, WP6;
> doplňky IMPLY/XNOR/perzistence z revize ověřil orchestrátor) — hradla,
> intervaly biasu přes hrubé prohledání, XOR/XNOR přes prohledání mřížky
> $[-3,3]$ s krokem 0,25 i 200 000 náhodných $(w, b)$, 1D práh přes všechny
> kandidáty. Nic zde není odhad.
>
> **Sekce „Příklady k procvičení" na konci `priklady_08.md` NEMÁ řešení
> ani zde** — schválně, doplní si je uživatel sám.

---

## Příklad 1 — Znaménková konvence biasu

**a)** $w \cdot x \ge t \iff w \cdot x - t \ge 0$. Aditivní tvar je
$z = w \cdot x + \mathtt{bias}$, tedy **`bias` $= -t$** (bias je záporně vzatý
práh).

**b)**

| práh $t$ | `bias` |
|:---:|:---:|
| $2{,}5$ | `-2.5` |
| $-0{,}75$ | `+0.75` |
| $0$ | `0.0` (ekvivalentně `None`) |
| $3{,}0$ | `-3.0` |
| $-1{,}25$ | `+1.25` |

**c)** $t = -\mathtt{bias} = 0{,}5$. Neuron testuje $x_1 + x_2 \ge 0{,}5$, tedy
„aspoň jeden vstup je 1" — to je **OR**.

**d)** Jen pro $t = 0$. Rozlišují se proto, že `bias=None` je **strukturální**
vlastnost modelu („neuron bez biasu", PyTorch `bias=False`), ne číselná hodnota:
hranice je pak *nucena* procházet počátkem a nemůže se posunout. `forward`
v té větvi nepřičítá nic (`+ 0.0`), takže numericky to vyjde stejně jako
`bias=0.0`, ale koncepčně jde o jiný model (u `bias=0.0` je bias zapnutý a jen
náhodou nulový).

**e)** Nelze. Množina $\{x : w \cdot x = 0\}$ prochází počátkem pro **každé**
`weights`; násobení vah konstantou mění jen délku/směr normály (hranice se
natáčí a mění se strmost `z`), ale hranice pořád prochází počátkem. Posun
mimo počátek zajišťuje výhradně `bias`. Proto „váhy natáčejí, bias posouvá".

---

## Příklad 2 — Návrh vah a biasu pro AND a OR

**a)** `X @ weights` $= [0, 1, 1, 2]$. Tři různé hodnoty: $0$ pro $(0,0)$,
$1$ pro $(0,1)$ i $(1,0)$ (symetrie vah), $2$ pro $(1,1)$.

**b) AND.** Požadavky: $2 + \mathtt{bias} \ge 0$ a $1 + \mathtt{bias} < 0$, tedy
$$\mathtt{bias} \in [-2,\ -1).$$
Levý konec je **uzavřený**, protože `Step` používá neostrou nerovnost: pro
`bias = -2.0` je v bodě $(1,1)$ přesně $z = 0$ a `Step(0) = 1`, což je pořád
AND. Pravý konec je **otevřený**: pro `bias = -1.0` by v bodech $(0,1)$, $(1,0)$
bylo $z = 0$, tedy `Step = 1` — to už AND není.

**c) OR.** Požadavky: $1 + \mathtt{bias} \ge 0$ a $0 + \mathtt{bias} < 0$, tedy
$$\mathtt{bias} \in [-1,\ 0).$$

**d)** Středy: **AND `bias = -1.5`**, **OR `bias = -0.5`** — přesně hodnoty
z README, pipeline i testů. Střed maximalizuje **odstup od hranice**: obě
nejbližší hodnoty `z` leží ve vzdálenosti $0{,}5$ od nuly ($\pm 0{,}5$), zatímco
u krajů intervalu je jedna z nich nulová, takže i nepatrný šum ve vstupu (nebo
zaokrouhlení) může překlopit rozhodnutí.

**e)** AND, `bias = -1.5`: $z = [-1{,}5,\ -0{,}5,\ -0{,}5,\ +0{,}5]$,
`Step(z)` $= [0, 0, 0, 1]$ ✓.
OR, `bias = -0.5`: $z = [-0{,}5,\ +0{,}5,\ +0{,}5,\ +1{,}5]$,
`Step(z)` $= [0, 1, 1, 1]$ ✓. (Shoda s referenčními čísly BRIEF §5.)

**f)** $z = 0 \Rightarrow x_1 + x_2 + \mathtt{bias} = 0 \Rightarrow
x_2 = -x_1 - \mathtt{bias}$. Tedy AND: $x_2 = -x_1 + 1{,}5$; OR:
$x_2 = -x_1 + 0{,}5$. Obě přímky mají směrnici $-1$ (stejné váhy), liší se jen
posunem — bias je posouvá rovnoběžně. V polorovině $z \ge 0$ leží u AND jen
$(1,1)$, u OR body $(0,1)$, $(1,0)$, $(1,1)$.

**g) NAND.** S `weights = [-1.0, -1.0]` je
$z = -(x_1 + x_2) + \mathtt{bias}$, tedy hodnoty
$[\mathtt{bias},\ \mathtt{bias}-1,\ \mathtt{bias}-1,\ \mathtt{bias}-2]$.
Požadavky (výstupy 1, 1, 1, 0): $\mathtt{bias} - 1 \ge 0$ a
$\mathtt{bias} - 2 < 0$, tedy
$$\mathtt{bias} \in [1,\ 2), \qquad \text{střed } \mathtt{bias} = 1{.}5 .$$
Kontrola: $z = [1{,}5,\ 0{,}5,\ 0{,}5,\ -0{,}5] \to [1, 1, 1, 0]$ ✓.
(Ověřeno i to, že `bias = 0.5` **nestačí** — dá $[1, 0, 0, 0]$.)

**h) IMPLY.** S `weights = [-1.0, 1.0]` je
$z(0,0) = \mathtt{bias}$, $z(0,1) = 1+\mathtt{bias}$,
$z(1,0) = -1+\mathtt{bias}$, $z(1,1) = \mathtt{bias}$ (stejné jako $z(0,0)$ —
váhy se u tohoto bodu vzájemně vyruší). Požadavky (výstupy 1, 1, 0, 1):
$\mathtt{bias} \ge 0$ a $-1+\mathtt{bias} < 0$, tedy
$$\mathtt{bias} \in [0,\ 1), \qquad \text{střed } \mathtt{bias} = 0{,}5 .$$
Kontrola: $z = [0{,}5,\ 1{,}5,\ -0{,}5,\ 0{,}5] \to [1, 1, 0, 1]$ ✓ — shoda
s tabulkou IMPLY v úvodu. Do `config.yaml`: `gates.imply.weights = [-1.0, 1.0]`,
`gates.imply.bias = 0.5`.

---

## Příklad 3 — Ruční vyhodnocení (`weights = [2.0, -1.0]`, `bias = -0.5`)

**a), b)**

| bod | $x$ | `z` | `Step` | `ReLU` | `ReLU6` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | $(1{,}0;\ 1{,}0)$ | $+0{,}5$ | 1 | $0{,}5$ | $0{,}5$ |
| B | $(0{,}0;\ 0{,}0)$ | $-0{,}5$ | 0 | $0{,}0$ | $0{,}0$ |
| C | $(0{,}25;\ 0{,}0)$ | $\ \ 0{,}0$ | **1** | $0{,}0$ | $0{,}0$ |
| D | $(4{,}0;\ 1{,}0)$ | $+6{,}5$ | 1 | $6{,}5$ | $6{,}0$ |

**c)** `Step(0.0) = 1`, protože docstring `Step.forward` definuje
`f(x) = 1, pokud x >= 0` — **neostrá** nerovnost. Kdyby bylo $z > 0$, vyšla by
u C nula a zároveň by se změnily uzavřené konce intervalů z příkladu 2 b)/2 g)
(`bias = -2.0` by přestalo dávat AND).

**d)** `ReLU(6.5) = 6.5`, `ReLU6(6.5) = 6.0`. `ReLU6` shora ořezává na 6, protože
její `output_range` je konečný `(0.0, 6.0)`; `ReLU` má `(0.0, np.inf)`, tedy
shora neomezený, a hodnotu propustí beze změny.

**e)** `Tanh` je **rostoucí a lichá**, `tanh(0) = 0`, takže
$\operatorname{sign}(\mathtt{Tanh}(z)) = \operatorname{sign}(z)$: A kladné,
B záporné, **C přesně 0**, D kladné. Znaménko `z` tedy rozhoduje samo.

**f)** $\mathtt{Sigmoid}(+0{,}5) = 1/(1 + e^{-0{,}5}) = 1/(1+0{,}6065)
= \mathbf{0{,}6225}$ (přesně $0{,}62245933$);
$\mathtt{Sigmoid}(-0{,}5) = \mathbf{0{,}3775}$ (přesně $0{,}37754067$).
Součet je 1, protože $\mathtt{Sigmoid}(-z) = 1 - \mathtt{Sigmoid}(z)$
(symetrie kolem $z = 0$, kde je hodnota $0{,}5$).

**g)** `X` má tvar $(4, 2)$, `weights` tvar $(2,)$, takže `z` má tvar
$\mathbf{(4,)}$ a `forward` vrací rovněž $(4,)$ — jedna hodnota na vzorek.
Smyčka se nesmí psát proto, že (i) zmrazený kontrakt slibuje `np.ndarray`, na
který navazující kód volá numpy operace (`reshape` pro mřížku konturového
grafu, porovnání, `np.where`), (ii) `DecisionBoundaryPlotter` neuron vyhodnocuje
na hustě vzorkované mřížce (desetitisíce bodů), kde je python smyčka o řády
pomalejší.

---

## Příklad 4 — XOR

**a)** $z(0,0) = \mathtt{bias}$, $z(0,1) = w_2 + \mathtt{bias}$,
$z(1,0) = w_1 + \mathtt{bias}$, $z(1,1) = w_1 + w_2 + \mathtt{bias}$.

**b)** XOR žádá $[0, 1, 1, 0]$, tedy
$$\mathtt{bias} < 0,\quad w_2 + \mathtt{bias} \ge 0,\quad
w_1 + \mathtt{bias} \ge 0,\quad w_1 + w_2 + \mathtt{bias} < 0 .$$

**c)** Součet dvou nerovností pro body s výstupem **1**:
$$w_1 + w_2 + 2\,\mathtt{bias} \ \ge\ 0 .$$
Součet dvou nerovností pro body s výstupem **0**:
$$\mathtt{bias} + (w_1 + w_2 + \mathtt{bias}) \;=\; w_1 + w_2 + 2\,\mathtt{bias}
\ <\ 0 .$$
Táž veličina $w_1 + w_2 + 2\,\mathtt{bias}$ by musela být zároveň
$\ge 0$ i $< 0$ — **spor**. Závěr: **žádná volba `weights` a `bias`
nerealizuje s aktivací `Step` hradlo XOR**; XOR není lineárně separovatelný
(Minsky–Papert).

**d)** $z(0,0) + z(1,1) = \mathtt{bias} + (w_1 + w_2 + \mathtt{bias})
= w_1 + w_2 + 2\,\mathtt{bias} = (w_2 + \mathtt{bias}) + (w_1 + \mathtt{bias})
= z(0,1) + z(1,0)$ — platí identicky, pro libovolné parametry. Ale XOR žádá, aby
oba členy levé strany byly **záporné** (součet $< 0$) a oba členy pravé strany
**nezáporné** (součet $\ge 0$); jedno číslo nemůže být obojí.

**e)** Nejvýš **3 ze 4** bodů lze klasifikovat správně. *(Ověřeno: prohledání
mřížky $w_1, w_2, \mathtt{bias} \in [-3, 3]$ s krokem 0,25 nenašlo žádné řešení
s nulovou chybou; 200 000 náhodných trojic dalo minimum 1 chybu.)* Geometricky:
body téže třídy leží na **protilehlých** vrcholech jednotkového čtverce, takže
každá přímka, která oddělí jeden z nich, nechá druhý na špatné straně.

**f)** Pipeline v bloku 1 reprodukuje AND i OR přesně, v bloku 2 u XOR
nejlépe 3/4 — a **tak to má být**. Je to demonstrace matematického faktu, ne
chyba k opravě; přesně tohle motivuje víc­vrstvé sítě v cv9+.

**g) XNOR.** XNOR žádá $[1, 0, 0, 1]$ — přesně opačné znaménko nerovnosti
u každého bodu oproti XOR. Analogicky k bodu c): sečtením nerovností pro
body s výstupem **1** ($z(0,0)\ge 0$, $z(1,1)\ge 0$) dostaneme
$w_1+w_2+2\,\mathtt{bias} \ge 0$; sečtením nerovností pro výstup **0**
($z(0,1)<0$, $z(1,0)<0$) dostaneme $w_1+w_2+2\,\mathtt{bias} < 0$ — týž
spor. Žádná volba `(weights, bias)` tedy nerealizuje ani XNOR. Do
`config.yaml` (`gates.xor`, `gates.xnor`) stačí vyplnit libovolnou dvojici,
např. `weights = [1.0, 1.0]`, `bias = -1.5` pro obě — pipeline ukáže
přesnost `0.25` u obou hradel (ověřeno numericky), což ilustruje bod f).

**h)** OR-neuron: $h_1 = [0, 1, 1, 1]$. NAND-neuron: $h_2 = [1, 1, 1, 0]$.
Nový vstup $H = [[0,1],\ [1,1],\ [1,1],\ [1,0]]$. AND-neuron
(`weights = [1.0, 1.0]`, `bias = -1.5`) nad $H$:
$z = [-0{,}5,\ +0{,}5,\ +0{,}5,\ -0{,}5] \to [0, 1, 1, 0] = $ **XOR** ✓
(ověřeno numericky).
Použity **3 neurony ve 2 vrstvách**. Bodu c) to neodporuje, protože c) vylučuje
jen **jeden** neuron / jednu nadrovinu nad původními vstupy. Skrytá vrstva mění
**reprezentaci**: v souřadnicích $(h_1, h_2)$ už jsou třídy lineárně
separovatelné (body třídy 0 jsou $(0,1)$ a $(1,0)$, třída 1 je dvakrát $(1,1)$).

---

## Příklad 5 — Volba prahu `x0` (1D logistická regrese)

Data: `x = [10, 11, 12, 13, 14, 15, 16]`, `y = [0, 0, 1, 0, 0, 1, 1]`.

**a)** Vektor predikcí `x >= x0` se mění jen tehdy, když `x0` **překročí
datový bod**; mezi dvěma sousedními hodnotami `x` je počet chyb konstantní.
Proto stačí jeden reprezentant z každé mezery — $x_0 = 14{,}2$ dá tytéž
predikce jako $x_0 = 14{,}5$.

**b)** Všech osm kandidátů (ověřeno numericky):

| `x0` | predikce | počet chyb |
|:---:|:---|:---:|
| $9{,}5$ | `[1 1 1 1 1 1 1]` | 4 |
| $10{,}5$ | `[0 1 1 1 1 1 1]` | 3 |
| $11{,}5$ | `[0 0 1 1 1 1 1]` | 2 |
| $12{,}5$ | `[0 0 0 1 1 1 1]` | 3 |
| $13{,}5$ | `[0 0 0 0 1 1 1]` | 2 |
| **$14{,}5$** | `[0 0 0 0 0 1 1]` | **1** |
| $15{,}5$ | `[0 0 0 0 0 0 1]` | 2 |
| $16{,}5$ | `[0 0 0 0 0 0 0]` | 3 |

**c)** Optimum je **`x0 = 14.5`**, a je **jednoznačné** (jediný kandidát
s jednou chybou). Zbývá **1 chyba**: pacient s `x = 12` má `y = 1`, ale je
predikován jako 0.

**d)** Separovatelný není, protože popisky se na ose **prokládají**: jednička
(`x = 12`) leží mezi nulami (`x = 10, 11` pod ní a `x = 13, 14` nad ní). Žádný
jediný bod na ose je neoddělí. Nejmenší úprava: přepsat `y` u `x = 12`
z 1 na 0 — pak je `y = [0,0,0,0,0,1,1]` a `x0 = 14.5` dává **0 chyb**.

**e)** `weight = 1.0` $\Rightarrow$ `bias` $= -1{,}0 \cdot 14{,}5 =$ **`-14.5`**.
`weight = 4.0` $\Rightarrow$ `bias` $= -4{,}0 \cdot 14{,}5 =$ **`-58.0`**.
`bias` se liší, **`x0` je v obou případech totéž**:
$-(-14{,}5)/1{,}0 = -(-58{,}0)/4{,}0 = 14{,}5$.

**f)** V `x = 16`: `weight = 1.0` dá $z = 1{,}5$ a
$\mathtt{Sigmoid} = \mathbf{0{,}8176}$; `weight = 4.0` dá $z = 6{,}0$ a
$\mathtt{Sigmoid} = \mathbf{0{,}9975}$. Vyšší hodnotu dá **větší `weight`**.
Geometricky `weight` určuje **strmost** sigmoidy (směrnice v bodě `x0` je
$\mathtt{weight}/4$ pro $T = 1$): velká `weight` → ostrý, téměř skokový přechod
(blízko `Step`), malá `weight` → plochá, nejistá křivka. V grafu z
`plot_sigmoid_1d` se to pozná jako různě strmá křivka procházející **týmž**
bodem `x0`.

**g)** V `x0` je $z = \mathtt{weight} \cdot x_0 + \mathtt{bias}
= \mathtt{weight}\cdot(-\mathtt{bias}/\mathtt{weight}) + \mathtt{bias} = 0$,
a $\mathtt{Sigmoid}(0) = 1/(1+e^0) = \mathbf{0{,}5}$ — pro libovolnou kladnou
`weight` i libovolnou `temperature`. Hodnota $0{,}5$ je přirozená hranice,
protože právě tam jsou obě třídy stejně pravděpodobné, tedy tam `z` mění
znaménko.

**h)** `Sigmoid` je striktně rostoucí a $\mathtt{Sigmoid}(0) = 0{,}5$, takže
$\mathtt{Sigmoid}(z) \ge 0{,}5 \iff z \ge 0 \iff \mathtt{Step}(z) = 1$ —
**tvrdé predikce jsou identické**. Sigmoida přidává (i) **míru jistoty**
(vzdálenost od hranice jako pravděpodobnost, „0,52 vs. 0,99"), (ii) spojitý,
diferencovatelný výstup, na kterém půjde v cv9 počítat gradient, (iii) hladké
stínování rozhodovací oblasti v grafech.

**i)** **Neposune.** $\mathtt{Sigmoid}(z) = 1/(1+e^{-z/T}) = 0{,}5
\iff z/T = 0 \iff z = 0$ pro každé $T > 0$, a $z = 0$ je právě
$x = -\mathtt{bias}/\mathtt{weight} = x_0$. `temperature` mění **jen strmost**,
nikoli polohu prahu. *(Ověřeno: `weight=1`, `bias=-14.5`, $T = 0{,}5$ dá
v `x = 14.5` přesně 0,5.)*

---

## Příklad 6 — Aktivace, obory hodnot, teplota

**a)**

| aktivace | `output_range` |
|:---|:---|
| `Step` | `(0.0, 1.0)` |
| `Sigmoid` | `(0.0, 1.0)` |
| `Tanh` | `(-1.0, 1.0)` |
| `ReLU` | `(0.0, np.inf)` |
| `ReLU6` | `(0.0, 6.0)` |

**b)** **Dosažené** meze: `Step` (0 pro každé $z<0$, 1 pro každé $z \ge 0$),
`ReLU6` (0 pro $z \le 0$, 6 pro $z \ge 6$), u `ReLU` dolní mez 0 (pro $z \le 0$;
horní „mez" $+\infty$ není hodnota, jen vyjádření neomezenosti).
**Asymptotické** meze: `Sigmoid` a `Tanh` — $0$ a $1$ (resp. $\pm 1$) se nikdy
nenabývají, jen se k nim limitně blíží. Proto jejich docstringy mluví
o **otevřeném** intervalu. Konkrétně $\mathtt{Sigmoid}(-50) \approx
1{,}93 \cdot 10^{-22} > 0$ a $\mathtt{Tanh}(7) = 0{,}99999834 < 1$.

**c)** Pro `z = [-2.0, -0.5, 0.0, 0.5, 3.0, 7.0]`:

| aktivace | výstup |
|:---|:---|
| `Step` | `[0, 0, 1, 1, 1, 1]` |
| `ReLU` | `[0, 0, 0, 0.5, 3, 7]` |
| `ReLU6` | `[0, 0, 0, 0.5, 3, 6]` |

Liší se **jen poslední prvek** ($z = 7$): `ReLU` vrátí 7, `ReLU6` ořeže na 6,
protože $7 > 6$.

**d)**

| `temperature` | `Sigmoid(0.5)` | `Sigmoid(-0.5)` |
|:---:|:---:|:---:|
| $1{,}0$ | $0{,}6224593$ | $0{,}3775407$ |
| $0{,}1$ | $0{,}9933071$ | $0{,}0066929$ |
| $0{,}01$ | $1{,}0000000$ | $\approx 1{,}9\cdot10^{-22} \approx 0$ |

Posloupnost se blíží **`Step(z)`**: 1 pro kladné `z`, 0 pro záporné.

**e)** Čím menší `temperature`, tím strmější přechod — v limitě
$T \to 0^+$ konverguje `Sigmoid` bodově ke `Step` (s jedinou výjimkou $z = 0$,
kde je `Sigmoid` rovna 0,5 pro **každé** $T$). Ukazuje to **blok 3** pipeline
(explorační sweep; `temperature_sweep` v `config.yaml` jde od `0.001` do `0.3`).

**f)** Dva důvody, oba z `output_range = (0.0, np.inf)`:
(i) výstup je **shora neomezený**, takže ho nelze čít jako pravděpodobnost ani
smysluplně prahovat na 0,5 (hodnota 37 „neznamená víc jistoty" v žádné
kalibrované škále);
(ii) na **celé** negativní polopřímce dává `ReLU` přesně 0, takže informace
„jak daleko pod hranicí" se ztratí — bod těsně pod hranicí je nerozlišitelný od
bodu hluboko v druhé třídě, a žádné prahování to nezachrání.

**g)** `Tanh` prahovaná na 0 (neostře) dá `[0, 0, 1, 1, 1, 1]`;
`Sigmoid` prahovaná na 0,5 (neostře) dá `[0, 0, 1, 1, 1, 1]` — a obojí je
totéž jako `Step`, protože všechny tři podmínky jsou ekvivalentní s $z \ge 0$.
Hodnoty pro kontrolu:

| `z` | $-2$ | $-0{,}5$ | $0$ | $0{,}5$ | $3$ | $7$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `Tanh` | $-0{,}9640276$ | $-0{,}4621172$ | $0$ | $+0{,}4621172$ | $+0{,}9950548$ | $+0{,}9999983$ |
| `Sigmoid` | $0{,}1192029$ | $0{,}3775407$ | $0{,}5$ | $0{,}6224593$ | $0{,}9525741$ | $0{,}9990890$ |

Obecný vztah: $\mathtt{Tanh}(z) = 2\,\mathtt{Sigmoid}(2z) - 1$, ekvivalentně
$\mathtt{Sigmoid}(z) = \tfrac{1}{2}\big(1 + \mathtt{Tanh}(z/2)\big)$.
`Tanh` je tedy jen přeškálovaná a posunutá `Sigmoid` — proto stejná rozhodnutí,
jiný obor hodnot.

**h)** Věcně: zmrazený kontrakt slibuje **`np.ndarray`** na vstupu i výstupu.
`list` nemá `.shape` ani `.reshape`, nepodporuje elementwise porovnání ani
`np.where`, takže se rozbije navazující kód — `Neuron.forward` posílá `z` přímo
do `activation.forward`, a `DecisionBoundaryPlotter` výsledek přetváří
(`reshape`) na mřížku pro konturový graf. Prakticky: plotter vyhodnocuje neuron
na desetitisících bodů mřížky, kde je python smyčka o řády pomalejší než jedna
vektorizovaná numpy operace — a vektorizované myšlení je přesně to, co se
v cv9 přenese na celé vrstvy sítě.

**i)** Ze samotných `weights`/`bias` nejde poznat, **jak** se `z` převádí na
výstup — `z=1.0` dá `Step`=1, `Sigmoid`=0,731, `Tanh`=0,762, `ReLU`=1,0,
`ReLU6`=1,0: čtyři různé aktivace na stejném `z` dají čtyři různá čísla, tři
z nich navíc numericky odlišná od `Step`. Bez jména aktivace by `Neuron.load`
musel HÁDAT, kterou z pěti tříd vytvořit — a `weights`/`bias` samy o sobě tuto
informaci nenesou (je to metadata o *chování*, ne o *parametrech*). Konkrétní
příklad: `Neuron(Linear([1.0], bias=0.0), Step())` a
`Neuron(Linear([1.0], bias=0.0), Sigmoid())` mají identické `weights` i
`bias`, ale pro `x=0.2` dají `Step`→`1.0` (tvrdá třída), `Sigmoid`→`0,55`
(pravděpodobnost blízko 0,5, nejistá predikce) — naprosto jiná odpověď.
