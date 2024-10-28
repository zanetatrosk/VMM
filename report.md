# VMM Závěrečná zpráva

**20. 10. 2024, Filip Čihák a Žaneta Trošková**

Téma: Zpracování dat s využitím neuronových sítí

## Popis projektu

V rámci projektu byla vytvořena webová aplikace pro klasifikaci psího plemene z fotografie, která využívá konvoluční neuronovou síť implementovanou pomocí frameworku Keras/TensorFlow v jazyce Python. Aplikace pak pro používá backendovou část Python framework Django a pro frontendovou část framework React s jazykem TypeScript. Uživatel tak nahraje fotografii psa a vybere model který chce k rozpoznání využit. Aplikace poté pomocí zvoleného modelu vyhodnotí plemeno, kterému je pes na fotografii vlastnostmi nejpodobnější.

## Způsob řešení

### Tvorba datasetu

Prvním krokem pro vytrénovaní funkčního modelu pomocí neuronové sítě je získání dostatečně robustního datasetu, v tomto případě tedy dostatek reprezentativních fotografií psů různých plemen anotovaných správným plemenem.

Zprvu se jevil jako vhodný dataset Stanford Dogs {footnote: http://vision.stanford.edu/aditya86/ImageNetDogs/}, rozdělený do 120 plemen, přičemž ke každému plemeni je k dispozici přibližně 150 fotografií. První experimentace ovšem ukázala, že pro vytrénování vlastní CNN s tolika třídami by bylo potřeba řádově vyšší množsví vzorků v datasetu, jelikož se dařilo model úspešně učit jen několik málo epoch a validační přesnost ve všech testovaných konfiguracích dosahovala pouze několika málo procent.

Proběhlo tak hledání dalších dat k využití pro trénování, a pro populárnější psí plemena se podařilo na platformě Kaggle najít dalších několik datasetů. {footnote: konkrétně https://www.kaggle.com/datasets/aseemdandgaval/23-pet-breeds-image-classification, https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset, https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set} Fotografie shodujících se plemen tak byly zkombinovány a vznikl tak dataset celkem 8 500 vzorků rozdělených poměrně rovnoměrně mezi 16 plemen oblíbených mezi majiteli psů. Na nich se již ukázalo jako možné vytvořit model splňující úkol alespoň s hrubou přesností.

### Architektura neuronové sítě

Po konzultaci se cvičícím bylo upřesněno, že by měla být vytrénována vlastní konvoluční neuronová síť bez použití metody transfer learningu s využitím checkpointu jiné neuronové sítě. K samotné klasifikaci byla tak byla vytvořena CNN se standardní architekturou využívající nejprve posloupnost konvolučních vrstev s následným max-poolingem, následovaných plně propojenými vrstvami. Jako aktivační funkce byla použita ReLU. Poslední plně propojená vrstva pak počtem neuronů odpovídá počtu tříd, které má model klasifikovat, a využívá aktivační funkci softmax, která vrací pravděpodobnost, že vstupní vzorek patří do jednotlivých tříd. K trénování byl využit optimizační algoritmus Adam a jako loss funkce byla použita SparseCategoricalCrossentropy. 

Ve snaze o potlačení tzv. overfittingu, tedy situaci, kdy se model v trénovacích datech zafixuje na nevhodné vzory a následně se mu nedaří generalizovat na nová data, byly využity techniky augmentace dat a dropout vrstev. Augmentace dat je provedena před samotným trénováním a zahrnuje náhodné překlopení, rotaci a zvětšení vstupních fotografií. Dropout vrstvy pak náhodně vypínají některé neurony v průběhu trénování s cílem zamezit přílišné specializaci jednotlivých neuronů na trénovací data.

Dále bylo dbáno na konfigurovatelnost architektury CNN, parametrů trénování i předzpracování dat, aby bylo možné snadno experimentovat s různými konfiguracemi a zjišťovat, jak se jednotlivé změny projeví na výsledném modelu.

## Implementace

### Příprava prostředí

Jelikož je trénování neuronové sítě výpočetně náročné, je vhodné k tomuto účelu využít dedikovanou grafickou kartu (GPU), na níž lze za použití ovladačů provádět rychle paralelní výpočty využíváné mj. v machine learningu. Autoři disponují kartou AMD Radeon RX 6800, ale TensorFlow bohužel oficiálně nenabízí verzi schopnou využít GPU od AMD. Byl proto využit balíček TensorFlow DirectML {footnote: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin}, jehož API je podporováno všemi moderními grafickými kartami. Kromě TensorFlow je pak využito knihoven numpy a matplotlib (pouze pro vizualizaci dat a vykreslení výsledků učení), které je dostačující instalovat pomocí balíčkovacího systému pip.

### Skript pro trénování modelu

Pro celý proces trénování modelu byl vytvořen skript train.py. Ten provádí následující kroky:

1. Načtení datasetu, rozdělení na trénovací a validační množinu a augmentace dat trénovací množiny pomocí třídy ImageDataGenerator z Keras. Fotografie jsou navíc zmenšeny na požadovanou velikost pro trénování a hodnoty pixelů jsou normalizovány na interval [0, 1], který je vhodnější pro strojové učení. Skript poté vypíše informace o počtu tříd a vzorků v obou množinách.
2. Přípravné kroky: uložení seznamu tříd do souboru v pořadí odpovídajícímu indexům výstupní vrstvy modelu a případně zobrazení několika fotografií z datasetu pro kontrolu.
3. Definice modelu: pomocí třídy Sequential z Keras jsou definovány po sobě jdoucí vrstvy modelu. Dále jsou definovány pomocné callback funkce pro ukládání modelu při dosažení lepších výsledků po epoše trénování, pro zpomalení rychlosti učení pro případ, kdy se již jinak nedaří minimalizovat loss funkci, a pro zastavení trénování, pokud již model po větším počtu epoch nevykazuje zlepšení.
4. Kompilace a trénování: Model je kompilován s optimizačním algoritmem a loss funkcí popsanou výše a je zahájeno trénování tak, že framework vypisuje průběh epoch, jejich výsledky (tedy dosaženou hodnotu loss funkce a přesnost modelu jak v trénovácí, tak validační fázi) a aktivaci výše definovaných callback funkcí v rámci trénování. Na konci trénování jsou tyto informace případně vykresleny do grafu.

Skript je napsán tak, že veškeré konfigurace se provádí nastavením konstant na začátku souboru. Dataset je koncipován tak, že v uvedené složce se očekává jedna podsložka pro každou klasifikační třídu, která obsahuje všechny fotografie do ní spadající. Složka pro uložení modelu pouze musí existovat a je do ní uložen výsledný model a seznam tříd. Ostatní parametry jsou číselné a je možné je měnit dle potřeby. Dále se předáním argumentu plot při spouštění skriptu (tedy např. python train.py plot) vyvolá interaktivní vykreslení grafů (viz popis výše).

### Webová aplikace

#### Instalace a spuštění webové aplikace

Pro spuštění backend části webové aplikace je potřeba mít nainstalovaný Python(3.8.10) a doinstalovat framework Django a další potřebné knihovny (Numpy, TensorFlow, Keras, Django REST framework, Pillow). Spuštění backend části aplikace se provede příkazem `python manage.py runserver`.

Pro spuštění frontend části webové aplikace je potřeba mít nainstalovaný Node.js (18 a vyšší) a doinstalovat React a další potřebné knihovny příkazem `npm install`. Následné spuštění frontend části aplikace se provede příkazem `npm start`. Aplikace bude dostupná na adrese `http://localhost:3000`.

Po spuštění aplikace se objeví formulář, kde má uživatel možnost nahrát obrázek psa a pole s vybráním modelu, který bude použit na rozpoznání plemene. 
Aplikace nabízí následující modely:
- 16 breeds model - model, který byl trénován na 16 plemenech psů, má přesnost na cca 50%.
- TODO

Po vybrání konkrétního modelu se uživateli zobrazí seznam plemen, které model rozpoznává.
Po stisknutí tlačítka "Upload" se u6ivateli zobrazí výsledek v podobě tabulky 10 nejpravděpodobnějších plemen psů, které by mohlo být na obrázku seřazených sestupně podle pravděpodobnosti.

#### Ukázka aplikace

Na obrázku je vidět ukázka webové aplikace, kde byl nahrán obrázek psa a vybrán model 16 breeds model. Aplikace vrátila tabulku s 10 nejpravděpodobnějšími plemeny psů, které by mohlo být na obrázku. Aplikace správně určila, že na obrázku je pes plemene Golden Retriever (i když jen s pravděpodobností 27%).

![Dog Breed Classifier](example.png)

## Experimentace

V následující 

16 plemen ()

| Velikost obrázků | Konvoluční vrstvy (počty filtrů) | Skryté vrstvy (počty neuronů) | Validační přesnost |
|---|---|---|---|
| 224x224 | 64-128-256 | Do-64 | 0.439 |
| 224x224 | 64-128-256 | Do-64-32 | 0.438 |
| 224x224 | 64-128-256 | Do-128-64-32 | 0.412 |
| 224x224 | 64-128-256 | Do-128-64-Do-32 | 0.423 | <- saved magicky vytvořený model s 0.56 přesností (velikost 160x160)
| 224x224 | 64-128-256 | Do-256-128-Do-64-32 | 0.381 |
| 224x224 | 64-128-256 | Do-512-256-Do-128-64-32 | 0.436 |

| 160x160 | 

| 224x224 | 16-32-64 | Do-64-32 | 0.435 |

8 plemen (beagle, boxer, golden_retriever, husky, poodle, pug, rottweiler, yorkshire_terrier)

| Velikost obrázků | Konvoluční vrstvy (počty filtrů) | Skryté vrstvy (počty neuronů) | Validační přesnost |
|---|---|---|---|
| 160x160 | 16-32-32 | Do-32 | 0.565 |
| 160x160 | 16-32-32 | Do-64-32 | 0.576 | <- saved
| 160x160 | 32-64-128 | Do-64-32 | 0.580 |
| 160x160 | 16-32-32 | Do-128-96 | 0.550 |
| 160x160 | 16-32-32 | Do-16 | 0.395 |
| 224x224 | 16-32-32 | Do-64-32 | 0.567 |
| 160x160 | 16-32-32 | Do-30-18-12 | 0.470 |
| 160x160 | 32-32-64-64 | Do-64-32 | 0.611 | <- saved as v2

## Diskuse a závěr

TODO
