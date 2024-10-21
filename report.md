# VMM Závěrečná zpráva

**20. 10. 2024, Filip Čihák a Žaneta Trošková**

Téma: Zpracování dat s využitím neuronových sítí

## Popis projektu

V rámci projektu byla vytvořena webová aplikace pro klasifikaci psího plemene z fotografie, která využívá konvoluční neuronovou síť implementovanou pomocí frameworku Keras/TensorFlow v jazyce Python. Aplikace pak pro používá backendovou část Python framework Django a pro frontendovou část framework React s jazykem TypeScript. Uživatel tak nahraje fotografii psa a aplikace vyhodnotí plemeno, kterému je pes na fotografii vlastnostmi nejpodobnější.

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



## Experimentace

TODO

## Diskuse a závěr

TODO
