Geokódovací nástroj pro nespolehlivá data. Vyvinuto pro potřeby geokódování migrační databáze s velkým množstvím překlepů, chyb, nesourodých formátů a nekonzistentních transkripcí.

Nástroj běží na Pythonu 3, bez nutnosti instalovat další balíčky.

== Jak to zprovoznit

* Stáhnout obsah repozitáře do složky.
* Nainstalovat Python 3 (potvrdit, že instalace má být umístěna v proměnné PATH).
* Spustit generování geokódovací databáze příkazem (v příkazové řádce)

    python load_geonames.py

  Z webu se přitom stáhne cca 400 MB dat, je třeba cca 2 GB volného místa.
  
* Zobrazit si nápovědu ke spouštění příkazem

    python geocode.py --help

== Běh nástroje
Nástroj přijímá na vstupu bezhlavičkový CSV soubor, kde:

* jeden sloupec obsahuje název nebo ISO kód státu, v němž se místo nachází,
* další sloupec obsahuje název místa, které má být geokódováno.

Pořadí těchto sloupců lze určit volbami na příkazové řádce, např.

    python geocode.py -s 4 -n 2
    
určí, že název místa se nachází v druhém a název státu ve čtvrtém sloupci.

Pomocí dotazů do geokódovací databáze za pomoci transkriptoru (viz níže) se nástroj snaží geokódovat předaná místa. Nástroj vytváří na výstupu CSV soubor, v němž jsou přidány vpravo následující sloupce:
* Název a stát nalezeného místa.
* Zeměpisnou šířku a zeměpisnou délku.
* Chybu určení místa.
* Zdroj geokódování.
* Počet míst nalezených pro daný název.
* ID daného místa v geokódovacím zdroji.
* Typ místa z geokódovacího zdroje.
* Důležitost místa (u obydlených míst počet obyvatel).

== Transkriptor
V souboru `geocode.py` je obsažen transkriptor, který se snaží odstranit chyby ve vstupních datech. Ten ke každému dotazu vytvoří několik upravených variant, které potom vstupují jako dotazy do geokódovací databáze.

Varianty se vytvářejí následujícím postupem:

* Na základě udané země (státu) se dle konfigurace z `conf.json` vybere transkripční ruleset - tedy všechna pravidla, která se použijí pro transkripci (pro Ukrajinu je jiný než pro Vietnam).
* Pro každé pravidlo se vygenerují všechny varianty jeho aplikace a ty se vrátí. Pravidlo je definováno vstupem (regex), což je regulární výraz určující, na jakou část textu má být použito, a variantami, tedy všemi možnostmi, kterými se nahradí každý výskyt regexu v dotazu.

Příklad:
* Vstup: hoktemberyan
* Pravidla:
** `\bho -> o` (`\b` značí začátek či konec slova)
** `e -> e, i`
** `ya -> a, i`
* Varianty na výstupu:
** oktemberin
** oktembirin
** oktemberan
** oktembiran
** oktimberin
** oktimbirin
** oktimberan
** oktimbiran

Pokud chcete za běhu vidět, jaké dotazy jsou po transkripci posílány do databáze, stačí odkomentovat funkce `print` v metodě `Geocoder.query` (kolem řádku 170).

== Geokódovací databáze
Před spuštěním programu je nejprve nutné vytvořit geokódovací databázi (soubor `geo.db`). Ta se sama stáhne z webu Geonames.org - jde o globální databázi jmen a souřadnic.

=== Testování
Pro testování toho, co databáze obsahuje, můžete využít utilitku sqltest.py. Stačí ji v Pythonu spustit a zadávat SQL příkazy. Databáze obsahuje dvě (obrovské) tabulky, v jedné jsou jména a ID lokací (protože jedna lokace může mít více jmen), v druhé ID lokací a další informace včetně souřadnic. Takže pro testování úspěšnosti dotazu lze použít např.

    select * from geonames left join geolocations on geonames.id=geolocations.id where name='praha';

Pozor, všechny názvy jsou uloženy malými písmeny (lowercase)!

== Nástroje

=== Deduplikace
Přiložena je utilitka `dedup.py`, která z daného souboru vytvoří nový soubor, kde jsou již pouze unikátní řádky. Hodí se pro preprocessing excelovských souborů, aby se opakované záznamy nemusely geokódovat dvakrát. Nakonec je samozřejmě třeba geokódované výsledky zpět najoinovat - např. pomocí funkce VLOOKUP v Excelu, kde se dá otevřít výstupní CSV soubor.