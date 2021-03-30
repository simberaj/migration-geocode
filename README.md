# Migration Geocode: a geocoding tool for messy data

This geocoding tool was developed to facilitate geocoding of large datasets containing freeform placenames with lots of typographical errors, mistakes, incoherent formats and inconsistent transcriptions.

It runs on Python 3 in conjunction with a PostgreSQL database with a pg_trgm extension to enable fuzzy matching.

## How to make it run

* Download it to a machine with Python 3 that has geopy and psycopg2 packages installed.
* Modify the conf.json file to contain connection details to your PostgreSQL database.
* Fill the database with data from GeoNames by running

        python load_geonames.py

  on the command line. This will download appx. 400 MB data and require appx. 2 GB of free space on your hard drive. It will create two tables in the target database: `geonames`, containing mappings of location names to location IDs, and `geolocations`, containing further information per location ID.
  
* The geocoder is then run by invoking

        python geocode.py
  
  (use `--help` to show help on command line options)

## Input and output
The geocoder accepts an input headerless CSV file (semicolon delimited) with two columns:

* the first one containing a name (referenced in the `conf.json` file) or an ISO code of a country,
* the second one containing a placename in that country to be geocoded.

The order of these columns can be modified using command line parameters such as

    python geocode.py -s 4 -n 2
    
which will instruct the tool to look for the placename in the second and the country name in the fourth column of the CSV file.

The tool works by applying transcription rules defined in `conf.json` to the input placename and matching the transcribed strings against the PostgreSQL database filled by the GeoNames gazetteer.

The output will be a CSV file with the following columns added:

* Placename and country that was matched.
* Latitude and longitude.
* Approximate coordinate uncertainty (error).
* Source of the geocoding (will be `geonames`).
* Number of places matched for the given placename transcription.
* ID of the matched place in the GeoNames gazetteer.
* GeoNames place type of the matched place.
* Importance of the matched place (equals to the number of inhabitants for populated places).

## Transcription
A key part of the geocoder is the transcriptor which tries to mitigate noise from the input data. It creates several variants of the input placename that can be input as query strings against the GeoNames gazetteer.

The transcription goes as follows:

* A transcription ruleset from `conf.json` is selected based on the country of the placename - different transcription rules will be applied for Ukrainian placenames and Vietnamese placenames, for example.
* Each transcription rule from the ruleset, in the order defined by the configuration, is checked against the placename. If the regular expression matches, each occurrence of it is replaced by its defined transcription. If there are multiple transcription variants defined for the rule, all possible combinations of the transcription are generated.

An example:

* Input placename: hoktemberyan
* Ruleset:
  * `\bho -> o` (`\b` marks beginning or end of word)
  * `e -> e, i`
  * `ya -> a, i`
* Output transcription variants:
  * oktemberin
  * oktembirin
  * oktemberan
  * oktembiran
  * oktimberin
  * oktimbirin
  * oktimberan
  * oktimbiran

If you want to see the actual queries being done to the database, uncomment the `print` function call in `Geocoder.query` (around line 170).

## Fuzzy matching
To further improve the result, instead of matching the gazetteer records exactly, fuzzy trigram matching is applied with the pg_trgm extension to produce strings equal or similar (as measured by the fraction of shared three-character substrings) to the queried transcribed variant. The most similar result is selected, with the place importance indicator from GeoNames is used to break ties.

## Utilities

### Deduplication
The `dedup.py` auxiliary utility can be used to produce a dataset with duplicate rows dropped, in order to avoid repeated geocoding of the same string. If used, the geocoding result must then be reverse-mapped to the original dataset, e.g. by Excel's VLOOKUP.
