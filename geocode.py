import re
import itertools
import argparse
import collections
import csv
import json
import warnings

import geopy
import psycopg2
import psycopg2.extras

DEFAULT_CONF = 'conf.json'

# generic replacement of "-" to space
# remove all meaningful state names from the query, using levenshtein 2
# dot to comma
# in the end, strip all -,. marks

Placename = collections.namedtuple('Placename', 'name country')


class CSVIO:
    MAIN_LOC_FIELDS = ['wgslon', 'wgslat', 'address']
    AUX_LOC_FIELDS = [
        'locerr', 'type', 'population',
        'country_match', 'name_match', 'source', 'id'
    ]
    
    def __init__(self, filename, header=True, encoding='utf8', **csvconf):
        self.csvconf = csvconf
        self.header = header
        self.file = open(filename, self.MODE, encoding=encoding, newline='')
        if not self.csvconf:
            self.csvconf = self._default_csv_config()
        self.gate = self._create_gate()
    
    def _create_gate(self):
        return self.GATES[self.header](self.file, **self.csvconf)

    def _default_csv_config(self):
        return {'delimiter' : ';'}
            
class Reader(CSVIO):
    GATES = {True : csv.DictReader, False : csv.reader}
    MODE = 'r'

    def __init__(self, filename, name_field, country_field=None, **kwargs):
        super().__init__(filename, **kwargs)
        self.name_field = name_field if self.header else int(name_field)
        self.country_field = country_field if self.header else int(country_field)
    
    def __iter__(self):
        for line in self.gate:
            yield Placename(
                line[self.name_field],
                (
                    line[self.country_field]
                    if self.country_field is not None
                    else None
                ),
            ), line

    def _default_csv_config(self):
        dialect = csv.Sniffer().sniff(self.file.read(1024))
        self.file.seek(0)
        return {'dialect' : dialect}
        
        
class HistoryReader(Reader):
    def read(self):
        history = {}
        for placename, line in self:
            if placename not in history and line['address']:
                history[placename] = self._line_to_location(line)
        return history
        
    def _line_to_location(self, line):
        if self.header:
            address = line['address']
            point = geopy.location.Point(line['wgslat'], line['wgslon'])
            raw = {fld : line.get(fld) for fld in self.AUX_LOC_FIELDS}
        else:
            raise NotImplementedError
        return geopy.location.Location(address=address, point=point, raw=raw)

        
class Writer(CSVIO):
    GATES = {True : csv.DictWriter, False : csv.writer}
    MODE = 'w'
    
    def _create_gate(self, fieldnames=None, **csvconf):
        if self.header:
            if fieldnames is None:
                return None
            else:
                self.csvconf['fieldnames'] = fieldnames
                return super()._create_gate()
        else:
            return super()._create_gate() 

    def write(self, location, line):
        if self.gate is None:
            self.gate = self._create_gate(
                fieldnames=self._get_fieldnames(line)
            )
            self.gate.writeheader()
        output = line.copy()
        if location:
            if self.header:
                output.update({
                    'wgslon' : location.longitude,
                    'wgslat' : location.latitude,
                    'address' : location.address
                })
                for key in self.AUX_LOC_FIELDS:
                    output[key] = location.raw.get(key)
            else:
                output += ([
                    location.longitude,
                    location.latitude,
                    location.address
                ] + [
                    location.raw.get(key) for key in self.AUX_LOC_FIELDS
                ])
        self.gate.writerow(output)
    
    @classmethod
    def _get_fieldnames(cls, line):
        # a line can be a dict or a string
        # a location is a geopy.location.Location object
        if hasattr(line, 'keys'):
            return list(line.keys()) + cls.MAIN_LOC_FIELDS + cls.AUX_LOC_FIELDS
        else:
            return None
    
    
class GeocodingEngine:
    def __init__(self, conf_file, **kwargs):
        self.farm = self._create_farm(conf_file, **kwargs)
        self.history = {}
        
    def run(self, source, target, name_field=0, country_field=None, **csvconf):
        i = 0
        reader = Reader(source, name_field, country_field, **csvconf)
        writer = Writer(target, **csvconf)
        for placename, line in reader:
            if placename in self.history:
                location = self.history[placename]
            else:
                location = self.farm.geocode(placename)
                self.history[placename] = location
            writer.write(location, line)
            i += 1
            print(i, ' '*10, end='\r')
    
    def load_history(self, history_file, **csvconf):
        self.history.update(HistoryReader(history_file, **csvconf).read())
    
    @staticmethod
    def _create_farm(conf_file, **kwargs):
        with open(conf_file, encoding='utf8') as cfobj:
            return GeocoderFarm.from_config(json.load(cfobj), **kwargs)
        
            
class GeocoderFarm:
    DEFAULT_RULESET = 'generic'

    def __init__(self, geocoders, country_register, transcriptor=None):
        self.geocoders = geocoders
        self.country_register = country_register
        self.transcriptor = transcriptor
    
    def geocode(self, placename):
        countrycode, ruleset_name = self.country_register.find(placename.country)
        for geocoder in self.geocoders:
            if self.transcriptor:
                ruleset = self.transcriptor.get_ruleset(ruleset_name, geocoder.is_web)
                variants = ruleset.apply(placename.name)
            else:
                variants = [placename.name.lower()]
            if geocoder.is_list:
                result = geocoder.geocode_list(variants, country=countrycode)
                if result:
                    return result
            else:
                for variant in variants:
                    result = geocoder.geocode(variant, country=countrycode)
                    if result:
                        return result
        return None
    
    @classmethod
    def from_config(cls, config, no_transcription=False, **kwargs):
        geocoders = [
            cls._create_geocoder(gdef['name'], gdef['settings'], **kwargs)
            for gdef in config['geocoders']
        ]
        if no_transcription:
            transcriptor = None
        else:
            transcriptor = Transcriptor.from_config(config['substitutions'])
        register = CountryRegister(config['countries'])
        return cls(geocoders, register, transcriptor)
        
    @classmethod
    def _create_geocoder(cls, name, settings, no_fuzzy=False):
        for gclass in cls.GEOCODERS:
            if name in gclass.names:
                return gclass(name, settings, fuzzy=(not no_fuzzy))
                
 

    
# output format: found name, lat, long, locerr, source, choices, id, type, importance
# extractor needs to ensure: locerr, id, type, importance
    
class Geocoder:
    is_list = False
    is_web = True
    
    def geocode(self, name):
        raise NotImplementedError
    
class GeonamesGeocoder:
    is_list = True
    is_web = False
    names = ['geonames']
    minmatch = .6
    FUZZY_QRY = """SELECT
        (
            similarity(name, %(name)s)
            + geonames_importance_log(l.loccat, l.population)
        ) AS crit,
        l.country AS country_match,
        n.name AS name_match,
        l.id, l.wgslat, l.wgslon, l.loctype AS type, l.population
    FROM geonames n, geolocations l
    WHERE
        n.name %% %(name)s
        AND n.id=l.id
        AND l.country=%(country)s
    ORDER BY crit DESC LIMIT 1;
    """    
    EXACT_QRY = """SELECT
        1 + geonames_importance_log(l.loccat, l.population) AS crit,
        l.country AS country_match,
        n.name AS name_match,
        l.id, l.wgslat, l.wgslon, l.loctype AS type, l.population
    FROM geonames n, geolocations l
    WHERE
        n.name=%(name)s
        AND n.id=l.id
        AND l.country=%(country)s
    ORDER BY crit DESC LIMIT 1;
    """

    def __init__(self, name, config, fuzzy=True):
        self.conn = psycopg2.connect(
            **config,
            cursor_factory=psycopg2.extras.DictCursor
        )
        self.cursor = self.conn.cursor()
        if fuzzy:
            self.qry = self.FUZZY_QRY
        else:
            self.qry = self.EXACT_QRY
      
    def geocode(self, name, country):
        result = self._query(name, country)
        return self._unfold(result)
        
    def geocode_list(self, names, country):
        maxcrit = 0
        maxres = None
        for name in names:
            result = self._query(name, country)
            if result:
                crit = result['crit']
                if crit and crit > maxcrit:
                    maxcrit = crit
                    maxres = result
                if maxcrit > 1:
                    break
        return self._unfold(maxres) if (maxres and maxcrit >= self.minmatch) else None
  
    def _query(self, name, country):
        # print(name, country)
        self.cursor.execute(self.qry, {'name' : name, 'country' : country})
        return self.cursor.fetchone()
    
    def _unfold(self, qryres):
        # print('result', qryres)
        return geopy.location.Location(
            address=(qryres['country_match'] + ', ' + qryres['name_match'].upper()),
            point=geopy.location.Point(qryres['wgslat'], qryres['wgslon']),
            raw={'locerr' : None, 'source' : 'geonames', **qryres}
        )
      
    
class GeopyGeocoder(Geocoder):
    names = [] # TODO

    def __init__(self, name, settings):
        self.interface = geopy.geocoders.get_geocoder_for_service(name)(**settings)
    
    def geocode(self, name, country):
        return self.interface.geocode(name) # TODO need proper unfolding and attribute filling
    
    
GeocoderFarm.GEOCODERS = [GeonamesGeocoder, GeopyGeocoder]
  
  
class CountryRegister:
    DEFAULT_COUNTRY = {
        'code' : None,
        'ruleset' : 'generic',
        'neighbours' : []
    }
    
    def __init__(self, countries):
        self.countries = countries
        self.countries[None] = self.DEFAULT_COUNTRY
        self.index = self._prepare_index(self.countries)
        self.failed = set()
    
    @staticmethod
    def _prepare_index(countries):
        '''Prepares the country list from the config for fast name lookups.'''
        cnindex = {}
        for cdict in countries.values():
            ccode = cdict['code']
            for key in cdict:
                if key.startswith('name'):
                    if isinstance(cdict[key], str):
                        cnindex[cdict[key]] = ccode
                    else:
                        for name in cdict[key]:
                            cnindex[name.lower()] = ccode
        return cnindex
        
    def code(self, cdef):
        if cdef is None:
            return None
        elif len(cdef) == 2 and cdef.upper() in self.countries: # country code
            return cdef.upper()
        else:
            try:
                return self.index[cdef.lower()]
            except KeyError:
                if cdef not in self.failed:
                    warnings.warn(
                        'country {} not found in country list,\
                        performance will be reduced'.format(cdef)
                    )
                    self.failed.add(cdef)
                return None
        
    def find(self, cdef):
        ccode = self.code(cdef)
        return ccode, self.ruleset(ccode)
    
    def ruleset(self, ccode):
        ruleset = self.countries[ccode]['ruleset']
        return 'generic' if ruleset == 'slavic' else ruleset # hack
    
        
class Replacement:
    def apply(self, input):
        # returns a list of strings in order of probability
        raise NotImplementedError
        
    @staticmethod
    def unsplit(spl, pattern):
        # for all possible permutations of replacement variants
        yield from (''.join([
                val
                for pair in zip(spl[:-1], permut)
                    for val in pair
            ] + [spl[-1]]
        ) for permut in itertools.product(pattern, repeat=(len(spl) - 1)))
    
    @classmethod
    def from_config(cls, config):
        variants = config['variants']
        if 'regex' in config:
            if any('\\' in var for var in variants):
                cl = DynamicRegexReplacement
            else:
                cl = StaticRegexReplacement
            return cl(config['regex'], variants)
        else:
            search = config['string']
            if len(variants) == 1:
                return SimpleStaticStringReplacement(search, variants[0])
            else:
                return VariantStaticStringReplacement(search, variants)

      
class SimpleStaticStringReplacement(Replacement):
    def __init__(self, search, replace):
        self.search = search
        self.replace = replace
    
    def apply(self, input):
        yield input.replace(self.search, self.replace)
    
    def __repr__(self):
        return '<Repl: {0.search!r} -> {0.replace!r}>'.format(self)
     
class VariantStaticStringReplacement(Replacement):
    def __init__(self, search, replacements):
        self.search = search
        self.replacements = replacements
    
    def apply(self, input):
        parts = input.split(self.search)
        yield from self.unsplit(parts, self.replacements)

    def __repr__(self):
        return '<VarRepl: {0.search!r} -> {0.replacements!r}>'.format(self)
     
        
class StaticRegexReplacement(Replacement):
    def __init__(self, regex, replacements):
        self.pattern = re.compile(regex)
        self.replacements = replacements
        
    def apply(self, input):
        parts = self.pattern.sub('@', input).split('@')
        # the @ is used as a hack to avoid re.split which does
        # not split on zero-width patterns such as word boundaries
        yield from self.unsplit(parts, self.replacements)
    
    def __repr__(self):
        return '<VarRegex: {0.pattern!r} -> {0.replacements!r}>'.format(self)
     
    
class DynamicRegexReplacement(StaticRegexReplacement):
    def apply(self, input):
        variants = []
        start = 0
        for match in self.pattern.finditer(input):
            variants.append([input[start:match.start()]])
            variants.append([match.expand(var) for var in self.replacements])
            start = match.end()
        if start != len(input):
            variants.append([input[start:]])
        nvars = 1
        for varlist in variants:
            nvars *= len(varlist)
        for i in range(nvars):
            ix = i
            variant = []
            for item in variants:
                variant.append(item[ix % len(item)])
                ix //= len(item)
            yield ''.join(variant)
    
        
class Ruleset:
    def __init__(self, name, is_web, rules=None, maxvars=100):
        self.name = name
        self.is_web = is_web
        self.rules = rules if rules else []
        self.maxvars = maxvars
    
    def add(self, rule):
        self.rules.append(rule)
        
    def apply(self, string):
        '''Generates all variants of the given start query using the given transcription rules.'''
        string = string.lower()
        current = [string]
        for rule in self.rules: # apply every substitution rule
            new = []
            for var in current: # for every current variant before this rule
                for newvar in rule.apply(var):
                    if newvar not in new:
                        new.append(newvar)
            current = new
        current = current[:self.maxvars]
        if string not in current:
            current.append(string)
        return current
    
        
class Transcriptor:
    def __init__(self, rulesets):
        self.rulesets = rulesets
    
    def get_ruleset(self, name, is_web):
        return self.rulesets[(name,is_web)]
    
    
    @classmethod
    def from_config(cls, substitutions):
        ruleset_names = cls._find_all_ruleset_names(substitutions)
        rulesets = {
            (name, is_web) : Ruleset(name, is_web)
            for name in ruleset_names
                for is_web in (True, False)
        }
        for ruledef in substitutions:
            rule = Replacement.from_config(ruledef)
            setnames = ruleset_names if ruledef['rulesets'] == 'all' else ruledef['rulesets']
            for setname in setnames:
                rulesets[setname,False].add(rule)
                if ruledef['web']:
                    rulesets[setname,True].add(rule)
        return cls(rulesets)
    
    @staticmethod
    def _find_all_ruleset_names(substitutions):
        return list(set(
            name
            for rule in substitutions
                for name in rule['rulesets']
                if rule['rulesets'] != 'all'
        ))
        
  
      
  
DESCRIPTION = '''Geocodes the provided CSV file with location names and countries.
Tries to accommodate for spelling and transcription mistakes.
Uses a local PostgreSQL database (if you don't have it, use load_geonames to
create it).'''
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input', metavar='input_file', help='a delimited text file with locations and countries')
    parser.add_argument('output', metavar='output_file', help='path to save the output')
    parser.add_argument('-c', '--countrycol', metavar='col_key', help='zero-based index or name of the column with the location countries (default: 1)', default=1)
    parser.add_argument('-n', '--namecol', metavar='col_key', help='zero-based index or name of the column with the location names (default: 2)', default=2)
    parser.add_argument('-C', '--config', metavar='config_file', help='a path to the configuration file (default: {})'.format(DEFAULT_CONF), default=DEFAULT_CONF)
    parser.add_argument('-e', '--encoding', metavar='enc_name', help='name of the encoding of the input files (default: utf8)', default='utf8')
    parser.add_argument('-H', '--history', metavar='history_file', help='a delimited text file with already geocoded locations')
    parser.add_argument('-l', '--headerless', action='store_false', help='the file does not have field names, use indexes to determine column positions', dest='header')
    parser.add_argument('-T', '--no-transcription', action='store_true', help='do not use transcription rules')
    parser.add_argument('-F', '--no-fuzzy', action='store_true', help='do not use fuzzy string search')
    args = parser.parse_args()
    engine = GeocodingEngine(
        args.config,
        no_transcription=args.no_transcription,
        no_fuzzy=args.no_fuzzy
    )
    if args.history:
        engine.load_history(
            args.history,
            name_field=args.namecol,
            country_field=args.countrycol,
            encoding=args.encoding
        )
    engine.run(
        args.input,
        args.output,
        name_field=args.namecol,
        country_field=args.countrycol, 
        encoding=args.encoding,
        header=args.header
    )
