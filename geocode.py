import os
import re
import itertools
import argparse
import collections
import sqlite3
import csv
import json
import random

import geopy

DEFAULT_CONF = 'conf.json'
DEFAULT_DBNAME = 'geo.db'

# generic replacement of "-" to space
# remove all meaningful state names from the query, using levenshtein 2
# dot to comma
# in the end, strip all -,. marks


def levenshtein(s1, s2):
  '''Computes Levenshtein distance between two strings.'''
  if len(s1) < len(s2):
    return levenshtein(s2, s1)
  if len(s2) == 0:
    return len(s1)
  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
      current_row = [i + 1]
      for j, c2 in enumerate(s2):
          # j+1 instead of j since previous_row and current_row are one character longer
          insertions = previous_row[j+1] + 1 # than s2
          deletions = current_row[j] + 1       
          substitutions = previous_row[j] + (c1 != c2)
          current_row.append(min(insertions, deletions, substitutions))
      previous_row = current_row
  return previous_row[-1]

# Geonames importance order.
# P (city, village)
# A (country, state, region)
# S (spot, building, farm)
# R (road, railroad)
# L (park, area)
# H (stream, lake)

GN_LESSER_IMPORT = collections.defaultdict(int,  # everything as 0
  {'A' : 100, 'S' : 1, 'R' : 0.5, 'L' : 0.25, 'H' : 0.125})
  
def geonamesImportance(loccat, population=None):
  if loccat == 'P':
    return int(population) + 2
  else:
    return GN_LESSER_IMPORT[loccat]

class GeocoderFarm:
  ADD_INFO_KEYS = ['locerr', 'country', 'source', 'choices', 'id', 'type', 'importance']

  def __init__(self, config, db=None):
    if not isinstance(config, dict): config = self.loadConfig(config)
    self.transcriptors = {}
    for webStat in (True, False):
      self.transcriptors[webStat] = Transcriptor(config, web=webStat)
    self.options = []
    for optDef in config['geocoders']:
      moreconf = {'db' : db} if optDef['name'] == 'geonames' else {}
      self.options.append(self.createOption(optDef, **moreconf))
    self.allOptions = self.options.copy()
    self.contProb = config['continueProbability'] if 'continueProbability' in config else 0
    self.history = collections.defaultdict(list)
    # self.geocoders = {False : [GeonamesGeocoder(db)] if db else [], True : []}
    
  @staticmethod
  def loadConfig(cfile):
    with open(cfile, encoding='utf8') as cfobj:
      return json.load(cfobj)
    
  def createOption(self, optDef, **kwargs):
    isWeb = optDef['web']
    transcriptor = self.transcriptors[isWeb]
    options = {key : value for key, value in optDef.items() if key not in ('name', 'web')}
    options.update(kwargs)
    return GeocoderOption.create(optDef['name'], transcriptor, isWeb, **options)
    
  def toList(self, geoResult):
    return [
      geoResult.address,
      geoResult.latitude,
      geoResult.longitude,
      # self.options[0].getCountryCode(geoResult.raw[])
    ] + [geoResult.raw[key] for key in self.ADD_INFO_KEYS]
  
  def geocodeToList(self, name, country):
    contProb = 1
    i = 0
    name, country = name.strip(), country.strip()
    while i < len(self.options) and random.random() < contProb:
      try:
        # print(name)
        # print(country)
        result = self.options[i].geocode(name=name, country=country)
      except (geopy.exc.GeocoderQuotaExceeded, 
              geopy.exc.GeocoderAuthenticationFailure,
              geopy.exc.GeocoderInsufficientPrivileges,
              geopy.exc.GeocoderUnavailable,
              geopy.exc.GeocoderNotFound) as gexc:
        print('geocoder {} failed, removing; reason: {}'.format(self.options[i].name, gexc))
        self.options.pop(i)
      except (geopy.exc.GeocoderTimedOut,
              geopy.exc.GeocoderParseError,
              geopy.exc.GeocoderQueryError):
        continue # a problem with a particular geocoder, try again later perhaps
      else:
        if result:
          self.record(name, country, result)
          return self.toList(result)
        i += 1
    return [] 
    # self.fromHistory(name, country) # no success, try if something similar succeeded
  
  def record(self, name, country, result):
    self.history[name[0]].append((name, country, result))
  
  def fromHistory(self, fromName, fromCountry):
    if fromName:
      for toName, toCountry, result in self.history[fromName[0]]:
        if levenshtein(fromName, toName) <= 1:
          return self.toList(result)
    return [] # no success

    
# output format: found name, lat, long, locerr, source, choices, id, type, importance
# extractor needs to ensure: locerr, id, type, importance
    
def extractGeonames(result):
  # locerr is unknown, id is direct
  result.raw['locerr'] = None
  result.raw['type'] = result.raw['loctype']
  result.raw['importance'] = result.raw['imp']
  
    
class GeocoderOption:
  EXTRACTORS = {
    'geonames' : extractGeonames
  }
  CACHE_SIZE = 5000

  def __init__(self, geocoder, name, extractor, transcriptor=None, web=True):
    self.geocoder = geocoder
    self.name = name
    self.extractor = extractor # extracts additional info from raw data
    self.transcriptor = transcriptor
    self.useCountry = (name == 'geonames')
    self.web = web
    self.cache = collections.OrderedDict()
    
  @classmethod
  def create(cls, name, transcriptor, web, **kwargs):
    gcClass = GeonamesGeocoder if name == 'geonames' else geopy.geocoders.get_geocoder_for_service(name)
    return cls(gcClass(**kwargs), name, cls.EXTRACTORS[name], transcriptor, web)
  
  def geocode(self, name, country=None):
    if name in self.cache:
      cached = self.cache[name]
      del self.cache[name] # reinsert the used element to the end of the cache
      self.cache[name] = cached
      return cached
    else:
      direct = self.query(name, country)
      if direct is None:
        return None
      final = self.reformat(direct)
      self.cache[name] = final
      if len(self.cache) > self.CACHE_SIZE:
        self.cache.popitem(last=False) # remove the oldest cache element
      return final
    
  def query(self, name, country=None):
    variants = self.transcriptor(name, country)
    if name not in variants: variants.append(name)
    # print(name, country)
    for var in variants:
      # print(var)
      direct = self.geocoder.geocode(var)
      if direct:
        return direct
    return None
  
  def reformat(self, direct):
    if isinstance(direct, geopy.location.Location):
      direct = [direct]
    main = direct[0]
    # main.raw = collections.defaultdict(lambda: None, main.raw)
    self.extractor(main)
    main.raw['choices'] = len(direct)
    main.raw['source'] = self.name
    return main

    
# TODO: EXTRACTORS, geocoder config and api keys for web geocoders
      
class GeonamesGeocoder:
  BASE_QRY = """select *, importance(geolocations.loccat, geolocations.population) as imp
    from geonames, geolocations
    where {}
    and geonames.id=geolocations.id
    order by imp desc;"""

  NORM_WHERE = 'geonames.name=?'
  LEV_WHERE = 'levenshtein(geonames.name, ?) <= ?'
  # COUNTRY_WHERE = ' and geolocations.country=?'
    

  def __init__(self, db):
    self.dbPath = db
    self.db = sqlite3.connect(self.dbPath)
    self.db.row_factory = sqlite3.Row
    self.db.create_function('levenshtein', 2, levenshtein)
    self.db.create_function('importance', 2, geonamesImportance)
    self.cursor = self.db.cursor()
    self.queries = self.assembleQueries()
  
  @classmethod
  def assembleQueries(cls):
    qryDict = {}
    for pars in itertools.product((True, False), repeat=2):
      qryDict[pars] = cls.BASE_QRY.format(
        (cls.LEV_WHERE if pars[0] else cls.NORM_WHERE) # + 
        # (cls.COUNTRY_WHERE if pars[1] else '')
      )
    return qryDict
    
  def geocode(self, query, exactly_one=False, levenshtein=None, country=None):
    qry = self.queries[(bool(levenshtein), bool(country))]
    args = [query.lower()]
    if levenshtein: args.append(levenshtein)
    # if country: args.append(country)
    # print('qry', args)
    self.cursor.execute(qry, args)
    qryres = self.cursor.fetchall()
    if qryres:
      reformed = self.unfold(qryres)
      return reformed[0] if exactly_one else reformed
    else:
      return None
  
  def unfold(self, qryres):
    for res in qryres:
      print(dict(res))
    return [geopy.location.Location(address=(res['country'] + ', ' + res['name'].upper()), point=geopy.location.Point(res['wgslat'], res['wgslon']), raw=dict(res)) for res in qryres]
      
      
  
class Transcriptor:
  '''Performs place name transcriptions to account for the inaccuracies present
  in the input and to match them to the name database (which contains mostly
  English-transcribed names).'''

  DEFAULT_COUNTRY = {
    'code' : None,
    'ruleset' : 'generic',
    'neighbours' : []
  }

  def __init__(self, config, web=False):
    # prepare country lookup and transcription rules from the configuration
    self.countries = config['countries']
    self.countries[None] = self.DEFAULT_COUNTRY
    self.countryNameIndex = self.prepareCountryNameIndex(self.countries)
    self.substitutions = self.prepareSubstitutions(config['substitutions'], web)
    self.failedCountries = set()
    
  def __call__(self, query, cdef=None):
    '''Transcribes a given query using rulesets configured for the given
    country (which may be defined using an ISO code or a name that is in the
    configuration).
    Returns a list of possible transcriptions.'''
    country = self.countries[self.getCountryCode(cdef)]
    toTrans = query.lower()
    results = self.transcribe(toTrans, self.substitutions[country['ruleset']])
    basic = set(results) # retain only unique transcriptions
    # try neighbouring countries too
    for rulesetName in self.getNeighRulesetNames(country):
      neighVars = [item for item in self.transcribe(toTrans, self.substitutions[rulesetName]) if item not in basic]
      results.extend(neighVars)
      basic.update(neighVars)
    return list(results)
    
  def getNeighRulesetNames(self, country):
    '''Returns all rulesets of countries that neighbour the given country.'''
    rulesets = set()
    for neighCode in country['neighbours']: # try neighbouring country rulesets
      try:
        rulesets.add(self.countries[neighCode]['ruleset'])
      except KeyError:
        pass # ignore countries not found
    return rulesets
    
  def getCountryCode(self, cdef):
    '''Given an ISO country code or name specified in config, returns
    the ISO code of that country.
    If the country is not found in config, returns None, which means that
    DEFAULT_COUNTRY is selected.'''
    if cdef is None:
      return None
    elif len(cdef) == 2 and cdef.upper() in self.countries: # country code
      return cdef.upper()
    else:
      try:
        return self.countryNameIndex[cdef.lower()]
      except KeyError:
        if cdef not in self.failedCountries:
          print('Warning: Country {} not found in country list, performance will be reduced'.format(cdef))
          self.failedCountries.add(cdef)
        return None
    
  @classmethod
  def transcribe(cls, start, rules):
    '''Generates all variants of the given start query using the given transcription rules.'''
    current = set([start])
    for regex, joinvars in rules: # apply every substitution rule
      new = set()
      # if joinvars == ['']: print(regex)
      for var in current: # for every current variant before this rule
        parts = regex.sub('@', var).split('@') # find all occurences of the match
        # the @ is used as a hack to avoid re.split which does
        # not split on zero-width patterns such as word boundaries
        # now create all variants by filling the spaces with all permutations
        # of variants mentioned
        new.update(cls.unsplit(parts, joinvars))
        for joinvar in joinvars: 
          new.add(joinvar.join(parts))
      current = new
    return list(current)
          
  @staticmethod
  def unsplit(spl, pattern):
    permno = len(spl) - 1
    # for all possible permutations of replacement variants
    for permut in itertools.product(pattern, repeat=permno):
      # join the spl by permut joiners
      tmp = []
      for i in range(permno):
        tmp.append(spl[i])
        tmp.append(permut[i])
      tmp.append(spl[-1])
      yield ''.join(tmp)

          
  @staticmethod
  def prepareCountryNameIndex(countries):
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
  
  @staticmethod
  def prepareSubstitutions(subConfList, web=False):
    '''Prepares the substitutions for the transcriptor based on the given config.
    Each substitution has a regular expression determining its usage,
    a handful of variants that the transcriptor tries
    and a handful of rulesets that determine for which countries/languages
    it should be used.'''
    prepared = {}
    common = []
    for subConf in subConfList:
      if not web or subConf['web']: # exclude non-web substitutions from web queries
        regex = re.compile(subConf['regex']) # compile for faster matching
        subst = (regex, subConf['variants'])
        if subConf['rulesets'] == 'all':
          common.append(subst)
          for lst in prepared.values():
            lst.append(subst) # all existing rulesets, propagate to the unknown afterwards
        else:
          for ruleset in subConf['rulesets']: # all rulesets that are concerned
            if ruleset not in prepared:
              prepared[ruleset] = common.copy() # all common encountered before
            prepared[ruleset].append(subst)
    return collections.defaultdict(lambda: common.copy(), prepared)

    
class ReaderWriter:
  '''Reads and writes location CSV files.'''

  def __init__(self, geocoder, statecol=-1, namecol=-1):
    self.geocoder = geocoder
    self.statecol = None if statecol < 1 else statecol - 1
    if namecol < 1:
      raise ValueError('invalid name column index: ' + str(namecol))
    self.namecol = namecol - 1
    
  def run(self, input, output, encoding='utf8'):
    todo = []
    for inpath, outpath in self.generatePaths(input, output):
      with open(inpath, encoding=encoding, newline='') as infile:
        dialect = csv.Sniffer().sniff(infile.read(1024))
        infile.seek(0)
        reader = csv.reader(infile, dialect)
        with open(outpath, 'a', encoding=encoding, newline='') as outfile:
          writer = csv.writer(outfile, dialect)
          i = 0
          suc = 0
          for line in reader:
            name, country = line[self.namecol], line[self.statecol]
            geores = self.geocoder.geocodeToList(name, country)
            if geores:
              suc += 1
              writer.writerow(line + geores)
            else:
              todo.append(line)
            i += 1
            print(i, end='\r')
          print('resolving todos', end='\r')
          t = 0
          for line in todo:
            name, country = line[self.namecol], line[self.statecol]
            histres = self.geocoder.fromHistory(name, country)
            if histres:
              suc += 1
            t += 1
            print('resolving todos: {}/{}'.format(t, len(todo)), end='\r')
            writer.writerow(line + histres)
    print(suc, 'out of', i, 'lines geocoded: {:.2%}'.format(suc / i)) 
      
  @classmethod
  def generatePaths(cls, input, output):
    # if the input is a directory, examine all files in it
    # if the output is a directory, create the files with the same names
    # if the output is a file, all the content is merged into it
    if os.path.isdir(input):
      inputs = [os.path.join(input, fname) for fname in os.listdir(input)]
    else:
      inputs = [input]
    if os.path.isdir(output):
      outputs = [os.path.join(output, os.path.basename(inp)) for inp in inputs]
    else:
      if os.path.isfile(output): os.unlink(output)
      outputs = [output] * len(inputs)
    for i in range(len(inputs)):
      yield inputs[i], outputs[i]
    
     
      
      
  
DESCRIPTION = '''Geocodes the provided CSV file with location names and countries.
Tries to accommodate for spelling and transcription mistakes.
Uses a local GeoNames database first (if you don't have it, use load_geonames to
create it). In case of failure, turns to Web geocoding services if they are enabled
(MapQuest, OSM, ArcGIS, Yandex and OpenCage).'''
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('input', metavar='input_file', help='a delimited text file (or folder) with locations and countries')
  parser.add_argument('output', metavar='output_file', help='path to save the output')
  parser.add_argument('-s', '--statecol', metavar='col_index', help='the index of the column with the location countries (set to -1 if not present, default: 1)', default=1, type=int)
  parser.add_argument('-n', '--namecol', metavar='col_index', help='the index of the column with the location names (default: 2)', default=2, type=int)
  parser.add_argument('-c', '--config', metavar='config_file', help='a path to the configuration file (default: {})'.format(DEFAULT_CONF), default=DEFAULT_CONF)
  parser.add_argument('-e', '--encoding', metavar='enc_name', help='name of the encoding of the input files (default: utf8)', default='utf8')
  parser.add_argument('-d', '--db', metavar='file', help='path to the local geo database to use for geocoding (default: {})'.format(DEFAULT_DBNAME), default=DEFAULT_DBNAME)
  args = parser.parse_args()
  geocoder = GeocoderFarm(args.config, args.db)
  io = ReaderWriter(geocoder, statecol=args.statecol, namecol=args.namecol)
  io.run(args.input, args.output, args.encoding)

