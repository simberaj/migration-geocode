import argparse
import sys
import io
import os
import tempfile
import sqlite3
import urllib.request
import zipfile
import csv

from geocode import DEFAULT_DBNAME

GEONAME_URI = 'http://download.geonames.org/export/dump/allCountries.zip'
GEONAME_INNER = 'allCountries.txt'
DBINIT_SCRIPT_LOC = 'dbinit.sql'
DBPOSTFILL_SCRIPT_LOC = 'dbpostfill.sql'
# DEFAULT_DBNAME = 'geo.db'
CHUNK_SIZE = 1000000

with open(DBINIT_SCRIPT_LOC) as dbInitFile:
  DBINIT_SCRIPT = dbInitFile.read()

with open(DBPOSTFILL_SCRIPT_LOC) as dbInitFile:
  DBPOSTFILL_SCRIPT = dbInitFile.read()

def download(source, target):
  print('Downloading Geonames...')
  with open(target, 'wb') as tgtfile:
    with urllib.request.urlopen(source) as srcfile:
      chunk = srcfile.read(CHUNK_SIZE)
      size = CHUNK_SIZE
      while chunk:
        print('{} bytes downloaded\r'.format(size), end='')
        tgtfile.write(chunk)
        chunk = srcfile.read(CHUNK_SIZE)
        size += CHUNK_SIZE

def csvExtract(fname):
  print('Extracting archive...')
  with zipfile.ZipFile(fname, 'r') as gnzip:
    innerName = os.path.splitext(os.path.basename(fname))[0] + '.txt'
    with gnzip.open(innerName) as gnraw:
      with io.TextIOWrapper(gnraw, encoding='utf8') as csvfile:
        # reader = csv.reader(csvfile, delimiter='\t')
        # for line in reader:
        i = 0
        for line in csvfile:
          yield line.strip().split('\t')
          i += 1
          if i % 10000 == 0:
            print('{} lines processed\r'.format(i), end='')

def getGeonames(fname=None):
  if fname is None:
    try:
      handle, fname = tempfile.mkstemp()
      os.close(handle)
      download(GEONAME_URI, fname)
      yield from csvExtract(fname)
    finally:
      if fname is not None:
        os.unlink(fname)
  else:
    yield from csvExtract(fname)
      
# geonameid         : integer id of record in geonames database
# name              : name of geographical point (utf8) varchar(200)
# asciiname         : name of geographical point in plain ascii characters, varchar(200)
# alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
# latitude          : latitude in decimal degrees (wgs84)
# longitude         : longitude in decimal degrees (wgs84)
# feature class     : see http://www.geonames.org/export/codes.html, char(1)
# feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
# country code      : ISO-3166 2-letter country code, 2 characters

def insertNames(cursor, id, names):
  # print(id, str(names).encode('cp852', errors='ignore'))
  for name in names:
    cursor.execute('insert into geonames(name, id) values (?,?)', (name, id))

def insertLoc(cursor, *args):
  # print(str(args).encode('cp852', errors='ignore'))
  cursor.execute('''insert into geolocations(
    id, wgslat, wgslon, loccat, loctype, country, population
    ) values (?, ?, ?, ?, ?, ?, ?);''', args)
     
     
def createGeonamesDatabase(dbName, sourceName=None):
  dbConn = sqlite3.connect(dbName)
  # dbConn.isolation_level = None # autocommit every command
  try:
    dbConn.executescript(DBINIT_SCRIPT)
    cursor = dbConn.cursor()
    for row in getGeonames(sourceName):
      id, name, ascii, altNameCSV, wgslat, wgslon, category, type, country = row[:9]
      population = row[14]
      names = set([name.strip(), ascii.strip()] + [n.strip() for n in altNameCSV.split(',')])
      if '' in names: names.remove('')
      insertNames(cursor, int(id), (n.lower() for n in names))
      insertLoc(cursor, int(id), float(wgslat), float(wgslon), category, type, country, int(population))
    dbConn.commit()
    print('\nCreating lookup indexes (might take several minutes)...')
    dbConn.executescript(DBPOSTFILL_SCRIPT)
    print('Done.')
  finally:
    dbConn.close()
    
DESCRIPTION = '''Fills a SQLite database with GeoNames data for geocoding.
Performs automatic download when necessary.
Requires appx. 2 GB of space for the database.'''
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('-d', '--dbname', metavar='dbname', help='a path to the database to fill (if it does not exist, it will be created)', default=DEFAULT_DBNAME)
  parser.add_argument('-f', '--file', metavar='zipfile', help='a path to the GeoNames zip file to use')
  args = parser.parse_args()
  # import cProfile
  # cProfile.run('createGeonamesDatabase(args.dbname, args.file)')
  createGeonamesDatabase(args.dbname, args.file)