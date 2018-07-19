import argparse
import json
import sys
import io
import os
import tempfile
import urllib.request
import zipfile
import csv

import psycopg2

GEONAME_URI = 'http://download.geonames.org/export/dump/allCountries.zip'
GEONAME_INNER = 'allCountries.txt'
CHUNK_SIZE = 10000000
PROG_DIR = os.path.dirname(__file__)

FULL_COUNTRIES = {'CZ', 'KG', 'BY', 'KZ', 'LV', 'LT', 'EE', 'GE', 'HU', 'MD', 'PL', 'RU', 'US', 'UA', 'VN', 'AZ', 'AM', 'UZ', 'TJ', 'RO', 'SK'}
MIN_NONFULL_POP = 10000
MAXORD = 880 # disregard non-latin and exotic characters

def readSQL(path):
    with open(path) as sqlfile:
        text = sqlfile.read()
    return [command.replace('\n', ' ').strip() for command in text.split(';') if command.strip() and not command.replace('\n', ' ').strip().startswith('--')]

DBINIT = readSQL(os.path.join(PROG_DIR, 'dbinit.sql'))
DBPOSTFILL = readSQL(os.path.join(PROG_DIR, 'dbpostfill.sql'))

# print(DBINIT)
# print(DBPOSTFILL)
# raise RuntimeError

with open(os.path.join(PROG_DIR, 'dbconf.json')) as dbConnConfFile:
    dbconf = json.load(dbConnConfFile)

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

def csvExtract(fname, sourceName=None):
    print('Extracting archive...')
    with zipfile.ZipFile(fname, 'r') as gnzip:
        innerName = os.path.splitext(os.path.basename(sourceName if sourceName else fname))[0] + '.txt'
        with gnzip.open(innerName) as gnraw:
            with io.TextIOWrapper(gnraw, encoding='utf8') as csvfile:
                i = 0
                for line in csvfile:
                    yield line.strip().split('\t')
                    i += 1
                    if i % 10000 == 0:
                        print('{} lines processed\r'.format(i), end='')

def getGeonames(localName=None, remoteName=None):
    if localName is None:
        if remoteName is None:
            remoteName = GEONAME_URI
        sourceName = remoteName.split('/')[-1]
        try:
            handle, fname = tempfile.mkstemp()
            os.close(handle)
            download(remoteName, fname)
            yield from csvExtract(fname, sourceName=sourceName)
        finally:
            if fname is not None:
                os.unlink(fname)
    else:
        yield from csvExtract(localName)
      
# geonameid         : integer id of record in geonames database
# name              : name of geographical point (utf8) varchar(200)
# asciiname         : name of geographical point in plain ascii characters, varchar(200)
# alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
# latitude          : latitude in decimal degrees (wgs84)
# longitude         : longitude in decimal degrees (wgs84)
# feature class     : see http://www.geonames.org/export/codes.html, char(1)
# feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
# country code      : ISO-3166 2-letter country code, 2 characters

     
NAME_COLS = ('name', 'id')
LOC_COLS = ('id', 'wgslat', 'wgslon', 'loccat', 'loctype', 'country', 'population')
     
def fillTempFiles(geonamesGenerator, nameFile, locFile):
    nameWriter = csv.writer(nameFile, delimiter=';')
    locWriter = csv.writer(locFile, delimiter=';')
    for row in geonamesGenerator:
        nameRows, locRow = parseGeonames(row)
        if nameRows is not None:
            # print(nameRows, locRow)
            for nameRow in nameRows:
                nameWriter.writerow(nameRow)
            locWriter.writerow(locRow)
            # raise RuntimeError
    nameFile.seek(0)
    locFile.seek(0)
     
def createGeonamesDatabase(dbConf, localName=None, remoteName=None):
    with tempfile.TemporaryFile() as nameFile, tempfile.TemporaryFile() as locFile:
        nameFileStr = io.TextIOWrapper(nameFile, encoding='utf8', newline='')
        locFileStr = io.TextIOWrapper(locFile, encoding='utf8', newline='')
        print('Creating import files...')
        fillTempFiles(getGeonames(localName, remoteName), nameFileStr, locFileStr)
        dbConn = psycopg2.connect(**dbConf)
        try:
            cursor = dbConn.cursor()
            for sql in DBINIT:
                cursor.execute(sql)
            print('Importing geonames...')
            copyTable(cursor, nameFile, 'geonames')
            print('Importing geolocations...')
            copyTable(cursor, locFile, 'geolocations')
            dbConn.commit()
            print('\nCreating lookup indexes (might take several minutes)...')
            cursor = dbConn.cursor()
            for sql in DBPOSTFILL:
                cursor.execute(sql)
            print('Done.')
        finally:
            dbConn.close()

def copyTable(cursor, file, tablename):
    sql = '''COPY {} FROM STDIN WITH (FORMAT CSV, DELIMITER ';', NULL '', QUOTE '\"')'''.format(tablename)
    cursor.copy_expert(sql, file)
            
def parseGeonames(row):
    id, name, ascii, altNameCSV, wgslat, wgslon, category, type, country = row[:9]
    population = row[14]
    if country not in FULL_COUNTRIES and (not population or int(population) < MIN_NONFULL_POP):
        return None, None
    names = set([name.strip(), ascii.strip()] + [n.strip() for n in altNameCSV.split(',')])
    if '' in names: names.remove('')
    return (
        [(n.lower(), int(id)) for n in names if all(ord(c) < MAXORD for c in n)],
        (int(id), float(wgslat), float(wgslon), category, type, country, int(population)),
    )
    
DESCRIPTION = '''Fills a PostgreSQL database with GeoNames data for geocoding.

Performs automatic download when necessary or uses local file if -f is given.
Requires appx. 2 GB of space for the database.
'''
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('-f', '--file', metavar='zipfile', help='a path to the GeoNames zip file to use')
  parser.add_argument('-r', '--remote', metavar='url', help='a URL of the GeoNames zip file to use')
  args = parser.parse_args()
  createGeonamesDatabase(dbconf, args.file, args.remote)