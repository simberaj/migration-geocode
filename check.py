import sys

import numpy
import pandas as pd

RADIUS = 6378

def distance(la1, fi1, la2, fi2):
    return 2 * RADIUS * numpy.arcsin(numpy.sqrt(
        numpy.sin(numpy.radians(fi2 - fi1)) ** 2
        + numpy.cos(numpy.radians(fi1)) * numpy.cos(numpy.radians(fi2)) * numpy.sin(numpy.radians(la2 - la1)) ** 2
    ))

assdf = pd.read_csv(sys.argv[1], sep=';')
refdf = pd.read_csv(sys.argv[2], sep=';').drop_duplicates()

maindf = pd.merge(assdf, refdf, on=('birth_country', 'birth_place'), how='left')

maindf['country'] = maindf['birth_country'].replace({
    'Moldavská republika' : 'Moldavsko',
    'Ruská federace' : 'Rusko',
    'Společ.nezávisl.států' : 'Rusko',
    'Česká Republika' : 'Česko',
    'Česká republika' : 'Česko',
})
maindf['distance'] = distance(maindf.wgslon, maindf.wgslat, maindf.label_wgslon, maindf.label_wgslat)
maindf['accurate'] = maindf.distance <= 10
nrows = len(maindf.index)
print(nrows, 'total rows')
maindf['geocoded'] = ~numpy.isnan(maindf.wgslon)
noblank = maindf[maindf.geocoded]
geocrows = len(noblank.index)
print(geocrows, 'geocoded', '{:.1%}'.format(geocrows / nrows), 'completeness')
accurrows = noblank.accurate.sum()
print(accurrows, 'geocoded accurately', '{:.1%}'.format(accurrows / geocrows), 'accuracy')

print(maindf.dtypes)
bycountry = maindf.groupby('country').agg({'geocoded' : numpy.sum, 'accurate' : numpy.sum, 'distance' : len})
bycountry['completeness'] = bycountry.geocoded / bycountry.distance
bycountry['accuracy'] = bycountry.accurate / bycountry.geocoded

print(bycountry)

if len(sys.argv) > 3:
    maindf.to_csv(sys.argv[3], sep=';', index=False)
