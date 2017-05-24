import sys
import pandas as pd

print('locs')
inlocs = pd.read_csv(sys.argv[1], sep=';', encoding='utf8')
locs = inlocs.iloc[:,[0,1,3,4]]
locs.columns = ['country', 'place', 'wgslat', 'wgslon']
del inlocs

print('counts')
incounts = pd.read_csv(sys.argv[2], sep=';', encoding='cp1250')
counts = incounts[['stp_narození', 'mistonar']]
counts.columns = ['country', 'place']
del incounts

print(locs.shape, counts.shape)
print('joining')
joined = pd.merge(counts, locs, how='inner', sort=False)
print(joined)

def aggregatePlaces(df):
  return pd.Series({
    'wgslat' : set(df.wgslat).pop(),
    'wgslon' : set(df.wgslon).pop(),
    'count' : len(df)
  })

joined.wgslat.fillna(0)

print(joined.shape)
print(joined[joined.wgslat>0].shape)
  
# print('summarizing')
# summarized = joined.groupby(['country', 'place']).apply(aggregatePlaces)
# print(summarized)

# summarized.to_csv(sys.argv[3], sep=';')