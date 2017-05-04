CREATE TABLE IF NOT EXISTS geonames(
  name VARCHAR(200),
  id INTEGER
);

CREATE TABLE IF NOT EXISTS geolocations(
  id INTEGER,
  wgslat REAL,
  wgslon REAL,
  loccat CHAR(1),
  loctype VARCHAR(10),
  country VARCHAR(5),
  population INTEGER
);