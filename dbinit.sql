DROP TABLE IF EXISTS geonames;
DROP TABLE IF EXISTS geolocations;

CREATE TABLE geonames(
  name VARCHAR(200),
  id BIGINT
);

CREATE TABLE geolocations(
  id BIGINT PRIMARY KEY,
  wgslat REAL,
  wgslon REAL,
  loccat CHAR(1),
  loctype VARCHAR(10),
  country VARCHAR(5),
  population BIGINT
);