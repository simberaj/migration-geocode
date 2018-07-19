CREATE INDEX IF NOT EXISTS name_idx ON geonames(name);
CREATE INDEX IF NOT EXISTS locid_idx ON geolocations(id);
CLUSTER geonames USING name_idx;
CLUSTER geolocations USING locid_idx;

-- takes 44 mins;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX trgm_name_idx ON geonames USING GIST (name gist_trgm_ops);

-- CREATE OR REPLACE FUNCTION geonames_importance(category char, population integer) RETURNS integer AS $$
    -- SELECT CASE
        -- WHEN category = 'P' THEN population + 5
        -- WHEN category = 'A' THEN 100
        -- WHEN category = 'S' THEN 4
        -- WHEN category = 'R' THEN 3
        -- WHEN category = 'L' THEN 2
        -- WHEN category = 'H' THEN 1
    -- ELSE 0 END
-- $$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

CREATE OR REPLACE FUNCTION geonames_importance_log(
        category char,
        population bigint
    ) RETURNS double precision AS $$
    SELECT log(CASE
            WHEN category = 'P' THEN population + 6
            WHEN category = 'A' THEN 100
            WHEN category = 'S' THEN 5
            WHEN category = 'R' THEN 4
            WHEN category = 'L' THEN 3
            WHEN category = 'H' THEN 2
        ELSE 1 END
    ) / 100
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

SET pg_trgm.similarity_threshold = 0.6;

