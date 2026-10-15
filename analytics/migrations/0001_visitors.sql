-- Only city aggregates and counters are retained permanently.
CREATE TABLE IF NOT EXISTS visitor_totals (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pageviews INTEGER NOT NULL DEFAULT 0,
  located_pageviews INTEGER NOT NULL DEFAULT 0,
  first_seen INTEGER,
  last_seen INTEGER
);
INSERT OR IGNORE INTO visitor_totals (id) VALUES (1);

CREATE TABLE IF NOT EXISTS visitor_cities (
  city_key TEXT PRIMARY KEY,
  country_code TEXT NOT NULL CHECK (length(country_code) = 2),
  city TEXT NOT NULL,
  region TEXT NOT NULL,
  latitude REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
  longitude REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
  pageviews INTEGER NOT NULL CHECK (pageviews > 0),
  first_seen INTEGER NOT NULL,
  last_seen INTEGER NOT NULL
);

-- Short-lived event IDs make retries idempotent, including concurrent retries.
CREATE TABLE IF NOT EXISTS visitor_events (
  event_id TEXT PRIMARY KEY,
  city_key TEXT,
  country_code TEXT,
  city TEXT,
  region TEXT,
  latitude REAL,
  longitude REAL,
  recorded_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS visitor_events_expiry ON visitor_events (expires_at);

-- One INSERT and its trigger form one SQLite transaction: a retry cannot
-- increment the total without its city, or increment the same event twice.
CREATE TRIGGER IF NOT EXISTS count_visitor_event
AFTER INSERT ON visitor_events
BEGIN
  UPDATE visitor_totals
  SET pageviews = pageviews + 1,
      located_pageviews = located_pageviews + (NEW.city_key IS NOT NULL),
      first_seen = COALESCE(first_seen, NEW.recorded_at),
      last_seen = MAX(COALESCE(last_seen, NEW.recorded_at), NEW.recorded_at)
  WHERE id = 1;

  INSERT INTO visitor_cities
    (city_key, country_code, city, region, latitude, longitude, pageviews, first_seen, last_seen)
  SELECT NEW.city_key, NEW.country_code, NEW.city, NEW.region,
         NEW.latitude, NEW.longitude, 1, NEW.recorded_at, NEW.recorded_at
  WHERE NEW.city_key IS NOT NULL
  ON CONFLICT (city_key) DO UPDATE SET
    pageviews = pageviews + 1,
    first_seen = MIN(first_seen, excluded.first_seen),
    last_seen = MAX(last_seen, excluded.last_seen);
END;

-- No raw IP is written to any table. The key is a daily rotating HMAC.
CREATE TABLE IF NOT EXISTS visitor_geo_cache (
  client_hash TEXT PRIMARY KEY,
  location_json TEXT,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS visitor_geo_cache_expiry ON visitor_geo_cache (expires_at);

CREATE TABLE IF NOT EXISTS visitor_rate_limits (
  client_hash TEXT PRIMARY KEY,
  window_start INTEGER NOT NULL,
  request_count INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS visitor_rate_limits_expiry ON visitor_rate_limits (expires_at);

-- Reservation is atomic across Worker instances and data centers.
CREATE TABLE IF NOT EXISTS visitor_ipinfo_budget (
  month TEXT PRIMARY KEY,
  attempts INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
