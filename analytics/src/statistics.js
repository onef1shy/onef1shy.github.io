export async function readStatistics(db, now, onlineCities = new Set()) {
  const [totalsResult, citiesResult] = await db.batch([
    db.prepare('SELECT pageviews, located_pageviews, first_seen, last_seen FROM visitor_totals WHERE id = 1'),
    db.prepare('SELECT city_key, country_code, city, region, latitude, longitude, pageviews, first_seen, last_seen FROM visitor_cities ORDER BY country_code, region, city'),
  ]);
  const total = totalsResult.results[0];
  const cities = citiesResult.results.map(row => ({
    countryCode: row.country_code, city: row.city, region: row.region,
    latitude: row.latitude, longitude: row.longitude, pageviews: row.pageviews,
    firstSeen: row.first_seen, lastSeen: row.last_seen, online: onlineCities.has(row.city_key),
  }));
  return {
    version: 1, totalPageviews: total.pageviews, locatedPageviews: total.located_pageviews,
    countries: new Set(cities.map(city => city.countryCode)).size,
    cityCount: cities.length, firstSeen: total.first_seen, lastSeen: total.last_seen,
    generatedAt: now, cities,
  };
}
