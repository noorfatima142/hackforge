# Dataset declaration

Team: [your team name]. Track: Air Quality (Pakistan).
Note: this was written late (Tue 20 Oct night / Wed 21 Oct), after the Mon 19 Oct 12:00 PM deadline.
The commit timestamp shows the real time.

## Source 1: OpenAQ
- Publisher: OpenAQ (aggregates government, embassy and community monitors)
- URL: https://openaq.org (API v3: https://api.openaq.org/v3)
- Licence: depends on each data provider; recorded per file in manifest.csv
- Update frequency: hourly
- Content: hourly PM2.5 for Lahore, Karachi, Islamabad, Peshawar

## Source 2: Open-Meteo
- Publisher: Open-Meteo (built on ERA5 reanalysis, Copernicus/ECMWF)
- URL: https://open-meteo.com
- Licence: CC BY 4.0 (attribution required)
- Update frequency: hourly
- Content: temperature, humidity, wind, pressure, precipitation per city

## Non-track source
Not added. We did not have time to integrate a third source, so the requirement for
one credible source outside the track list is NOT met. Candidate: NASA FIRMS fire data.

## Expected volume
About 65,800 gold city-hour rows (and about 267,000 cleaned sensor-hour rows),
Oct 2024 to Oct 2026, from many sensors per city.

## Why this data fits
Primary user: a provincial or district environment officer deciding where and when to
issue smog advisories. Hourly PM2.5 plus weather supports next-day forecasts per city.

## Baseline declared
Persistence (tomorrow = today). See BASELINE_DECLARATION.md.