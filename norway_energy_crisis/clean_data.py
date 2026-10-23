"""Validate the published Norwegian panel and attach regional ERA5-Land temperature."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
START, END = '2020-10-01', '2022-03-31'
REGIONS = {'NO1': ('Oslo', 59.9139, 10.7522), 'NO5': ('Bergen', 60.3913, 5.3221)}
STATIC = ['Q22', 'Q23', *[f'Q27_{i}' for i in range(1, 11)], 'Q28', 'Q29']
SOURCES = ['households.csv', 'survey.csv', 'answers.csv', 'demand.csv', 'prices.csv', 'demand_areas.csv']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamps(frame):
    require(frame[['Date', 'Hour']].notna().all().all(), 'Missing date/hour.')
    hours = pd.to_numeric(frame.Hour, errors='raise').to_numpy()
    require(np.isfinite(hours).all() and np.equal(hours, np.floor(hours)).all()
            and ((hours >= 1) & (hours <= 24)).all(), 'Hour must be an integer from 1 to 24.')
    dates = pd.DatetimeIndex(pd.to_datetime(frame.Date, errors='raise'))
    require(dates.tz is None and dates.equals(dates.normalize()), 'Expected local dates without time zones.')
    return dates + pd.to_timedelta(hours - 1, unit='h')


def normalize_weather(payload, index):
    """Match the publisher's artificial 24-hour local day; never fill other gaps."""
    hourly = payload['hourly']
    utc = pd.DatetimeIndex(pd.to_datetime(hourly['time'], utc=True))
    values = np.asarray(hourly['temperature_2m'], dtype=float)
    require(len(utc) == len(values) and utc.is_unique and np.isfinite(values).all(),
            'Weather must have unique UTC hours and finite temperatures.')
    require(np.all(np.diff(utc.asi8) == 3600 * 10**9), 'Weather UTC series has gaps.')
    local = utc.tz_convert('Europe/Oslo').tz_localize(None)
    series = pd.Series(values, index=local)
    repeated = series.index[series.index.duplicated(keep='first')]
    series = series[~series.index.duplicated(keep='first')].reindex(index)
    missing = series.index[series.isna()]
    real_local_hours = pd.date_range(index[0], index[-1], freq='h', tz='Europe/Oslo').tz_localize(None)
    spring_hours = index.difference(real_local_hours)
    require(missing.difference(spring_hours).empty, 'Weather has unresolved non-DST gaps.')
    for hour in missing:
        before, after = hour - pd.Timedelta(hours=1), hour + pd.Timedelta(hours=1)
        require(before in series.index and after in series.index
                and pd.notna(series[before]) and pd.notna(series[after]), 'Cannot interpolate DST hour.')
        series.loc[hour] = (series[before] + series[after]) / 2
    require(series.notna().all(), 'Unresolved weather gaps.')
    return series.to_numpy(dtype=np.float32), {
        'spring_hours_interpolated': missing.astype(str).tolist(),
        'autumn_second_hours_omitted': repeated[repeated.isin(index)].astype(str).tolist()}


def weather(area, index, cache):
    city, lat, lon = REGIONS[area]
    parameters = dict(latitude=lat, longitude=lon, start_date='2020-09-29', end_date='2022-04-02',
                      hourly='temperature_2m', models='era5_land', timezone='UTC')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{area}_era5_land.json'
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved['request'] == parameters, f'Weather cache parameters differ: {path}')
    else:
        url = 'https://archive-api.open-meteo.com/v1/archive?' + urlencode(parameters)
        print(f'Fetching ERA5-Land regional temperature: {city}', flush=True)
        with urlopen(url, timeout=180) as response:
            payload = json.load(response)
        # Validate before caching a response.
        normalize_weather(payload, index)
        saved = dict(request=parameters, response=payload)
        save_json(path, saved)
    values, adjustments = normalize_weather(saved['response'], index)
    return values, dict(city=city, request=parameters, cache_sha256=digest(path), **adjustments,
                       source='Open-Meteo Historical Weather API; Copernicus ERA5-Land',
                       documentation='https://open-meteo.com/en/docs/historical-weather-api',
                       attribution='Weather data by Open-Meteo.com; ERA5-Land by Copernicus/ECMWF (CC BY 4.0).',
                       limitation='City-centre regional reanalysis proxy, not household observations or an operational forecast.')


def validate_surveys(households, answers, survey):
    require(households.ID.is_unique and answers.ID.is_unique, 'Duplicate household/survey IDs.')
    require(households.ID.notna().all() and answers.ID.notna().all(), 'Missing household ID.')
    require(set(households.ID) == set(answers.ID), 'Household and survey IDs do not match.')
    require(households.Demand_data.isin(['Yes', 'No']).all(), 'Unknown Demand_data flag.')
    for column in answers.columns.drop('ID'):
        definition = survey[survey.Question_ID.eq(column)]
        require(len(definition) > 0, f'Unknown survey question: {column}')
        values = pd.to_numeric(answers[column], errors='raise').dropna()
        require(np.isfinite(values).all(), f'Nonfinite survey answer: {column}')
        options = definition.Option_number.dropna().to_numpy()
        if len(options):
            require(values.isin(options).all(), f'Unknown answer code: {column}')
    return {q: {str(int(row.Option_number)): str(row.Option_translation)
                for row in survey[survey.Question_ID.eq(q)].itertuples() if pd.notna(row.Option_number)}
            for q in STATIC}


def clean(raw=ROOT, output=ROOT / 'cleaned', cache=ROOT / 'weather_cache'):
    raw, output = Path(raw), Path(output)
    require(not output.exists(), f'Output already exists: {output}; choose a new directory.')
    for name in SOURCES:
        require((raw / name).is_file(), f'Missing source: {name}')
    households = pd.read_csv(raw / 'households.csv')
    survey = pd.read_csv(raw / 'survey.csv')
    answers = pd.read_csv(raw / 'answers.csv')
    categories = validate_surveys(households, answers, survey)
    metered = households[households.Demand_data.eq('Yes')].sort_values('ID').merge(
        answers[['ID', 'Q32', *STATIC]], on='ID', validate='one_to_one', how='left')
    ids = metered.ID.to_numpy(dtype=np.int64)
    require(metered.Price_area.isin(REGIONS).all(), 'Unexpected metered household price area.')
    require(all(REGIONS[row.Price_area][0] == row.Region for row in metered.itertuples()),
            'Metered household region and price area disagree.')
    index = pd.date_range(START, END + ' 23:00', freq='h')
    panel = np.full((len(ids), len(index)), np.nan, dtype=np.float32)
    seen = np.zeros(panel.size, dtype=bool)
    rows, zeros = 0, 0
    with pd.read_csv(raw / 'demand.csv', chunksize=1_000_000,
                     dtype={'ID': 'int64', 'Demand_kWh': 'float64'}) as chunks:
        for frame in chunks:
            time = timestamps(frame)
            hi = np.searchsorted(ids, frame.ID.to_numpy())
            require((hi < len(ids)).all() and np.array_equal(ids[hi], frame.ID.to_numpy()),
                    'Unknown household in demand.')
            ti = index.get_indexer(time)
            require((ti >= 0).all(), 'Demand timestamp outside the published household period.')
            values = frame.Demand_kWh.to_numpy()
            require(np.isfinite(values).all() and (values >= 0).all(), 'Missing/invalid consumption; no target imputation.')
            key = hi * len(index) + ti
            require(len(np.unique(key)) == len(key) and not seen[key].any(), 'Duplicate household/hour demand key.')
            seen[key] = True
            panel.ravel()[key] = values
            rows += len(frame)
            zeros += int((values == 0).sum())
    require(seen.all(), 'Incomplete household/hour grid.')
    prices = pd.read_csv(raw / 'prices.csv')
    pt = timestamps(prices)
    require(not prices.duplicated(['Price_area', 'Date', 'Hour']).any(), 'Duplicate regional price key.')
    require(prices.Price_area.isin(['NO1', 'NO2', 'NO3', 'NO4', 'NO5']).all(), 'Unknown price area.')
    require(np.isfinite(prices[['Price_NOK_MWh', 'Price_NOK_kWh']]).all().all(), 'Invalid price values.')
    require(np.all(np.abs(prices.Price_NOK_kWh - prices.Price_NOK_MWh / 1000) <= .00501),
            'Price unit columns disagree beyond published rounding.')
    price_grid = []
    areas = list(REGIONS)
    for area in areas:
        selected = prices.Price_area.eq(area)
        series = pd.Series(prices.loc[selected, 'Price_NOK_MWh'].to_numpy() / 1000, index=pt[selected]).reindex(index)
        require(series.notna().all(), f'Price join is incomplete for {area}.')
        price_grid.append(series.to_numpy())
    aggregate = pd.read_csv(raw / 'demand_areas.csv')
    timestamps(aggregate)
    require(not aggregate.duplicated(['Price_area', 'Date', 'Hour']).any(), 'Duplicate aggregate key.')
    require(aggregate.Price_area.isin(['NO1', 'NO2', 'NO3', 'NO4', 'NO5']).all(), 'Unknown aggregate area.')
    require(np.isfinite(aggregate[['Demand_MWh', 'Meter_count', 'Demand_kWh_avg']]).all().all()
            and (aggregate.Meter_count > 0).all() and (aggregate.Demand_MWh >= 0).all(), 'Invalid aggregate data.')
    temperatures, weather_metadata = [], {}
    for area in areas:
        values, metadata = weather(area, index, cache)
        temperatures.append(values)
        weather_metadata[area] = metadata
    cohort = metered.Q32.eq(1).to_numpy()
    arrays = dict(demand=panel, prices=np.asarray(price_grid, dtype=np.float64),
                  temperature=np.asarray(temperatures), times=index.asi8,
                  household_ids=ids, area_index=np.array([areas.index(x) for x in metered.Price_area], dtype=np.int8),
                  cohort_mask=cohort, static_codes=metered[STATIC].to_numpy(dtype=np.float32))
    output.mkdir(parents=True)
    for name, values in arrays.items():
        np.save(output / f'{name}.npy', values, allow_pickle=False)
    audit = dict(state='complete', rows=rows, households=len(ids), hours=len(index),
                 primary_households=int(cohort.sum()),
                 primary_by_area={area: int((cohort & metered.Price_area.eq(area).to_numpy()).sum()) for area in areas},
                 missing_demand=0, duplicate_demand_keys=0, negative_demand=0, genuine_zero_readings=zeros,
                 start=str(index[0]), end=str(index[-1]), areas=areas, static_questions=STATIC,
                 category_labels=categories, weather=weather_metadata,
                 units=dict(demand='kWh per one-hour interval', price='NOK/kWh = Price_NOK_MWh / 1000', temperature='Celsius'),
                 assumptions=['Q32=1 identifies hourly spot contracts; assume the later reported contract applied throughout.',
                              'Selected dwelling/heating/EV descriptors are assumed stable during the observation period.',
                              'This observational dataset does not establish causal effects of pricing.'],
                 source_files={name: dict(sha256=digest(raw / name), rows=(len(households) if name=='households.csv'
                     else len(survey) if name=='survey.csv' else len(answers) if name=='answers.csv'
                     else rows if name=='demand.csv' else len(prices) if name=='prices.csv' else len(aggregate))) for name in SOURCES})
    audit['output_hashes'] = {x.name: digest(x) for x in output.glob('*.npy')}
    save_json(output / 'manifest.json', audit)
    (output / 'REPORT.md').write_text(f'# Data cleaning\n\nValidated {rows:,} hourly readings for {len(ids):,} households. '
        f'No missing demand, duplicate demand keys or negative readings. Preserved {zeros:,} zeros. '
        f'The primary cohort has {cohort.sum()} hourly spot-contract households.\n\n'
        'Regional prices retain negative values and MWh-to-kWh precision. Temperature comes from ERA5-Land reanalysis; '
        'only the publisher-compatible daylight-saving adjustments were applied. See manifest.json for hashes, counts and assumptions.\n')
    print(json.dumps({key:audit[key] for key in ['rows','households','hours','primary_households','primary_by_area','genuine_zero_readings']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'cleaned')
    parser.add_argument('--weather-cache', type=Path, default=ROOT / 'weather_cache')
    args = parser.parse_args()
    clean(args.raw, args.output, args.weather_cache)
