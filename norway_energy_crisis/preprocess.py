"""Training-only preprocessing shared by Norwegian response training and inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json

CALENDAR_NAMES = ['hour_noon_peak', 'hour_sine_phase', 'weekday_sin', 'weekday_cos', 'year_sin', 'year_cos']
HISTORY_NAMES = ['demand_scaled', 'price_standardized', 'temperature_standardized', *CALENDAR_NAMES]
SPLITS = {'train': ('2020-10-01', '2022-01-01'),
          'validation': ('2022-01-01', '2022-02-01'), 'test': ('2022-02-01', '2022-04-01')}


def calendar(values):
    values = np.asarray(values)
    shape = values.shape
    time = pd.DatetimeIndex(pd.to_datetime(values.ravel()))
    require(not time.hasnans, 'Invalid calendar timestamp.')
    hour = 2 * np.pi * time.hour.to_numpy() / 24
    week = 2 * np.pi * time.dayofweek.to_numpy() / 7
    year = 2 * np.pi * (time.dayofyear.to_numpy() - 1 + time.hour.to_numpy() / 24) / np.where(time.is_leap_year, 366, 365)
    return np.column_stack(((1-np.cos(hour))/2, (1+np.sin(hour))/2,
                            np.sin(week), np.cos(week), np.sin(year), np.cos(year))).astype(np.float32).reshape(*shape, 6)


def eligible_origins(partitions, split_id, daily=False):
    """Use 24 past hours plus all 24 targets entirely inside the split."""
    result = [i for i in range(24, len(partitions)-23)
              if (not daily or i % 24 == 0) and np.all(partitions[i-24:i+24] == split_id)]
    return np.asarray(result, dtype=np.int64)


def encode_static(codes, area_index, categories=None):
    codes = np.column_stack((area_index, codes))
    if categories is None:
        categories = [sorted(float(x) for x in np.unique(codes[:, j]) if np.isfinite(x))
                      for j in range(codes.shape[1])]
    blocks = []
    for j, values in enumerate(categories):
        x = codes[:, j]
        block = np.column_stack([x == value for value in values] + [~np.isin(x, values)])
        blocks.append(block.astype(np.float32))
    return np.concatenate(blocks, axis=1), categories


class Preprocessor:
    """Immutable fitted statistics; reusable outside the training dataset."""

    def __init__(self, state):
        self.state = state
        self.scales = np.asarray(state['demand_scales'], dtype=np.float32)
        self.ids = list(state['household_ids'])
        self.static = np.asarray(state['static_encoded'], dtype=np.float32)

    def history_features(self, demand, prices, temperature, times, household_indices):
        demand, prices, temperature = [np.asarray(x, dtype=np.float32) for x in (demand, prices, temperature)]
        require(demand.ndim == 2 and prices.shape == demand.shape and temperature.shape == demand.shape,
                'Demand, price and temperature must have matching (batch,history) shapes.')
        indices = np.asarray(household_indices, dtype=int)
        require(indices.shape == (len(demand),) and ((indices >= 0) & (indices < len(self.ids))).all(), 'Invalid household indices.')
        require(np.isfinite(demand).all() and (demand >= 0).all()
                and np.isfinite(prices).all() and np.isfinite(temperature).all(), 'Invalid history values.')
        cyclical = calendar(times)
        if cyclical.ndim == 2:
            cyclical = np.broadcast_to(cyclical, (*demand.shape, 6))
        require(cyclical.shape == (*demand.shape, 6), 'Timestamp/history shape mismatch.')
        return np.concatenate(((demand/self.scales[indices, None])[..., None],
                  ((prices-self.state['price_mean'])/self.state['price_std'])[..., None],
                  ((temperature-self.state['temperature_mean'])/self.state['temperature_std'])[..., None],
                  cyclical), axis=-1).astype(np.float32)

    def price(self, prices):
        prices = np.asarray(prices, dtype=np.float32)
        require(np.isfinite(prices).all(), 'Candidate price must be finite; negative prices are allowed.')
        return (prices-self.state['price_mean'])/self.state['price_std']


class PreparedData:
    def __init__(self, folder, verify=True):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        if verify:
            for name, expected in self.manifest['output_hashes'].items():
                require(digest(self.folder/name) == expected, f'Prepared file changed: {name}')
        self.state = json.loads((self.folder/'preprocessing.json').read_text())
        self.preprocessor = Preprocessor(self.state)
        for name in ['demand','prices','temperature','times','area_index','static','partitions']:
            setattr(self,name,np.load(self.folder/f'{name}.npy',mmap_mode='r',allow_pickle=False))
        self.origins = {name:np.load(self.folder/f'{name}_origins.npy',allow_pickle=False) for name in SPLITS}
        self.households = len(self.preprocessor.ids)

    def batch(self, households, origins, lookback, horizon=24):
        require(lookback in (5,24) and 1 <= horizon <= 24, 'Use 5h/24h history and at most 24 forecast hours.')
        households, origins = np.asarray(households,dtype=int),np.asarray(origins,dtype=int)
        require(households.shape == origins.shape and households.ndim == 1, 'Batch indices must match.')
        offsets = origins[:,None]+np.arange(-lookback,0)
        future = origins[:,None]+np.arange(horizon)
        areas = self.area_index[households]
        history = self.preprocessor.history_features(self.demand[households[:,None],offsets],
                    self.prices[areas[:,None],offsets],self.temperature[areas[:,None],offsets],
                    self.times[offsets],households)
        # The original previous day is also available for the seasonal baseline.
        yesterday = self.demand[households[:,None],origins[:,None]+np.arange(-24,0)]
        result = dict(history=history, static=np.asarray(self.static[households]),
                      prices=self.preprocessor.price(self.prices[areas[:,None],future]),
                      calendar=calendar(self.times[future]),
                      target=np.asarray(self.demand[households[:,None],future]),
                      scales=self.preprocessor.scales[households], yesterday=np.asarray(yesterday),
                      household_indices=households, area_index=areas,
                      raw_prices=np.asarray(self.prices[areas[:,None],future]))
        return {key:torch.from_numpy(np.array(value,copy=True)) for key,value in result.items()}


def preprocess(cleaned=ROOT/'cleaned', output=ROOT/'prepared'):
    cleaned,output = Path(cleaned),Path(output)
    require(not output.exists(), f'Output exists: {output}; choose a new directory.')
    manifest = json.loads((cleaned/'manifest.json').read_text())
    for name,expected in manifest['output_hashes'].items():
        require(digest(cleaned/name)==expected,f'Cleaned file changed: {name}')
    mask = np.load(cleaned/'cohort_mask.npy')
    ids = np.load(cleaned/'household_ids.npy')[mask]
    demand = np.load(cleaned/'demand.npy',mmap_mode='r')[mask]
    area_index = np.load(cleaned/'area_index.npy')[mask]
    prices = np.load(cleaned/'prices.npy')
    temperature = np.load(cleaned/'temperature.npy')
    times = np.load(cleaned/'times.npy')
    dates = pd.DatetimeIndex(times)
    labels = np.full(len(times),255,dtype=np.uint8)
    for i,(_, (lo,hi)) in enumerate(SPLITS.items()):
        labels[(dates>=lo)&(dates<hi)] = i
    require((labels != 255).all(), 'Unexpected time outside configured splits.')
    training = labels == 0
    require(np.isfinite(demand).all() and np.isfinite(temperature).all(), 'Missing data blocks preprocessing.')
    scales = np.quantile(demand[:,training],.95,axis=1).astype(np.float32)
    scales = np.where(scales>0,scales,1.).astype(np.float32)
    static,categories = encode_static(np.load(cleaned/'static_codes.npy')[mask],area_index)
    state = dict(version=1,household_ids=ids.tolist(), demand_scales=scales.tolist(),
                 price_mean=float(prices[:,training].mean()),price_std=float(prices[:,training].std()),
                 temperature_mean=float(temperature[:,training].mean()),temperature_std=float(temperature[:,training].std()),
                 demand_scaling='household training-only 95th percentile; no clipping',
                 static_categories=categories,static_encoded=static.tolist(),
                 static_questions=['price_area',*manifest['static_questions']],
                 static_labels=manifest['category_labels'],history_features=HISTORY_NAMES,calendar_features=CALENDAR_NAMES,
                 areas=manifest['areas'],assumptions=manifest['assumptions'],
                 forecast_temperature='Persist last known temperature at each forecast origin; never use future measured temperature.')
    require(state['price_std']>0 and state['temperature_std']>0,'Training price/temperature has zero variance.')
    # Bands describe training price support; they never constrain actions or clip prices.
    state['price_band_edges'] = np.quantile(prices[:,training],[.25,.5,.75]).tolist()
    arrays = dict(demand=demand,prices=prices.astype(np.float32),temperature=temperature,
                  times=times,area_index=area_index,static=static,partitions=labels)
    origins = {name:eligible_origins(labels,i,daily=(name!='train')) for i,name in enumerate(SPLITS)}
    require(all(len(x)>0 for x in origins.values()),'A split has no usable 24h origins.')
    output.mkdir(parents=True)
    for name,value in arrays.items():np.save(output/f'{name}.npy',value,allow_pickle=False)
    for name,value in origins.items():np.save(output/f'{name}_origins.npy',value,allow_pickle=False)
    save_json(output/'preprocessing.json',state)
    audit = dict(state='complete',households=len(ids),split_dates=SPLITS,
                 origins={name:len(value) for name,value in origins.items()},
                 split_rule='All 24 history hours and all 24 target hours inside one split; common origins for 5h/24h models.',
                 cleaned_manifest_sha256=digest(cleaned/'manifest.json'))
    audit['output_hashes'] = {p.name:digest(p) for p in output.iterdir() if p.suffix in ('.npy','.json')}
    save_json(output/'manifest.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleaned',type=Path,default=ROOT/'cleaned')
    parser.add_argument('--output',type=Path,default=ROOT/'prepared')
    args=parser.parse_args()
    preprocess(args.cleaned,args.output)
