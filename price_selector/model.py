"""Frozen household-response adapter and SB3 DQN construction.

Usage from Python (supply training-only histories and their control timestamps):
    response = FrozenResponse('results/five_hour_response/model/gru.pt')
    env = PriceEnv(response, histories, starts, household_ids=response.household_ids)
    agent = create_dqn(env)
    agent.learn(total_timesteps=10_000)

No training or output writes happen on import. Histories must use the checkpoint's
household order. The simulator's forecast quality limits any learned policy.
"""

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from household_response.model import HouseholdGRU
from price_selector.environment import PriceEnv, DataPreprocessor


class FrozenResponse:
    """Load the existing GRU once; each call re-encodes its supplied history."""

    def __init__(self, checkpoint):
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.household_ids = list(bundle['household_ids'])
        self.means = bundle['household_means'].numpy()
        self.maxima = bundle['household_maxima'].numpy()
        self.preprocessor = DataPreprocessor(self.maxima, self.means, self.household_ids)
        self.model = HouseholdGRU(len(self.household_ids), bundle['width'])
        self.model.load_state_dict(bundle['weights'])
        self.model.eval().requires_grad_(False)

    def __call__(self, history_kwh, next_price):
        features = self.preprocessor.history_features(history_kwh)
        if features.ndim != 2:
            raise ValueError("Predict one (10,H) history at a time.")
        _, normalized_price = self.preprocessor.price(next_price)
        features = torch.from_numpy(features)
        price = torch.tensor([normalized_price], dtype=torch.float32)
        with torch.inference_mode():
            return self.model(features[None], price)[0].numpy() * self.maxima


def create_dqn(env: PriceEnv, *, seed=42, **kwargs):
    """Finite-horizon undiscounted DQN; overrides allow small smoke tests.

    gamma=1 makes the terminal reward exactly the five-term episode objective.
    A bounded replay buffer avoids SB3's million-transition default for the
    large household history observations (2,000 transitions ~430 MB at H=1117).
    """
    if 'gamma' in kwargs and kwargs['gamma'] != 1.0:
        raise ValueError("Use gamma=1 to preserve the undiscounted episode objective.")
    config = dict(learning_rate=1e-4, buffer_size=2_000, learning_starts=240,
                  batch_size=32, gamma=1.0, train_freq=4, gradient_steps=1,
                  target_update_interval=240, exploration_fraction=0.2,
                  exploration_final_eps=0.05, policy_kwargs={'net_arch': [128, 64]},
                  device='cpu', verbose=1)
    config.update(kwargs)
    return DQN('MultiInputPolicy', Monitor(env), seed=seed, **config)


def load_environments(data_folder, checkpoint):
    """Build midnight episodes; history plus 24h must stay in one saved split.

    Only train and validation partitions are loaded into environments. Actual
    future consumption is never supplied to the policy or simulator. Saved
    timestamps/partition labels determine episode eligibility, not future loads.
    """
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    folder = Path(data_folder)
    required = ['demand.npy', 'households.npy', 'times.npy', 'prices.npy',
                'partition_ids.npy', 'manifest.json']
    missing = [str(folder / name) for name in required if not (folder / name).is_file()]
    if not Path(checkpoint).is_file():
        missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError('Missing training inputs:\n' + '\n'.join(missing))
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest_bytes = (folder / 'manifest.json').read_bytes()
    if bundle.get('data_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('GRU checkpoint must match this prepared panel manifest.')
    manifest = json.loads(manifest_bytes)
    # Detect changed arrays as well as a changed manifest.
    for name in required[:-1]:
        expected = manifest.get('output_hashes', {}).get(name)
        digest = hashlib.sha256()
        with (folder / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if expected != digest.hexdigest():
            raise ValueError(f'Prepared data fingerprint mismatch: {name}')
    response = FrozenResponse(checkpoint)
    households = np.load(folder / 'households.npy', allow_pickle=False).tolist()
    if households != response.household_ids:
        raise ValueError('Prepared panel household order differs from the GRU checkpoint.')
    demand = np.load(folder / 'demand.npy', mmap_mode='r', allow_pickle=False)
    times = pd.DatetimeIndex(np.load(folder / 'times.npy', allow_pickle=False))
    partitions = np.load(folder / 'partition_ids.npy', allow_pickle=False)
    prices = np.load(folder / 'prices.npy', allow_pickle=False)
    if (demand.shape != (len(households), len(times))
            or partitions.shape != (len(times),) or prices.shape != (len(times),)):
        raise ValueError('Panel, prices, timestamps and partitions have inconsistent shapes.')
    if (times.hasnans or times.has_duplicates
            or not np.all(np.diff(times.asi8) == 30 * 60 * 1_000_000_000)):
        raise ValueError('Prepared timestamps must be consecutive half-hours.')
    if not np.isin(partitions, [0, 1, 2]).all() or not np.isin(prices, [0, 1, 2]).all():
        raise ValueError('Unknown saved partition or tariff ID.')
    environments, counts = {}, {}
    from price_selector.environment import PRICES
    for split, partition_id in [('train', 0), ('validation', 1)]:
        starts = [i for i in range(10, len(times) - 47)
                  if times[i].hour == 0 and times[i].minute == 0
                  and np.all(partitions[i - 10:i + 48] == partition_id)]
        if not starts:
            raise ValueError(f'No complete midnight-to-midnight episodes in {split} partition.')
        histories = np.stack([demand[:, i - 10:i].T for i in starts])
        environments[split] = PriceEnv(response, histories, times[starts],
                                       household_ids=households,
                                       previous_prices=PRICES[prices[np.array(starts) - 1]])
        counts[split] = len(starts)
    return environments, counts


def run_diagnostics(data_folder, checkpoint, output, *, seed=42):
    """Validation-only simulator/price/objective diagnostics; no policy training.

    Candidate search is exploratory on validation, not evidence of held-out
    performance or causal demand response. Only the three measured prices are used.
    """
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from price_selector.environment import PRICES, DEFAULT_WEIGHTS, score_episode
    from household_response.data import window_targets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    envs, counts = load_environments(data_folder, checkpoint)
    env = envs['validation']
    response = env.response
    rng = np.random.default_rng(seed)
    folder = Path(data_folder)
    demand = np.load(folder / 'demand.npy', mmap_mode='r')
    partitions = np.load(folder / 'partition_ids.npy')
    prices = np.load(folder / 'prices.npy')
    times = pd.DatetimeIndex(np.load(folder / 'times.npy'))
    checks = []
    def checked(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def dump(name, values):
        (output / name).write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')

    distribution = {}
    for split, pid in [('train', 0), ('validation', 1)]:
        selected = partitions == pid
        distribution[split] = {str(p): int(np.sum(prices[selected] == i))
                               for i, p in enumerate(PRICES)}
    print('Diagnostic 1: tariff coverage and one-step price sensitivity', flush=True)
    targets = window_targets(partitions, demand)['validation']
    # Predict three counterfactual prices on identical observed histories.
    predictions, truth, previous = [], [], []
    with torch.inference_mode():
        for selected in np.array_split(targets, max(1, int(np.ceil(len(targets) / 64)))):
            history = np.stack([demand[:, i-10:i].T for i in selected])
            features = torch.from_numpy(response.preprocessor.history_features(history))
            price_features = torch.tensor((PRICES - PRICES[0]) / np.ptp(PRICES), dtype=torch.float32)
            hidden = response.model.history_gru(features)[1][-1]
            heads = []
            for price in price_features:
                context = torch.cat((hidden, price.expand(len(selected), 1)), dim=1)
                heads.append(torch.nn.functional.softplus(response.model.next_demand(context)).numpy()
                             * response.maxima)
            predictions.append(np.stack(heads, axis=1))
            truth.append(np.asarray(demand[:, selected].T))
            previous.append(np.asarray(demand[:, selected-1].T))
    pred = np.concatenate(predictions)
    actual, past = np.concatenate(truth), np.concatenate(previous)
    known = np.isfinite(actual)
    paired = known & np.isfinite(past)
    factual = pred[np.arange(len(targets)), prices[targets]]
    def rmse(values, mask=known):
        return float(np.sqrt(np.mean(np.square(values[mask].astype(float) - actual[mask]))))
    shuffled_errors = []
    for _ in range(10):
        shuffled = pred[np.arange(len(targets)), rng.permutation(prices[targets])]
        shuffled_errors.append(rmse(shuffled))
    normal = pred[:, 1]
    total_normal = normal.sum(axis=1)
    coefficients = response.model.next_demand.weight[:, -1].detach().numpy()
    sensitivity = {
        'validation_windows': len(targets),
        'observed_target_readings': int(known.sum()),
        'factual_price_rmse_kwh': rmse(factual),
        'paired_factual_rmse_kwh': rmse(factual, paired),
        'paired_persistence_rmse_kwh': rmse(past, paired),
        'forced_normal_input_rmse_kwh': rmse(normal),
        'shuffled_price_rmse_mean_kwh': float(np.mean(shuffled_errors)),
        'shuffled_price_rmse_std_kwh': float(np.std(shuffled_errors)),
        'low_vs_normal_aggregate_change_pct': float(np.mean((pred[:, 0].sum(axis=1)/total_normal-1)*100)),
        'high_vs_normal_aggregate_change_pct': float(np.mean((pred[:, 2].sum(axis=1)/total_normal-1)*100)),
        'households_increasing_demand_with_price': int(np.sum(coefficients > 0)),
        'households_decreasing_demand_with_price': int(np.sum(coefficients < 0)),
        'households_price_insensitive': int(np.sum(coefficients == 0)),
        'architecture': 'Price enters only the last affine layer before softplus; immediate price-response direction is fixed per household for every history.',
        'interpretation': 'Forced and shuffled prices are inference perturbations, not retrained price-free baselines or causal estimates.'}
    checked('finite counterfactual one-step predictions', bool(np.isfinite(pred).all()))

    # Fixed candidate pool: constant tariffs; all 1/2/3/6-block Low/High events;
    # and 64 seeded random schedules. Search is only a finite feasibility probe.
    schedules = {}
    for name, tariff in [('normal',1),('low',0),('high',2)]:
        schedules[name] = np.full(12, tariff)
    for width in [1,2,3,6]:
        for start in range(13-width):
            for tariff in [0,2]:
                schedule = np.ones(12,dtype=int)
                schedule[start:start+width] = tariff
                schedules[f'event_{tariff}_{start}_{width}'] = schedule
    for trial in range(64):
        schedules[f'random_{trial:02}'] = rng.integers(0,3,12)
    # Drop exact duplicates to keep counts honest.
    unique, seen = {}, set()
    for name, schedule in schedules.items():
        key = tuple(schedule)
        if key not in seen:
            unique[name] = schedule
            seen.add(key)
    names = list(unique)
    schedule_array = np.stack(list(unique.values()))
    normal_index = names.index('normal')
    variant_weights = {
        'default': DEFAULT_WEIGHTS,
        'demand_only': dict(demand=1,energy=0,bill=0,switch=0,jump=0),
        'no_bill': dict(DEFAULT_WEIGHTS,bill=0),
        'no_energy': dict(DEFAULT_WEIGHTS,energy=0),
        'no_stability': dict(DEFAULT_WEIGHTS,switch=0,jump=0),
        'demand_weight_5': dict(DEFAULT_WEIGHTS,demand=5),
        'bill_weight_0.1': dict(DEFAULT_WEIGHTS,bill=.1),
    }
    terms = ['demand','energy','switch','jump','bill']
    per_day, term_arrays = [], []
    high_event_deltas, low_event_deltas = [], []
    numerical_differences = []
    model_root = Path(__file__).resolve().parents[1] / 'results/price_selector_dqn'
    agents = {name:DQN.load(model_root / path,device='cpu')
              for name,path in [('best','best/best_model.zip'),('final','final_model.zip')]
              if (model_root/path).is_file()}
    agent_records = {name:[] for name in agents}

    for day in range(counts['validation']):
        env.reset(options={'episode_index':day})
        reference = env.reference.copy()
        n = len(names)
        history = np.repeat(env.histories[day][None], n, axis=0)
        trajectories = []
        with torch.inference_mode():
            for t in range(48):
                features = torch.from_numpy(response.preprocessor.history_features(history))
                tariff = PRICES[schedule_array[:,t//4]]
                normalized = torch.tensor((tariff-PRICES[0])/np.ptp(PRICES),dtype=torch.float32)
                forecast = response.model(features,normalized).numpy()*response.maxima
                trajectories.append(forecast)
                history = np.concatenate((history[:,1:],forecast[:,None]),axis=1)
        demand_paths = np.stack(trajectories,axis=1).astype(np.float64)
        path_prices = PRICES[np.repeat(schedule_array,4,axis=1)]
        def bounded(x): return np.square(x/(1+x))
        target = reference.sum(axis=1).mean()
        energy0 = reference.sum(axis=0)
        bill0 = energy0*PRICES[1]
        jumps = np.diff(np.column_stack((np.full(n,env.initial_price),PRICES[schedule_array])),axis=1)
        parts = np.column_stack((
            bounded(np.maximum(np.abs(demand_paths.sum(axis=2)-target)/target-.1,0)).mean(axis=1),
            bounded(np.abs(demand_paths.sum(axis=1)-energy0)/energy0).mean(axis=1),
            (jumps != 0).mean(axis=1),
            np.square(jumps/np.ptp(PRICES)).mean(axis=1),
            bounded(np.maximum((demand_paths*path_prices[:,:,None]).sum(axis=1)/bill0-1,0)).mean(axis=1)))
        relative_rms = float(np.linalg.norm(demand_paths[normal_index]-reference)
                             / np.linalg.norm(reference))
        numerical_differences.append(relative_rms)
        checked(f'day {day}: batched Normal relative RMS difference below 0.01%',
                relative_rms < 1e-4)
        # Use the exact serial environment reference for all paired baselines.
        demand_paths[normal_index] = reference
        parts[normal_index, 0] = bounded(np.maximum(np.abs(reference.sum(axis=1)-target)/target-.1,0)).mean()
        parts[normal_index, 1] = 0
        parts[normal_index, 4] = 0
        # Independent public scorer checks the vectorized diagnostics.
        for index in [normal_index,names.index('high'),names.index('low')]:
            _, expected = score_episode(demand_paths[index],reference,path_prices[index],env.initial_price)
            checked(f'day {day}: vectorized score matches public scorer for {names[index]}',
                    bool(np.allclose(parts[index],[expected[k] for k in terms],rtol=1e-6,atol=1e-8)))
        term_arrays.append(parts)
        weights = np.array([DEFAULT_WEIGHTS[k] for k in terms])
        costs = parts@weights/weights.sum()
        best_index = int(costs.argmin())
        per_day.append(dict(date=str(env.starts[day].date()),normal_cost=float(costs[normal_index]),
                            best_candidate=names[best_index],best_cost=float(costs[best_index]),
                            best_schedule=schedule_array[best_index].tolist()))
        # A High/Low event in block 4 followed by Normal: price effect and later change.
        for tariff, destination in [(2,high_event_deltas),(0,low_event_deltas)]:
            index = names.index(f'event_{tariff}_4_1')
            difference = demand_paths[index].sum(axis=1)-reference.sum(axis=1)
            destination.append(dict(during_kwh=float(difference[16:20].sum()),
                                    following_5h_kwh=float(difference[20:30].sum()),
                                    rest_of_day_kwh=float(difference[20:].sum())))
        for name,agent in agents.items():
            obs,_=env.reset(options={'episode_index':day})
            actions=[]
            for _ in range(12):
                action=int(agent.predict(obs,deterministic=True)[0]);actions.append(action)
                obs,_,_,_,info=env.step(action)
            agent_records[name].append(dict(date=str(env.starts[day].date()),cost=info['cost'],
                                            metrics=info['metrics'],actions=actions))
        print(f'Diagnostic 2: validation day {day+1}/{counts["validation"]}; '
              f'{n} schedules; best={names[best_index]}',flush=True)

    values = np.stack(term_arrays)
    weight_results = {}
    for label,weights in variant_weights.items():
        weights = np.array([weights[k] for k in terms])
        costs = values@weights/weights.sum()
        best_by_day = costs.argmin(axis=1)
        fixed_index = int(costs.mean(axis=0).argmin())
        improvement = costs[:,normal_index]-costs.min(axis=1)
        weight_results[label] = dict(
            normal_mean_cost=float(costs[:,normal_index].mean()),
            best_shared_schedule=names[fixed_index],best_shared_mean_cost=float(costs[:,fixed_index].mean()),
            per_day_search_mean_cost=float(costs.min(axis=1).mean()),
            days_with_candidate_better_than_normal=int((improvement>1e-7).sum()),
            mean_absolute_cost_improvement=float(improvement.mean()),
            mean_relative_cost_improvement_pct=float(np.mean(improvement/np.maximum(costs[:,normal_index],1e-12))*100))
    baselines = {}
    weights = np.array([DEFAULT_WEIGHTS[k] for k in terms])
    costs = values@weights/weights.sum()
    for name in ['normal','low','high']:
        index=names.index(name)
        baselines[name]=dict(mean_cost=float(costs[:,index].mean()),
                             metrics=dict(zip(terms,values[:,index].mean(axis=0).tolist())))
    for name,records in agent_records.items():
        deltas=np.array([r['cost'] for r in records])-costs[:,normal_index]
        baselines[name]=dict(mean_cost=float(np.mean([r['cost'] for r in records])),
                             days_beating_normal=int((deltas < -1e-7).sum()),
                             action_counts=np.bincount([a for r in records for a in r['actions']],minlength=3).tolist())
    # Compare training-support tariff patterns against diagnostic event schedules.
    support=dict(training_tariff_transition_counts={f'{a}->{b}':int(np.sum((prices[:-1]==a)&(prices[1:]==b)&(partitions[:-1]==0)&(partitions[1:]==0)))
                                                     for a in range(3) for b in range(3)})
    checked('all objective terms finite and bounded',bool(np.isfinite(values).all() and (values>=0).all() and (values<=1).all()))
    result=dict(seed=seed,scope='validation only; no retraining; no held-out test-set evaluation',
                households=len(response.household_ids),validation_days=counts['validation'],
                candidate_schedules=n,tariff_counts=distribution,one_step=sensitivity,
                batch_vs_serial_max_relative_rms=max(numerical_differences),
                baselines=baselines,weight_ablation=weight_results,
                event_response=dict(high_event=high_event_deltas,low_event=low_event_deltas),
                support=support,checks_passed=checks,
                limitations=['Finite schedule search does not prove global optimality.',
                             'Per-day candidate selection uses simulator forecasts; it is an exploratory oracle, not a deployable policy.',
                             'Weight variants are evaluated on the same candidate pool without retraining DQN.',
                             'One-step forecast diagnostics do not validate 24-hour autoregressive accuracy or causal price response.'])
    verified_winners = []
    for day, row in enumerate(per_day):
        if row['best_cost'] >= row['normal_cost'] - 1e-7:
            continue
        env.reset(options={'episode_index': day})
        for action in row['best_schedule']:
            _, _, _, _, info = env.step(action)
        checked(f'Serial environment reproduces winning schedule on {row["date"]}',
                info['cost'] < row['normal_cost'] - 1e-7
                and bool(np.isclose(info['cost'],row['best_cost'],rtol=2e-4,atol=1e-7)))
        verified_winners.append(dict(date=row['date'],schedule=row['best_schedule'],
                                     normal_cost=row['normal_cost'],serial_cost=info['cost'],
                                     metrics=info['metrics']))
    result['serial_verified_winners'] = verified_winners
    dump('summary.json',result)
    dump('per_day_search.json',per_day)
    dump('dqn_validation.json',agent_records)
    dump('candidate_schedules.json',{k:v.tolist() for k,v in unique.items()})
    np.savez_compressed(output/'candidate_metrics.npz',terms=values,term_names=np.array(terms),
                        schedules=schedule_array,labels=np.array(names),dates=np.array([r['date'] for r in per_day]))
    lines=['# Price-response diagnostics','',f'Validation days: {counts["validation"]}; schedules/day: {n}; households: {len(response.household_ids)}.',
           '', '## One-step forecast and price sensitivity','',
           f'- Factual GRU paired RMSE: {sensitivity["paired_factual_rmse_kwh"]:.6f} kWh.',
           f'- Persistence paired RMSE: {sensitivity["paired_persistence_rmse_kwh"]:.6f} kWh.',
           f'- Shuffled-price RMSE: {sensitivity["shuffled_price_rmse_mean_kwh"]:.6f} kWh (10 permutations).',
           f'- Low vs Normal aggregate prediction change: {sensitivity["low_vs_normal_aggregate_change_pct"]:.3f}%.',
           f'- High vs Normal aggregate prediction change: {sensitivity["high_vs_normal_aggregate_change_pct"]:.3f}%.',
           f'- Households predicting increasing demand with price: {sensitivity["households_increasing_demand_with_price"]}/{len(response.household_ids)}.',
           '',sensitivity['architecture'],'','## Same-day policy comparisons','','| Policy | Mean cost (lower is better) |','|---|---:|']
    lines += [f'| {name} | {record["mean_cost"]:.7f} |' for name,record in baselines.items()]
    lines += ['','## Objective ablations','','| Objective | Normal cost | Best shared schedule cost | Per-day search cost | Days beating Normal |','|---|---:|---:|---:|---:|']
    lines += [f'| {name} | {r["normal_mean_cost"]:.7f} | {r["best_shared_mean_cost"]:.7f} | {r["per_day_search_mean_cost"]:.7f} | {r["days_with_candidate_better_than_normal"]} |' for name,r in weight_results.items()]
    lines += ['','## Interpretation limits','']+['- '+x for x in result['limitations']]
    lines += ['',f'All {len(verified_winners)} non-Normal search winners were independently verified through the serial Gymnasium environment.', '',f'Passed {len(checks)} numerical consistency checks. Detailed inputs, schedules, per-day results and metrics accompany this report.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({key:result[key] for key in ['one_step','baselines','weight_ablation']},indent=2),flush=True)
    print(f'Diagnostics saved to {output}',flush=True)


def main():
    """Train with `python -m price_selector.model`; outputs never overwrite a run."""
    import argparse
    import json
    import time
    from pathlib import Path

    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Train the 24-hour tariff DQN against the frozen GRU.')
    parser.add_argument('--data', type=Path, default=root / 'results/five_hour_response/data')
    parser.add_argument('--checkpoint', type=Path, default=root / 'results/five_hour_response/model/gru.pt')
    parser.add_argument('--output', type=Path, default=root / 'results/price_selector_dqn')
    parser.add_argument('--timesteps', type=int, default=100_000,
                        help='DQN decisions, each covering two simulated hours')
    parser.add_argument('--save-every', type=int, default=1200)
    parser.add_argument('--eval-episodes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--diagnostics', action='store_true', help='Run validation-only price/objective diagnostics instead of training')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and environment without training')
    args = parser.parse_args()
    if min(args.timesteps, args.save_every, args.eval_episodes, args.threads) < 1:
        parser.error('timesteps, save-every, eval-episodes and threads must be positive')
    torch.set_num_threads(args.threads)
    if args.diagnostics:
        if args.output == root / 'results/price_selector_dqn':
            args.output = root / 'results/price_selector_diagnostics'
        run_diagnostics(args.data, args.checkpoint, args.output, seed=args.seed)
        return
    environments, counts = load_environments(args.data, args.checkpoint)
    train_env, validation_env = environments['train'], environments['validation']
    check_env(train_env)
    print(json.dumps({'episode_counts': counts, 'hours_per_episode': 24,
                      'decisions_per_episode': 12, 'households': len(train_env.scale)}), flush=True)
    if args.check_only:
        print('Input fingerprints and environment checks passed.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(episode_counts=counts, objective='five bounded costs over 24h; no recovery',
                    response_model='frozen GRU; simulated outcomes', gamma=1.0)
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.monotonic()
            self.episodes = 0

        def _on_step(self):
            for info in self.locals['infos']:
                if 'metrics' in info:
                    self.episodes += 1
                    row = dict(timesteps=self.num_timesteps, episode=self.episodes,
                               cost=info['cost'], metrics=info['metrics'])
                    with (args.output / 'episodes.jsonl').open('a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    if self.episodes % 10 == 0:
                        elapsed = time.monotonic() - self.started
                        print(f"Step {self.num_timesteps:,}/{args.timesteps:,}; "
                              f"episode {self.episodes}; cost {info['cost']:.5f}; "
                              f"elapsed {elapsed / 60:.1f} min", flush=True)
            return True

    agent = create_dqn(train_env, seed=args.seed)
    validation_env.reset(seed=args.seed + 1)
    callbacks = [Progress(),
                 CheckpointCallback(save_freq=args.save_every, save_path=str(args.output / 'checkpoints'),
                                    name_prefix='dqn'),
                 EvalCallback(Monitor(validation_env), eval_freq=args.save_every,
                              n_eval_episodes=args.eval_episodes, deterministic=True,
                              best_model_save_path=str(args.output / 'best'),
                              log_path=str(args.output / 'validation'))]
    try:
        agent.learn(total_timesteps=args.timesteps, callback=callbacks, log_interval=100)
    except KeyboardInterrupt:
        agent.save(args.output / 'interrupted_model')
        print('Stopped; saved interrupted_model.zip.', flush=True)
        return
    agent.save(args.output / 'final_model')
    print(f'Training complete: {args.output / "final_model.zip"}', flush=True)


if __name__ == '__main__':
    main()
