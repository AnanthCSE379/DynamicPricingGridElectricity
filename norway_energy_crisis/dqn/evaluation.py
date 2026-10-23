"""Enumerated evaluation, price diagnostics, baselines and paired calendar-block intervals."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from norway_energy_crisis.clean_data import digest, require, save_json
from .environment import (TERM_NAMES, PriceEnv, ReferenceCache, append_history, fingerprint,
                          price_grid, score)


def read_json(path):
    return json.loads(Path(path).read_text())


def record(provider, episode, demand, prices, reference, scored):
    curves = [np.asarray(demand)[:, provider.area_index == a].sum(1).astype(float).tolist() for a in range(2)]
    reference_curves = [reference[:, provider.area_index == a].sum(1).astype(float).tolist() for a in range(2)]
    return dict(origin=episode.timestamp, **scored, prices=np.asarray(prices).tolist(),
                regional_kwh=curves, reference_regional_kwh=reference_curves)


def summarize(days, grid):
    costs = [d['cost'] for d in days]
    numeric_stats = [k for k, v in days[0]['stats'].items() if isinstance(v, (int, float))]
    prices = np.asarray([d['prices'] for d in days])
    return dict(mean_cost=float(np.mean(costs)), std_daily_cost=float(np.std(costs)), days_count=len(days),
                terms={k:float(np.mean([d['terms'][k] for d in days])) for k in TERM_NAMES},
                stats={k:float(np.mean([d['stats'][k] for d in days])) for k in numeric_stats},
                action_counts={f'{p:.2f}':int(np.sum(np.abs(prices-p)<1e-8)) for p in grid}, days=days)


def evaluate(env, policy):
    """Every eligible day exactly once; evaluation never samples days with replacement."""
    days = []
    for index, episode in enumerate(env.provider.episodes):
        obs, _ = env.reset(options={'episode_index':index})
        reward_sum = 0.
        for _ in range(24):
            if hasattr(policy, 'predict'):
                action = int(policy.predict(obs, deterministic=True)[0])
            else:
                action = int(policy(env, obs))
            obs, reward, done, truncated, info = env.step(action)
            reward_sum += reward
        require(done and not truncated and abs(reward_sum+100*info['cost'])<1e-5,
                'Episode reward does not equal -100 * cost.')
        scored = {k:info[k] for k in ('cost', 'terms', 'stats')}
        days.append(record(env.provider, episode, env.demands, env.prices, env.reference, scored))
    return summarize(days, env.grid)


def constants(provider, engine, cache_folder, output, chunk=4):
    """All 31 constant-price trajectories; reuse the common 21-price prefix for cap 2."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = price_grid(3.)
    refs = ReferenceCache(provider, engine, cache_folder)
    signature = fingerprint(dict(engine=engine.signature, provider=provider.manifest_hash, split=provider.split,
                                 source=digest(__file__), origins=[e.origin for e in provider.episodes]))
    rows = []
    median = provider.preprocessor.state['price_band_edges'][1]
    for index, episode in enumerate(provider.episodes):
        path = output / f'{episode.origin}.json'
        if path.is_file():
            row = read_json(path)
            require(row['signature'] == signature, 'Constant baseline cache changed; choose a new output.')
            rows.append(row['prices']); continue
        reference = refs.get(index)
        base = provider.features(episode)
        by_price = []
        for start in range(0, len(grid), chunk):
            prices = grid[start:start+chunk]
            history = np.repeat(base[None], len(prices), axis=0)
            demand = []
            for t in range(24):
                candidates = np.broadcast_to(prices[:, None], (len(prices), provider.households))
                pred = engine.predict(history, candidates, episode.future_calendar[t])
                demand.append(pred)
                history = append_history(history, pred, candidates, episode.future_calendar[t], provider.preprocessor)
            trajectories = np.asarray(demand).transpose(1, 0, 2)
            for j, price in enumerate(prices):
                schedule = np.full(24, price)
                scored = score(trajectories[j], schedule, reference, episode, provider.area_index, median)
                by_price.append(record(provider, episode, trajectories[j], schedule, reference, scored))
        save_json(path, dict(signature=signature, prices=by_price))
        rows.append(by_price)
        print(f'{engine.label} {provider.split}: constant-price days {index+1}/{len(provider.episodes)}', flush=True)
    return rows


def constant_baselines(rows, cap, validation_index):
    grid = price_grid(cap)
    n = len(grid)
    indices = dict(minimum=0, maximum=n-1, validation_fixed=int(validation_index))
    results = {name:summarize([r[i] for r in rows], grid) for name, i in indices.items()}
    results['daily_constant_oracle'] = summarize([min(r[:n], key=lambda x:x['cost']) for r in rows], grid)
    return results


def baseline_suite(provider, engine, cache_folder, output, cap, validation_index, rows=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = constants(provider, engine, cache_folder, output / 'constants')
    results = constant_baselines(rows, cap, validation_index)
    grid = price_grid(cap)
    median_index = int(np.argmin(abs(grid-provider.preprocessor.state['price_band_edges'][1])))
    results['median'] = summarize([r[median_index] for r in rows], grid)
    env = PriceEnv(provider, engine, cache_folder, cap)
    for name in ('projected_market', 'greedy'):
        path = output / f'{name}_cap{cap:.0f}.json'
        if path.is_file():
            results[name] = read_json(path); continue
        if name == 'greedy':
            policy = lambda current, obs:current.greedy_action()
        else:
            weights = np.bincount(provider.area_index, minlength=2)/provider.households
            def policy(current, obs):
                price = float(weights @ current.episode.future_prices[:, current.hour])
                return int(np.argmin(abs(current.grid-price)))
        results[name] = evaluate(env, policy)
        save_json(path, results[name])
    save_json(output / f'baselines_cap{cap:.0f}.json', results)
    return results


def bootstrap(candidate, reference, block=7, repetitions=5000, seed=42):
    """Paired non-circular moving calendar blocks; scenarios/seeds are not extra days."""
    differences = np.asarray(candidate, dtype=float)-np.asarray(reference, dtype=float)
    require(differences.ndim == 1 and len(differences)>0 and np.isfinite(differences).all(), 'Invalid paired days.')
    n = len(differences)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(n-block+1, size=(repetitions, int(np.ceil(n/block))))
    indices = (starts[:, :, None]+np.arange(block)).reshape(repetitions, -1)[:, :n]
    samples = differences[indices].mean(1)
    return dict(mean_cost_difference=float(differences.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                days=n, block_days=block, repetitions=repetitions)


def uncertainty(candidate, reference):
    return {str(b):bootstrap(candidate, reference, block=b) for b in (3, 7, 14)}


def price_support(data):
    grid = price_grid(3.)
    train = data.partitions == 0
    dates = pd.DatetimeIndex(pd.to_datetime(data.times))
    quarter = (dates.month.to_numpy()-1)//3+1
    result = dict(grid=grid.tolist(), neighborhood_half_width=.05, fitted_on='training regional hours only', regions={})
    for a, name in enumerate(data.state['areas']):
        periods = {}
        for label, mask in [('all', train)]+[(f'Q{q}', train & (quarter==q)) for q in range(1, 5)]:
            prices = np.asarray(data.prices[a, mask], dtype=np.float64)
            counts = np.sum(np.abs(prices[:, None]-grid[None])<=.0500001, axis=0)
            periods[label] = dict(total_hours=len(prices), count=counts.tolist(),
                                  percentage=(100*counts/max(len(prices), 1)).tolist())
        result['regions'][name] = periods
    return result


def add_support(result, table, area_index):
    grid = np.asarray(table['grid'])
    weights = np.bincount(area_index, minlength=2)/len(area_index)
    for day in result['days']:
        q = (pd.Timestamp(day['origin']).month-1)//3+1
        indices = np.abs(np.asarray(day['prices'])[:, None]-grid).argmin(1)
        fractions = []
        for a, area in enumerate(table['regions']):
            percentages = np.asarray(table['regions'][area][f'Q{q}']['percentage'])[indices]
            fractions.append(float(np.mean(percentages<1.)))
        day['stats']['below_1pct_seasonal_support_fraction'] = float(weights @ fractions)
    result['stats']['below_1pct_seasonal_support_fraction'] = float(np.mean([
        d['stats']['below_1pct_seasonal_support_fraction'] for d in result['days']]))
    return result


def plotting(output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output) / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def diagnostics(data, provider, engine, cache_folder, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'diagnostics.json'
    if path.is_file():
        return read_json(path)
    table = price_support(data)
    save_json(output / 'price_support.json', table)
    plt = plotting(output)
    grid = price_grid(3.)
    curves = []
    indices = sorted(set([0, len(provider.episodes)//2, len(provider.episodes)-1]))
    for index in indices:
        episode = provider.episodes[index]
        history = provider.features(episode)
        individual = []
        for model in engine.models:
            # Reuse candidate head code with each member without changing weights or histories.
            old = engine.models
            try:
                engine.models = [model]
                individual.append(engine.predict_candidates(history, grid, episode.future_calendar[0]))
            finally:
                engine.models = old
        mean = np.mean(individual, axis=0)
        regional = np.stack([mean[:, provider.area_index==a].sum(1) for a in range(2)])
        curves.append(dict(origin=episode.timestamp, regional_kwh=regional.tolist(),
                           household_p10=np.quantile(mean, .1, axis=1).tolist(),
                           household_p50=np.quantile(mean, .5, axis=1).tolist(),
                           household_p90=np.quantile(mean, .9, axis=1).tolist(),
                           high_vs_low_households_increasing_fraction=float(np.mean(mean[-1]>mean[0])),
                           member_total_absolute_disagreement_kwh=np.abs(individual[0]-individual[-1]).sum(1).tolist(),
                           member_regional_kwh=[[values[:, provider.area_index==a].sum(1).tolist()
                                                 for a in range(2)] for values in individual]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in curves:
        for a in range(2):
            ax[a].plot(grid, c['regional_kwh'][a], label=c['origin'][:10])
            ax[a].set(title=data.state['areas'][a], xlabel='Candidate price (NOK/kWh)', ylabel='Predicted next-hour kWh')
    ax[0].legend();fig.tight_layout();fig.savefig(output/'regional_price_response.png', dpi=150);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in curves:
        ax.plot(grid, c['household_p50'], label=c['origin'][:10])
        ax.fill_between(grid, c['household_p10'], c['household_p90'], alpha=.12)
    ax.set(xlabel='Candidate price (NOK/kWh)', ylabel='Household demand kWh; median and 10–90% band')
    ax.legend();fig.tight_layout();fig.savefig(output/'household_price_response.png', dpi=150);plt.close(fig)
    rows = constants(provider, engine, cache_folder, output/'validation_constants')
    costs = np.array([[r['cost'] for r in day] for day in rows])
    terms = {k:np.mean([[r['terms'][k] for r in day] for day in rows], axis=0).tolist() for k in TERM_NAMES}
    best_indices = {str(int(cap)):int(np.argmin(costs[:, :len(price_grid(cap))].mean(0))) for cap in (2., 3.)}
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in TERM_NAMES:
        ax.plot(grid, terms[k], label=k)
    ax.plot(grid, costs.mean(0), color='black', linestyle='--', label='weighted total')
    ax.set(xlabel='Constant price (NOK/kWh)', ylabel='Mean validation cost component')
    ax.legend();fig.tight_layout();fig.savefig(output/'constant_price_components.png', dpi=150);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ('energy_change_pct', 'peak_reduction_pct'),
                              ('Daily energy change (%)', 'Peak reduction (%)')):
        values = np.mean([[r['stats'][key] for r in day] for day in rows], axis=0)
        ax.plot(grid, values);ax.axhline(0, color='gray', linestyle='--')
        ax.set(xlabel='Constant price (NOK/kWh)', ylabel=label)
    fig.tight_layout();fig.savefig(output/'constant_recursive_outcomes.png', dpi=150);plt.close(fig)
    result = dict(curves=curves, constant_mean_costs=costs.mean(0).tolist(), constant_mean_terms=terms,
                  validation_fixed_indices=best_indices, grid=grid.tolist(),
                  warnings=['Counterfactual price curves are observational-model predictions, not causal estimates.',
                            'Rare prices remain available; support flags do not mask actions.',
                            'Weights and demand tolerance were not tuned to force switching or high-price use.'])
    save_json(path, result)
    return result


def final_report(output, selection, evaluated, baselines):
    output = Path(output)
    scenarios = list(evaluated)
    comparisons = {}
    wider_checks, switching_checks = [], []
    for scenario in scenarios:
        by_cap = evaluated[scenario]
        costs = {cap:np.mean([[d['cost'] for d in r['days']] for r in runs], axis=0)
                 for cap, runs in by_cap.items()}
        wider = uncertainty(costs['3'], costs['2'])
        switch = {}
        for cap in ('2', '3'):
            oracle = np.array([d['cost'] for d in baselines[scenario][cap]['daily_constant_oracle']['days']])
            switch[cap] = uncertainty(costs[cap], oracle)
        comparisons[scenario] = dict(wider_cap=wider, switching_vs_daily_constant=switch)
        if scenario in ('ensemble', 'reserved'):
            # Ignore differences at round-off scale from batch-size/kernel changes.
            wider_checks.append(wider['7']['ci95'][1]<-1e-6)
            chosen_cap = str(int(selection['cap']))
            actually_switches = np.mean([r['stats']['internal_switches'] for r in by_cap[chosen_cap]])>0
            switching_checks.append(switch[chosen_cap]['7']['ci95'][1]<-1e-6 and actually_switches)
    result = dict(selection=selection, paired_block_uncertainty=comparisons, numerical_cost_tolerance=1e-6,
                  robust_wider_range_benefit=all(wider_checks),
                  switching_benefit_survives_reserved=all(switching_checks),
                  seed_variability={scenario:{cap:dict(mean_cost=float(np.mean([r['mean_cost'] for r in runs])),
                                  std_seed_cost=float(np.std([r['mean_cost'] for r in runs])))
                                  for cap, runs in evaluated[scenario].items()} for scenario in scenarios},
                  limitations=['Optimization inside learned response models is not causal demand-response validation.',
                    'The price-free control separates physical predictions from charge accounting; it is not causal identification.',
                    'Prices model spot-energy charges, excluding complete retail bills and supplier profitability.',
                    'Contract and selected static survey descriptors are assumed stable retrospectively.',
                    'Regional ERA5-Land temperature is reanalysis, persisted within each episode.',
                    'Test months previously assessed response forecasting; DQN selection used validation only.',
                    'Independent 24-hour episodes do not measure later rebound or continuous-day operation.',
                    'This is equal wall-clock compute, not equal numbers of transitions.'])
    save_json(output/'comparison.json', result)
    lines = ['# Norwegian DQN comparison', '', f'Selected cap: {selection["cap"]:.2f} NOK/kWh; deployment seed: {selection["deployment_seed"]}.',
             '', '| Response scenario | Cap | Mean test cost | Seed SD | Switches/day |', '|---|---:|---:|---:|---:|']
    for scenario in scenarios:
        for cap, runs in evaluated[scenario].items():
            stats = result['seed_variability'][scenario][cap]
            switches = np.mean([r['stats']['internal_switches'] for r in runs])
            lines.append(f'| {scenario} | {cap} | {stats["mean_cost"]:.7f} | {stats["std_seed_cost"]:.7f} | {switches:.2f} |')
    lines += ['', f'Robust wider-range benefit: **{result["robust_wider_range_benefit"]}**.',
              f'Switching benefit against the daily constant oracle, surviving reserved dynamics: **{result["switching_benefit_survives_reserved"]}**.',
              '', 'Intervals use paired moving calendar blocks (7 days primary; 3 and 14 sensitivity), 5,000 resamples. Scenario models are not extra independent observations.',
              '', 'See test_metrics.json for all baselines, four cost components, physical outcomes, actions, sparse-price use and daily trajectories.',
              '', '## Limitations', ''] + ['- '+x for x in result['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plt = plotting(output)
    fig, ax = plt.subplots(figsize=(10, 4))
    width = .032
    for i, cap in enumerate(('2', '3')):
        counts = np.mean([[r['action_counts'].get(f'{p:.2f}', 0) for p in price_grid(3.)]
                          for r in evaluated['ensemble'][cap]], axis=0)
        ax.bar(price_grid(3.)+(i-.5)*width, counts/counts.sum()*100, width=width, label=f'{cap}.00 cap')
    ax.set(xlabel='Price (NOK/kWh)', ylabel='Percentage of test hours');ax.legend()
    fig.tight_layout();fig.savefig(output/'price_frequencies.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for cap in ('2', '3'):
        run = evaluated['ensemble'][cap][0]
        day = run['days'][len(run['days'])//2]
        axes[0].step(np.arange(24), day['prices'], where='post', label=f'{cap}.00 cap')
        axes[1].plot(np.sum(day['regional_kwh'], axis=0), label=f'{cap}.00 cap')
    axes[1].plot(np.sum(day['reference_regional_kwh'], axis=0), '--', label='Recorded-price reference')
    axes[0].set_ylabel('NOK/kWh');axes[1].set(xlabel='Hour', ylabel='Aggregate kWh')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(output/'example_day.png', dpi=160);plt.close(fig)
    return result
