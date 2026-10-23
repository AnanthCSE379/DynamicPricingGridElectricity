"""Time-budgeted Norwegian DQN training, resume, diagnostics and sealed final evaluation."""

import argparse
import io
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData
from .environment import EpisodeProvider, FrozenResponse, PriceEnv, price_grid
from .evaluation import (add_support, baseline_suite, constants, diagnostics, evaluate,
                         final_report, plotting, read_json)

SEEDS = (42, 7, 2026)
RUN_ORDER = [(cap, seed) for seed in SEEDS for cap in (2., 3.)]
DEFAULT_SETTINGS = dict(learning_rate=1e-4, buffer_size=3000, learning_starts=2400, batch_size=64,
                        train_freq=4, gradient_steps=1, target_update_interval=1000,
                        gamma=1., max_grad_norm=10., policy_kwargs={'net_arch':[256, 128]}, verbose=0)


class SlotClock:
    def __init__(self, seconds, consumed=0.):
        self.seconds = float(seconds)
        self.consumed = float(consumed)
        self.started = time.monotonic()

    @property
    def elapsed(self):
        return self.consumed + time.monotonic()-self.started

    def learning_rate(self, _=None):
        fraction = min(max(self.elapsed/self.seconds, 0.), 1.)
        return 1e-5 + 9e-5*(1-fraction)

    def epsilon(self, _=None):
        fraction = min(max(self.elapsed/(.3*self.seconds), 0.), 1.)
        return 1. - .95*fraction


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in
            ('__init__.py', 'environment.py', 'evaluation.py', 'train.py')}


def checkpoint_paths(response_root, name):
    mapping = dict(ensemble=[(42, 'price'), (7, 'price')], s42=[(42, 'price')], s7=[(7, 'price')],
                   reserved=[(2026, 'price')], control=[(42, 'no_price'), (7, 'no_price')])
    return [Path(response_root)/f'transformer_24h_seed{seed}_{kind}'/'best.pt' for seed, kind in mapping[name]]


def make_engine(response_root, name, data, device, selection_file=None):
    return FrozenResponse(checkpoint_paths(response_root, name), data, device, name, selection_file)


def make_agent(env, device, seed, clock, overrides=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)
    settings = {**DEFAULT_SETTINGS, **(overrides or {})}
    model = DQN('MultiInputPolicy', env, device=device, seed=seed, **settings)
    # SB3 updates these schedules internally on each collect/update; attach the clock
    # schedules themselves rather than setting exploration_rate in an on_step callback.
    model.lr_schedule = clock.learning_rate
    model.exploration_schedule = clock.epsilon
    model.exploration_rate = clock.epsilon()
    return model


def atomic_model(model, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        model.save(handle)
    temporary.replace(path)


def rng_snapshot(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                mps=torch.mps.get_rng_state() if device == 'mps' else None)


def restore_rng(state, device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if device == 'mps':
        torch.mps.set_rng_state(state['mps'])


def save_resume(path, model, env, clock, state, device):
    """One atomic local snapshot; replay/environment are not included by SB3.save()."""
    buffer = io.BytesIO();model.save(buffer)
    payload = dict(state=state, model=buffer.getvalue(), replay=model.replay_buffer,
                   environment=env.snapshot(), rng=rng_snapshot(device))
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Record consumed time after serializing the large replay buffer.
        pickle.dump(dict(elapsed=clock.elapsed), handle)
    temporary.replace(path)


def load_resume(path, env, seconds, signature, device):
    # These are trusted local training files, never downloaded pickle artifacts.
    with Path(path).open('rb') as handle:
        saved = pickle.load(handle)
        timing = pickle.load(handle)
    require(saved['state']['signature'] == signature, 'Resume inputs/settings changed; choose a new output.')
    clock = SlotClock(seconds, timing['elapsed'])
    model = DQN.load(io.BytesIO(saved['model']), env=env, device=device, force_reset=False)
    model.replay_buffer = saved['replay']
    model.replay_buffer.device = torch.device(device)
    # Initialize SB3's fresh Monitor/VecEnv wrappers, then restore the raw
    # environment and its RNG. Otherwise Monitor refuses the first resumed step.
    model.env.reset()
    env.restore(saved['environment'])
    model.lr_schedule = clock.learning_rate;model.exploration_schedule = clock.epsilon
    model.exploration_rate = clock.epsilon()
    restore_rng(saved['rng'], device)
    return model, clock, saved['state']


def fit(train_provider, val_provider, engine, output, cap, seed, seconds, cache_folder, device,
        validation_seconds=600., checkpoint_seconds=300., overrides=None):
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    signature = dict(version=1, sources=source_hashes(), data=train_provider.manifest_hash,
                     engine=engine.signature, cap=cap, seed=seed, seconds=seconds, device=device,
                     settings={**DEFAULT_SETTINGS, **(overrides or {})}, grid=price_grid(cap).tolist(),
                     objective='four_terms_no_switch_reward_scale100_v1',
                     validation_seconds=validation_seconds, checkpoint_seconds=checkpoint_seconds)
    config_path = output/'config.json'
    if config_path.exists():
        require(read_json(config_path) == signature, 'Fit configuration changed; choose a new output.')
    else:
        save_json(config_path, signature)
    resume_path = output/'resume.pkl'
    if (output/'result.json').is_file():
        resume_path.unlink(missing_ok=True)
        return read_json(output/'result.json')
    env = PriceEnv(train_provider, engine, cache_folder, cap)
    val_env = PriceEnv(val_provider, engine, cache_folder, cap)
    if resume_path.is_file():
        model, clock, state = load_resume(resume_path, env, seconds, signature, device)
    else:
        clock = SlotClock(seconds)
        model = make_agent(env, device, seed, clock, overrides)
        env.reset(seed=seed)  # establish a snapshot before the first learning call
        state = dict(signature=signature, best_score=None, history=[], next_validation=0.,
                     next_checkpoint=checkpoint_seconds, validation_time_estimate=0.)
    model.set_logger(configure(str(output), ['csv']))

    def validate():
        started = time.monotonic()
        result = evaluate(val_env, model)
        row = dict(seconds=clock.elapsed, decisions=model.num_timesteps,
                   mean_cost=result['mean_cost'], learning_rate=clock.learning_rate(), epsilon=clock.epsilon())
        state['history'].append(row)
        if state['best_score'] is None or result['mean_cost'] < state['best_score']:
            state['best_score'] = result['mean_cost']
            atomic_model(model, output/'best.zip')
            save_json(output/'best.json', dict(signature=signature, validation=result, **row,
                       feature_shapes={k:list(v) for k, v in env.feature_shapes.items()},
                       household_ids=train_provider.preprocessor.ids,
                       preprocessing=train_provider.preprocessor.state))
        save_json(output/'history.json', state['history'])
        model.logger.record('validation/cost', result['mean_cost'])
        model.logger.record('budget/seconds', clock.elapsed)
        model.logger.dump(step=model.num_timesteps)
        print(f'{output.name}: {model.num_timesteps} decisions; {clock.elapsed:.1f}/{seconds:.1f}s; '
              f'validation cost={result["mean_cost"]:.7f}; best={state["best_score"]:.7f}; '
              f'lr={clock.learning_rate():.7f}', flush=True)
        state['next_validation'] = (int(clock.elapsed/validation_seconds)+1)*validation_seconds
        state['validation_time_estimate'] = max(state.get('validation_time_estimate', 0.),
                                                1.1*(time.monotonic()-started))

    try:
        if not state['history']:
            validate()
            save_resume(resume_path, model, env, clock, state, device)
        next_progress = clock.elapsed
        # Reserve the measured validation duration so final evaluation belongs
        # within the slot rather than adding a full validation pass after it.
        while clock.elapsed + state.get('validation_time_estimate', 0.) < seconds:
            # Each call completes 24 transitions AND its gradient updates. Saving in
            # SB3's on_step callback would omit the newest replay transition.
            model.learn(total_timesteps=24, reset_num_timesteps=False, log_interval=None)
            require(env.hour == 0, 'Expected VecEnv automatic reset at the day boundary.')
            if clock.elapsed >= state['next_validation']:
                validate()
            if clock.elapsed >= state['next_checkpoint']:
                state['next_checkpoint'] = (int(clock.elapsed/checkpoint_seconds)+1)*checkpoint_seconds
                save_resume(resume_path, model, env, clock, state, device)
            if clock.elapsed >= next_progress:
                episodes = list(model.ep_info_buffer) if model.ep_info_buffer else []
                progress = dict(seconds=clock.elapsed, budget_seconds=seconds, decisions=model.num_timesteps,
                                updates=model._n_updates, best_validation_cost=state['best_score'],
                                mean_ep_reward=float(np.mean([e['r'] for e in episodes])) if episodes else None,
                                learning_rate=clock.learning_rate(), epsilon=clock.epsilon())
                save_json(output/'progress.json', progress)
                print(f'{output.name}: {model.num_timesteps} decisions, {clock.elapsed:.0f}/{seconds:.0f}s, '
                      f'epsilon={clock.epsilon():.3f}, lr={clock.learning_rate():.7f}', flush=True)
                next_progress = clock.elapsed+60.
        # Evaluate the final policy even when the last interval was not reached.
        if state['history'][-1]['decisions'] != model.num_timesteps:
            validate()
        best = read_json(output/'best.json')
        result = dict(cap=cap, seed=seed, validation=best['validation'], best_decisions=best['decisions'],
                      decisions=model.num_timesteps, elapsed_seconds=clock.elapsed,
                      updates=model._n_updates, best_checkpoint=str((output/'best.zip').resolve()))
        save_json(output/'result.json', result)
        resume_path.unlink(missing_ok=True)
        return result
    except (KeyboardInterrupt, Exception):
        # The last periodic snapshot is coherent even if an interruption happened
        # during a transition/update; never overwrite it with partial learner state.
        raise


def parity(data, engine, provider, response_root):
    episode = provider.episodes[0]
    history = provider.features(episode)
    prediction = engine.predict(history, 1., episode.future_calendar[0])
    if engine.device.type == 'mps':
        cpu = make_engine(response_root, 'ensemble', data, 'cpu')
        expected = cpu.predict(history, 1., episode.future_calendar[0])
        np.testing.assert_allclose(prediction, expected, rtol=5e-4, atol=1e-5)
        del cpu
    return dict(device=str(engine.device), all_households_finite=bool(np.isfinite(prediction).all()),
                parity_rtol=5e-4, parity_atol_kwh=1e-5)


def _run(data_folder, response_root, output, budget_hours=10., device='mps',
         validation_seconds=600., checkpoint_seconds=300., overrides=None, limit_days=None):
    require(budget_hours>0, 'Budget must be positive.')
    require(validation_seconds>0 and checkpoint_seconds>0, 'Invalid checkpoint/evaluation interval.')
    torch.set_num_threads(4)
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    data = PreparedData(data_folder)
    if limit_days is not None:
        # Internal fixture-only integration tests. Official runs never limit days.
        for split, origins in data.origins.items():
            data.origins[split] = np.array([i for i in origins if i%24==0][:limit_days])
    engine = make_engine(response_root, 'ensemble', data, device)
    seconds = budget_hours*3600/6
    settings = dict(version=1, sources=source_hashes(), data=digest(data.folder/'manifest.json'),
                    engine=engine.signature, budget_hours=budget_hours, slot_seconds=seconds,
                    device=device, run_order=[list(x) for x in RUN_ORDER],
                    settings={**DEFAULT_SETTINGS, **(overrides or {})}, limit_days=limit_days,
                    validation_seconds=validation_seconds, checkpoint_seconds=checkpoint_seconds)
    if (output/'settings.json').is_file():
        require(read_json(output/'settings.json') == settings, 'Experiment inputs/settings changed; choose a new output.')
    else:
        save_json(output/'settings.json', settings)
    if (output/'status.json').is_file() and read_json(output/'status.json').get('state') == 'complete':
        print(f'Already complete: {output}', flush=True);return
    train = EpisodeProvider(data, 'train');val = EpisodeProvider(data, 'validation')
    cache = output/'reference_cache'
    save_json(output/'status.json', dict(state='setup', stage='MPS parity and validation diagnostics'))
    save_json(output/'startup_checks.json', parity(data, engine, train, response_root))
    diagnostic = diagnostics(data, val, engine, cache, output/'diagnostics')
    validation_rows = constants(val, engine, cache, output/'diagnostics/validation_constants')
    for cap in (2., 3.):
        baseline_suite(val, engine, cache, output/'validation_baselines', cap,
                       diagnostic['validation_fixed_indices'][str(int(cap))], validation_rows)
    # Remove reference-creation overhead from the six timed training slots.
    save_json(output/'status.json', dict(state='setup', stage='training reference cache'))
    warm = PriceEnv(train, engine, cache, 3.)
    for i in range(len(train.episodes)):
        warm.reference_cache.get(i)
        if (i+1)%25 == 0 or i+1 == len(train.episodes):
            save_json(output/'status.json', dict(state='setup', stage='training reference cache',
                                                days_done=i+1, total_days=len(train.episodes)))
            print(f'Training reference days {i+1}/{len(train.episodes)}', flush=True)
    del warm
    if device == 'mps':
        torch.mps.empty_cache()
    results = []
    for cap, seed in RUN_ORDER:
        name = f'cap{int(cap)}_seed{seed}'
        save_json(output/'status.json', dict(state='training', run=name, completed_fits=len(results),
                                           total_fits=6, slot_seconds=seconds))
        result = fit(train, val, engine, output/name, cap, seed, seconds, cache, device,
                     validation_seconds, checkpoint_seconds, overrides)
        results.append(result)
        save_json(output/'validation_results.json', results)
        import gc
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    cap_means = {str(int(c)):float(np.mean([r['validation']['mean_cost'] for r in results if r['cap']==c])) for c in (2., 3.)}
    cap = 2. if cap_means['2'] <= cap_means['3']+1e-6 else 3.
    deployment = min((r for r in results if r['cap']==cap), key=lambda r:(r['validation']['mean_cost'], r['seed']))
    selection = dict(cap=cap, deployment_seed=deployment['seed'], deployment_checkpoint=deployment['best_checkpoint'],
                     cap_mean_validation_cost=cap_means, frozen_before_test=True,
                     validation_fixed_indices=diagnostic['validation_fixed_indices'],
                     rule='Three-seed mean ensemble-validation cost; ties within 1e-6 prefer cap2. Deployment seed selected by validation only.')
    selection_file = output/'selection.json'
    if selection_file.exists():
        require(read_json(selection_file)==selection, 'Frozen selection changed.')
    else:
        save_json(selection_file, selection)
    # Reserved dynamics and future-month snapshots are constructed only after the seal.
    test = EpisodeProvider(data, 'test', selection_file)
    save_json(output/'status.json', dict(state='testing', stage='all frozen policies and baselines'))
    support = read_json(output/'diagnostics/price_support.json')
    evaluated, baseline_results = {}, {}
    for scenario in ('ensemble', 's42', 's7', 'reserved', 'control'):
        current = engine if scenario=='ensemble' else make_engine(response_root, scenario, data, device, selection_file)
        evaluated[scenario], baseline_results[scenario] = {}, {}
        save_json(output/'status.json', dict(state='testing', scenario=scenario))
        rows = constants(test, current, cache, output/'test'/scenario/'constants')
        for c in (2., 3.):
            key = str(int(c))
            evaluated[scenario][key] = []
            for seed in SEEDS:
                path = output/'test'/scenario/f'cap{key}_seed{seed}.json'
                if path.is_file():
                    result = read_json(path)
                else:
                    policy = DQN.load(output/f'cap{key}_seed{seed}'/'best.zip', device=device)
                    env = PriceEnv(test, current, cache, c)
                    result = add_support(evaluate(env, policy), support, test.area_index)
                    save_json(path, result)
                    del policy
                evaluated[scenario][key].append(result)
            baseline_results[scenario][key] = baseline_suite(test, current, cache, output/'test'/scenario, c,
                         diagnostic['validation_fixed_indices'][key], rows)
        if current is not engine:
            del current
        if device == 'mps':
            torch.mps.empty_cache()
    save_json(output/'test_metrics.json', dict(policies=evaluated, baselines=baseline_results))
    # Price-free physics must not depend on the price schedule on ANY test day,
    # even though policy actions can differ as their observations change.
    control = evaluated['control']
    reference_curves = np.asarray([d['regional_kwh'] for d in control['2'][0]['days']])
    for runs in control.values():
        for policy_result in runs:
            np.testing.assert_allclose([d['regional_kwh'] for d in policy_result['days']],
                                       reference_curves, rtol=1e-5, atol=1e-5)
    result = final_report(output, selection, evaluated, baseline_results)
    # Curves expose compute differences rather than presenting unequal transitions as equal experience.
    plt = plotting(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cap, seed in RUN_ORDER:
        rows = read_json(output/f'cap{int(cap)}_seed{seed}'/'history.json')
        for ax, key in zip(axes, ('seconds', 'decisions')):
            ax.plot([r[key] for r in rows], [r['mean_cost'] for r in rows], label=f'{int(cap)} / {seed}')
            ax.set(xlabel=key, ylabel='Mean validation cost')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'learning_curves.png', dpi=160);plt.close(fig)
    save_json(output/'status.json', dict(state='complete', selection=selection,
                                       robust_wider_range_benefit=result['robust_wider_range_benefit'],
                                       switching_benefit=result['switching_benefit_survives_reserved']))
    print(f'Complete: {output/"REPORT.md"}', flush=True)


def run(*args, **kwargs):
    import fcntl
    output = Path(kwargs.get('output', args[2] if len(args)>2 else ROOT/'runs/dqn_v1'))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process owns {output}') from error
        try:
            return _run(*args, **kwargs)
        except BaseException as error:
            save_json(output/'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error), resume='Run the identical command; completed fits are reused.'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'prepared')
    parser.add_argument('--response-root', type=Path, default=ROOT/'runs/comparison_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/dqn_v1')
    parser.add_argument('--budget-hours', type=float, default=10.)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--evaluate', type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--engine', choices=['ensemble', 's42', 's7', 'reserved', 'control'], default='ensemble')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.check_only or args.evaluate:
        data = PreparedData(args.data)
        selection = args.output/'selection.json'
        engine = make_engine(args.response_root, args.engine, data, args.device, selection)
        provider = EpisodeProvider(data, args.split if args.evaluate else 'train', selection)
        if args.check_only:
            print(json.dumps(parity(data, engine, provider, args.response_root)), flush=True)
            env = PriceEnv(provider, engine, args.output/'reference_cache', 3.)
            clock = SlotClock(60.)
            agent = make_agent(env, args.device, 42, clock, dict(buffer_size=64, learning_starts=8, batch_size=8))
            agent.learn(total_timesteps=32)
            require(agent._n_updates>0 and all(torch.isfinite(p).all() for p in agent.policy.parameters()), 'Invalid DQN update.')
            print(f'Full household simulator and DQN update passed on {args.device}.');return
        meta = read_json(args.evaluate.with_name('best.json'))
        require(meta['signature']['data']==digest(data.folder/'manifest.json'), 'Policy/data mismatch.')
        cap = meta['signature']['cap']
        env = PriceEnv(provider, engine, args.output/'reference_cache', cap)
        model = DQN.load(args.evaluate, device=args.device)
        result = evaluate(env, model)
        path = args.output/'manual_evaluation'/f'{args.evaluate.parent.name}_{args.engine}_{args.split}.json'
        path.parent.mkdir(parents=True, exist_ok=True);save_json(path, result)
        print(json.dumps({k:v for k, v in result.items() if k!='days'}, indent=2));return
    run(args.data, args.response_root, args.output, args.budget_hours, args.device)


if __name__ == '__main__':
    main()
