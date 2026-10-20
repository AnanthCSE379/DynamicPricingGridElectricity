"""Plot the recorded tariff frequency, counting each half-hour timestamp once."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tariff-frequency-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(data, output):
    prices = np.load(Path(data) / 'prices.npy', allow_pickle=False)
    times = np.load(Path(data) / 'times.npy', allow_pickle=False)
    if (prices.ndim != 1 or prices.size == 0 or times.shape != prices.shape
            or not np.isin(prices, [0, 1, 2]).all()
            or len(np.unique(times)) != len(times)):
        raise ValueError('Require unique timestamps and matching tariff IDs: 0=Low, 1=Normal, 2=High.')
    counts = np.bincount(prices.astype(int), minlength=3)
    shares = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    bars = ax.bar(['Low', 'Normal', 'High'], counts,
                  color=['#259D85', '#4778C9', '#D87552'], width=.58)
    ax.bar_label(bars, labels=[f'{n:,}\n{p:.1f}%' for n, p in zip(counts, shares)],
                 padding=7, fontsize=12)
    ax.set(title='Recorded electricity tariff frequency', ylabel='Number of half-hour intervals')
    ax.set_ylim(0, counts.max() * 1.23)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0, labelsize=12)
    ax.text(.5, -.12, f'{counts.sum():,} unique half-hour intervals • all dataset partitions',
            transform=ax.transAxes, ha='center', fontsize=10, color='#555555')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)
    for label, n, share in zip(['Low', 'Normal', 'High'], counts, shares):
        print(f'{label}: {n:,} intervals ({share:.2f}%)')
    print(f'Saved: {output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'results/five_hour_response/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/plots/tariff_frequency.png')
    args = parser.parse_args()
    plot(args.data, args.output)
