import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

rows = json.loads(Path('verified-ab/comparison.json').read_text())
models = ['Qwen3.5-4B', 'MiniCPM5-2B']
lengths = [512, 2048, 4096]
colors = ['#94A3B8', '#2563EB']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.18, hspace=0.48, wspace=0.22)
fig.suptitle('CUDA RoPE layout fix: LLM throughput', x=0.075, y=0.965,
             ha='left', fontsize=22, fontweight='bold')
fig.text(0.075, 0.912, 'RTX 3090  |  Q4/HQQ weights, block 64  |  FP16 activations  |  Higher is better',
         fontsize=12, color='#475569')
legend_handles = []
for i, model in enumerate(models):
    subset = {r['context']: r for r in rows if r['model'] == model}
    for j, phase in enumerate(['prefill', 'decode']):
        ax = axes[i, j]
        key, std_key = phase + '_tps', phase + '_std'
        x = np.arange(3)
        upper = 6500 if phase == 'prefill' else 94
        for k, variant in enumerate(['before', 'after']):
            means = [subset[n][variant][key] for n in lengths]
            stds = [subset[n][variant][std_key] for n in lengths]
            bars = ax.bar(x + (k - 0.5) * 0.32, means, width=0.30,
                          color=colors[k], yerr=stds, capsize=3,
                          error_kw={'linewidth': 1, 'ecolor': '#334155'}, zorder=3)
            if i == 0 and j == 0:
                legend_handles.append(bars[0])
            for bar, mean, std in zip(bars, means, stds):
                label = f'{mean:,.0f}' if phase == 'prefill' else f'{mean:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2, mean + std + upper*0.013,
                        label, ha='center', va='bottom', fontsize=9, color='#334155')
        for pos, n in enumerate(lengths):
            before, after = subset[n]['before'][key], subset[n]['after'][key]
            top = max(subset[n][v][key]+subset[n][v][std_key] for v in ['before', 'after'])
            gain = 100 * (after / before - 1)
            ax.text(pos, top + upper*0.105, f'+{gain:.1f}%', ha='center',
                    fontsize=11, fontweight='bold', color='#1D4ED8')
        ax.set_title(f'{model} / {phase.capitalize()}', loc='left', fontweight='bold', fontsize=13, pad=12)
        ax.set_xticks(x, [f'{n:,}' for n in lengths])
        ax.set_xlabel('Input length (tokens)')
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_ylim(0, upper)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', color='#E2E8F0', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['bottom'].set_color('#CBD5E1')
fig.legend(legend_handles, ['Before: RoPE falls back to CPU', 'After: CUDA RoPE layout fix'],
           loc='upper left', bbox_to_anchor=(0.069, 0.895), ncol=2, frameon=False, fontsize=11)
fig.text(0.075, 0.03,
         'Bars: mean throughput; error bars: ±1 standard deviation over 5 repetitions after 1 discarded warmup.\n'
         'Decode: 128 generated tokens following each input. Same model/configuration; sequential runs.\n'
         'Percent labels: throughput increase vs. before (not whole-response latency reduction).\n'
         'Baseline b0239771; CUDA 13.2; driver 595.99.02. Both variants use the same benchmark JSON label fix.',
         fontsize=10, color='#475569', linespacing=1.5)
for ext in ['png', 'svg']:
    fig.savefig(f'cuda-rope-throughput.{ext}', dpi=180, facecolor='white')
print('Saved cuda-rope-throughput.png and cuda-rope-throughput.svg')
