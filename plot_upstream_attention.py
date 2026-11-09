from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=json.loads(Path('upstream-attention-ab/comparison.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,model in zip(axes,['Qwen3.5-4B','MiniCPM5-2B']):
 r=[x for x in rows if x['model']==model];x=np.arange(len(r));w=.36
 for offset,key,label,color in [(-w/2,'baseline','Original upstream','#7d8996'),(w/2,'attention_only','Attention-only','#167dba')]:
  ax.bar(x+offset,[v[key]['decode_tps'] for v in r],w,yerr=[v[key]['decode_std'] for v in r],capsize=3,label=label,color=color)
 for i,v in enumerate(r):ax.text(i,max(v[k]['decode_tps'] for k in ['baseline','attention_only'])+4,f"{v['attention_only']['decode_tps']/v['baseline']['decode_tps']:.2f}×",ha='center')
 ax.set(title=model,xlabel='Input length (tokens)',xticks=x,xticklabels=[v['input_tokens'] for v in r],ylim=(0,175))
 ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Decode throughput (tokens/s; higher is better)');axes[0].legend()
fig.suptitle('CUDA masked Attention: original upstream vs Attention-only patch')
fig.text(.5,.025,'RTX 3090 · Q4 · 32 output tokens · mean ± 1 SD, 3 runs after warmup\nBoth builds exclude the separate RoPE fix; baseline b0239771.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.10,1,.94));fig.savefig('attention-upstream-lengths.png',dpi=180);fig.savefig('attention-upstream-lengths.svg');plt.close(fig)

data=json.loads(Path('upstream-concurrency-ab/results.json').read_text())
groups=[[512],[2048],[4096],[512,512],[2048,2048],[128,4096],[512]*4,[128,512,1024,2048]]
labels=['1 × 512','1 × 2048','1 × 4096','2 × 512','2 × 2048','2 mixed\n128 + 4096','4 × 512','4 mixed\n128/512/1024/2048']
fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
summary=[]
for ax,model in zip(axes,['Qwen3.5-4B','MiniCPM5-2B']):
 x=np.arange(len(groups));w=.36
 for offset,variant,label,color in [(-w/2,'baseline','Original upstream','#7d8996'),(w/2,'attention-only','Attention-only','#167dba')]:
  selected=[next(r for r in data if r['model']==model and r['variant']==variant and r['lengths']==g) for g in groups]
  assert all(all('result' in worker for worker in r['workers']) for r in selected)
  ax.bar(x+offset,[r['wall_s'] for r in selected],w,label=label,color=color)
 for g in groups:
  a,b=[next(r for r in data if r['model']==model and r['variant']==v and r['lengths']==g) for v in ['baseline','attention-only']]
  summary.append({'model':model,'lengths':g,'baseline_wall_s':a['wall_s'],'attention_only_wall_s':b['wall_s'],'wall_reduction_percent':100*(1-b['wall_s']/a['wall_s'])})
 ax.set(title=model,xlabel='Concurrent processes × input tokens (mixed lengths listed)',xticks=x,xticklabels=labels,ylim=(0,max(r['wall_s'] for r in data)*1.12));ax.tick_params(axis='x',labelrotation=40);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Whole-group completion time (s; lower is better)');axes[0].legend()
fig.suptitle('Independent-process concurrency: original upstream vs Attention-only')
fig.text(.5,.02,'Each worker: batch 1, separate model/KV/cache, 32 output tokens, warmup + 3 repetitions.\nOne group launch per configuration. Includes loading, warmup, prefill, decode and sampling; not steady-state serving latency.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.17,1,.94));fig.savefig('attention-upstream-concurrency.png',dpi=180);fig.savefig('attention-upstream-concurrency.svg')
Path('upstream-concurrency-summary.json').write_text(json.dumps(summary,indent=2))
