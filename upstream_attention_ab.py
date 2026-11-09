from pathlib import Path
import subprocess,json
repo=Path('/home/yinhanke/code/MNN');out=Path.cwd()/'upstream-attention-ab';out.mkdir(exist_ok=True)
base='b0239771b07b317a258fbc384683bbd540741406'
names=['source/backend/cuda/execution/AttentionExecution.cu','source/backend/cuda/execution/AttentionExecution.hpp','source/backend/cuda/execution/RoPEExecution.cu','transformers/llm/engine/tools/llm_bench.cpp']
saved={n:(repo/n).read_bytes() for n in names}
original={n:subprocess.check_output(['git','-C',str(repo),'show',base+':'+n]) for n in names}
for n in names:
 (out/('original-'+Path(n).name)).write_bytes(original[n])
 (out/('saved-'+Path(n).name)).write_bytes(saved[n])
def run(args,label):
 print(label,flush=True)
 with (out/(label+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 for variant in ['baseline','attention-only']:
  for n in names:(repo/n).write_bytes(saved[n] if variant=='attention-only' and 'AttentionExecution.' in n else original[n])
  run(['cmake','--build','build-cuda-baseline','--target','llm_bench','-j','8'],variant+'-build')
  run(['./review_attention_backend'],variant+'-backend')
  for model in ['Qwen3.5-4B','MiniCPM5-2B']:
   for length in [512,2048,4096]:
    label=variant+'-'+model+'-'+str(length)
    run(['./build-cuda-baseline/llm_bench','-m','/home/yinhanke/code/models/'+model+'-MNN-Q4/text-config.json','-a','cuda','-c','2','-t','8','-p','0','-n','0','-pg',str(length)+',32','-rep','3','-qa','0','-fa','1','-mmp','0','-dyo','0','-j',str(out/(label+'.json'))],label)
finally:
 for n in names:(repo/n).write_bytes(saved[n])
 run(['cmake','--build','build-cuda-baseline','--target','llm_bench','-j','8'],'restored-build')
rows=[]
for model in ['Qwen3.5-4B','MiniCPM5-2B']:
 for length in [512,2048,4096]:
  pair=[json.loads((out/(v+'-'+model+'-'+str(length)+'.json')).read_text())['results'][0] for v in ['baseline','attention-only']]
  rows.append({'model':model,'input_tokens':length,'baseline':pair[0],'attention_only':pair[1]})
(out/'comparison.json').write_text(json.dumps({'upstream_commit':base,'scope':'Both variants exclude RoPE PR and benchmark JSON label fix; upstream JSON incorrectly labels CUDA as CPU. CLI uses -a cuda.','rows':rows},indent=2))
