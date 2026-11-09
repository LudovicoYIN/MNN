from pathlib import Path
import subprocess,json,time,os
root=Path.cwd(); out=root/'upstream-concurrency-ab';out.mkdir(exist_ok=True)
repo=Path('/home/yinhanke/code/MNN')
names=['source/backend/cuda/execution/AttentionExecution.cu','source/backend/cuda/execution/AttentionExecution.hpp','source/backend/cuda/execution/RoPEExecution.cu','transformers/llm/engine/tools/llm_bench.cpp']
saved={n:(repo/n).read_bytes() for n in names}
base='b0239771b07b317a258fbc384683bbd540741406'
original={n:subprocess.check_output(['git','-C',str(repo),'show',base+':'+n]) for n in names}
groups=[[512],[2048],[4096],[512,512],[2048,2048],[128,4096],[512]*4,[128,512,1024,2048]]
results=[]
def build(label):
 with (out/(label+'-build.log')).open('w') as f:subprocess.run(['cmake','--build','build-cuda-baseline','--target','llm_bench','-j','8'],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 for variant in ['baseline','attention-only']:
  for n in names:(repo/n).write_bytes(saved[n] if variant=='attention-only' and 'AttentionExecution.' in n else original[n])
  build(variant)
  for model in ['Qwen3.5-4B','MiniCPM5-2B']:
   modeldir=Path('/home/yinhanke/code/models')/(model+'-MNN-Q4')
   for lengths in groups:
    label=variant+'-'+model+'-'+'x'.join(map(str,lengths));print(label,flush=True)
    jobs=[];started=time.monotonic();peak=0
    try:
     for i,n in enumerate(lengths):
      key=label+'-'+str(i);cfg=json.loads((modeldir/'text-config.json').read_text());cache=out/(key+'-cache');cache.mkdir(exist_ok=True)
      cfg.update(base_dir=str(modeldir)+'/',tmp_path=str(cache)+'/',backend_type='cuda')
      cp=out/(key+'-config.json');cp.write_text(json.dumps(cfg));jp=out/(key+'.json');lp=out/(key+'.log');f=lp.open('w')
      args=[str(root/'build-cuda-baseline/llm_bench'),'-m',str(cp),'-a','cuda','-c','2','-t','8','-p','0','-n','0','-pg',str(n)+',32','-rep','3','-qa','0','-fa','1','-mmp','0','-dyo','0','-j',str(jp)]
      jobs.append((subprocess.Popen(args,cwd=cache,stdout=f,stderr=subprocess.STDOUT),f,jp,n))
     while any(p.poll() is None for p,_,_,_ in jobs):
      q=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
      if q.returncode==0:peak=max(peak,int(q.stdout.strip().splitlines()[0]))
      time.sleep(.5)
    finally:
     for p,f,_,_ in jobs:p.wait();f.close()
    workers=[]
    for p,f,jp,n in jobs:
     r=dict(input_tokens=n,exit_code=p.returncode)
     if p.returncode==0 and jp.exists():
      try:r['result']=json.loads(jp.read_text())['results'][0]
      except Exception:r['invalid_json']=True
     workers.append(r)
    row=dict(variant=variant,model=model,lengths=lengths,wall_s=time.monotonic()-started,sampled_whole_gpu_peak_mib=peak,workers=workers)
    if all('result' in w for w in workers):row['sum_worker_decode_tps']=sum(w['result']['decode_tps'] for w in workers)
    results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2))
finally:
 
 for n in names:(repo/n).write_bytes(saved[n])
 build('restored-candidate')
