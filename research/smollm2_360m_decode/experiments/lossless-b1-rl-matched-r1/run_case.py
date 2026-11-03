"""One full RL update, serial B1 inference, deterministic distinct group seeds."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/pradheep/tiny-prl/prime-rl');HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('arm',choices=['baseline','candidate','early']);p.add_argument('run_id');a=p.parse_args()
base=ROOT/'outputs/megakernel-20261025/lossless-b1-vadd2-r1/frozen-package/baseline.so'
current=ROOT/'outputs/megakernel-20261025/lossless-b2-vadd2-r1/frozen-package/baseline.so'
lib=base if a.arm=='baseline' else current
out=HERE/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
py_path=':'.join(map(str,[HERE/'python',ROOT/'src',ROOT/'packages/prime-rl-configs/src',ROOT.parent/'prime-kernels']))
trace=out/'trace';trace.mkdir()
codec=ROOT/'outputs/megakernel-20261025/lossless-b2-vadd2-r1/codec-build'
config=out/'benchmark.toml'
config.write_text(('[deployment]\nearly_trainer_start = true\n' if a.arm=='early' else '')+'[inference]\ndecode_megakernel = true\n[inference.vllm]\nmax_num_seqs = 1\n[inference.env_vars]\nPRIME_DECODE_MEGAKERNEL_360M_LOSSLESS = "1"\nTINY_MATCH_LIBRARY = '+json.dumps(str(lib))+'\nTINY_MATCH_TRACE = '+json.dumps(str(trace))+'\nPYTHONPATH = '+json.dumps(py_path)+'\nTORCH_EXTENSIONS_DIR = '+json.dumps(str(codec))+'\n[orchestrator.concurrency]\nmin_inflight = 2\nmax_inflight = 2\ninitial_inflight = 2\n')
model='/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'
command=['/home/pradheep/.local/bin/uv','run','--offline','--no-project','--python','/home/pradheep/.venv/bin/python','rl','@',str(ROOT/'examples/tiny/smollm2-360m-full-stream64.toml'),'@',str(config),'--max-steps','1','--model.name',model,'--output-dir',str(out),'--run.name','rl','--no-dashboard','--no-ckpt']
env=os.environ.copy();env.update(PYTHONPATH=py_path,TINY_MATCH_TRACE=str(trace),TORCH_EXTENSIONS_DIR=str(codec),MAX_JOBS='1',PRL_RUN_ID=a.run_id)
start=time.monotonic()
with (out/'rl.log').open('w') as log:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
summary={'arm':a.arm,'run_id':a.run_id,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'command':command,'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
(out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
raise SystemExit(r.returncode)
