"""Diagnostic two-update RL; timing is excluded from comparisons. with checkpoints and distinct group seeds."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/pradheep/tiny-prl/prime-rl');HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('arm',choices=['control','early']);p.add_argument('run_id');p.add_argument('--retain-checkpoint',action='store_true');a=p.parse_args()
current=ROOT/'outputs/megakernel-20261025/lossless-b2-vadd2-r1/frozen-package/baseline.so'
lib=current
out=HERE/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
py_path=':'.join(map(str,[HERE/'python',ROOT/'src',ROOT/'packages/prime-rl-configs/src',ROOT.parent/'prime-kernels']))
trace=out/'trace';trace.mkdir()
codec=ROOT/'outputs/megakernel-20261025/lossless-b2-vadd2-r1/codec-build'
config=out/'benchmark.toml'
config.write_text(('[deployment]\nearly_trainer_start = true\n' if a.arm=='early' else '')+'[inference]\ndecode_megakernel = true\n[inference.vllm]\nmax_num_seqs = 2\n[inference.env_vars]\nVLLM_BATCH_INVARIANT = "1"\nPRIME_DECODE_MEGAKERNEL_360M_LOSSLESS = "1"\nTINY_MATCH_LIBRARY = '+json.dumps(str(lib))+'\nTINY_MATCH_TRACE = '+json.dumps(str(trace))+'\nPYTHONPATH = '+json.dumps(py_path)+'\nTORCH_EXTENSIONS_DIR = '+json.dumps(str(codec))+'\n')
model='/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'
command=['/home/pradheep/.local/bin/uv','run','--offline','--no-project','--python','/home/pradheep/.venv/bin/python','rl','@',str(ROOT/'examples/tiny/smollm2-360m-full-stream64.toml'),'@',str(config),'@',str(ROOT/'examples/tiny/laptop.toml'),'--max-steps','2','--model.name',model,'--output-dir',str(out),'--run.name','rl','--no-dashboard']
env=os.environ.copy();env.update(PYTHONPATH=py_path,TINY_MATCH_TRACE=str(trace),TORCH_EXTENSIONS_DIR=str(codec),MAX_JOBS='1',PRL_RUN_ID=a.run_id)
started_unix=time.time();start=time.monotonic()
with (out/'rl.log').open('w') as log:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
summary={'arm':a.arm,'run_id':a.run_id,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'started_unix':started_unix,'command':command,'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
(out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if r.returncode == 0:
    import shutil
    files=[f for f in (out/'rl/checkpoints').rglob('*') if f.is_file()]
    if not files:
        files=[f for f in (out/'rl').rglob('*') if f.is_file() and 'checkpoints' in f.parts]
    assert files, 'Successful run must save a checkpoint'
    manifest=[]
    for f in sorted(files):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024),b''):
                h.update(block)
        manifest.append({'path':str(f.relative_to(out)), 'bytes':f.stat().st_size, 'sha256':h.hexdigest()})
    (out/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.rmtree(out/'rl/broadcasts')
    if not a.retain_checkpoint:
        roots={out/Path(entry['path']).parts[0]/Path(entry['path']).parts[1] for entry in manifest}
        assert all(root.name=='checkpoints' for root in roots), roots
        for root in roots: shutil.rmtree(root)
    print('Checkpoint bytes:',sum(row['bytes'] for row in manifest),flush=True)

raise SystemExit(r.returncode)
