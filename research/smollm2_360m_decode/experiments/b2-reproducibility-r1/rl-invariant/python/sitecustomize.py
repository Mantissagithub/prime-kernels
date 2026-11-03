"""Private benchmark-only deterministic requests and lossless-library override."""
import hashlib,json,os
from pathlib import Path

def record(kind,row):
    root=os.environ.get('TINY_MATCH_TRACE')
    if root:
        p=Path(root);p.mkdir(parents=True,exist_ok=True)
        with (p/f'{kind}-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')

if os.environ.get('TINY_MATCH_TRACE'):
    import httpx
    send=httpx.AsyncClient.send
    indices={};sessions={}
    async def seeded_send(self,request,*args,**kwargs):
        if request.url.path!='/inference/v1/generate':
            return await send(self,request,*args,**kwargs)
        body=json.loads(request.content)
        prompt=hashlib.sha256(json.dumps(body['token_ids'],separators=(',',':')).encode()).hexdigest()
        session=(prompt,request.headers.get('X-Session-ID'))
        if session not in sessions:
            index=indices.get(prompt,0);indices[prompt]=index+1
            sessions[session]=int(hashlib.sha256(f'{prompt}:{index}:4609'.encode()).hexdigest()[:8],16)%2147483647
        seed=sessions[session];body['sampling_params']['seed']=seed
        headers=dict(request.headers);headers.pop('content-length',None)
        request=httpx.Request(request.method,request.url,headers=headers,content=json.dumps(body).encode(),extensions=request.extensions)
        row={'prompt_sha256':prompt,'seed':seed,'input_tokens':len(body['token_ids'])}
        try:
            response=await send(self,request,*args,**kwargs)
            await response.aread();row['status']=response.status_code
            if response.status_code==200:
                data=response.json();ids=data['choices'][0]['token_ids']
                row['output_tokens']=len(ids);row['output_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            record('request',row);return response
        except BaseException as error:
            record('request',{**row,'error':type(error).__name__});raise
    httpx.AsyncClient.send=seeded_send

library=os.environ.get('TINY_MATCH_LIBRARY')
if library:
    import torch.utils.cpp_extension as cpp
    original=cpp.load
    def load(name,*args,**kwargs):
        if name!='prime_decode_megakernel_360m_lossless':return original(name,*args,**kwargs)
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from prime_kernels import decode_megakernel_360m_lossless as module
        counters={'execute_calls':0,'scheduled_tokens':0,'batch_histogram':{}}
        execute=GPUModelRunner.execute_model
        def counted(self,scheduler_output,*a,**kw):
            counts=scheduler_output.num_scheduled_tokens
            counters['execute_calls']+=1;counters['scheduled_tokens']+=sum(counts.values())
            key=str(len(counts));counters['batch_histogram'][key]=counters['batch_histogram'].get(key,0)+1
            return execute(self,scheduler_output,*a,**kw)
        GPUModelRunner.execute_model=counted
        refresh=module.refresh_weights
        def refreshed(layers):
            result=refresh(layers)
            record('kernel',{'event':'refresh','generation':module.cache_statistics()['generation'],'work':counters})
            return result
        module.refresh_weights=refreshed
        record('kernel',{'event':'library','path':library,'sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest()})
        return library
    cpp.load=load
