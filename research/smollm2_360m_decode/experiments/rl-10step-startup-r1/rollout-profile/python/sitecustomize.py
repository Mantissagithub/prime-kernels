"""Private benchmark-only deterministic requests and lossless-library override."""
import hashlib,json,os,time
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
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        import torch
        from prime_kernels import decode_megakernel_360m_lossless as module
        pending=[]
        current={}
        execute=GPUModelRunner.execute_model
        sample=GPUModelRunner.sample_tokens
        def flush_events():
            for begin,end,row in pending:
                end.synchronize()
                record('gpu-profile',{**row,'gpu_stream_span_ms':begin.elapsed_time(end)})
            pending.clear()
        def measured(method,phase,self,*a,**kw):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();started=time.perf_counter()
            result=method(self,*a,**kw)
            end.record()
            pending.append((begin,end,{**current,'phase':phase,'cpu_wall_ms':(time.perf_counter()-started)*1000}))
            return result
        def counted(self,scheduler_output,*a,**kw):
            if kw.get('dummy_run',False):return execute(self,scheduler_output,*a,**kw)
            counts=list(scheduler_output.num_scheduled_tokens.values())
            if not counts:return execute(self,scheduler_output,*a,**kw)
            current.clear();current.update(batch=len(counts),scheduled_tokens=sum(counts),
                shape='decode' if max(counts)==1 else ('mixed' if 1 in counts else 'prefill'))
            return measured(execute,'execute_model',self,scheduler_output,*a,**kw)
        def sampled(self,*a,**kw):
            return measured(sample,'sample_tokens',self,*a,**kw)
        GPUModelRunner.execute_model=counted
        GPUModelRunner.sample_tokens=sampled
        refresh=module.refresh_weights
        def refreshed(layers):
            result=refresh(layers)
            from prime_rl.inference.vllm import decode_backend
            flush_events()
            record('kernel',{'event':'refresh','generation':module.cache_statistics()['generation'],'backend_stats':dict(decode_backend.stats)})
            return result
        module.refresh_weights=refreshed
        record('kernel',{'event':'library','path':library,'sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest()})
        return library
    cpp.load=load
