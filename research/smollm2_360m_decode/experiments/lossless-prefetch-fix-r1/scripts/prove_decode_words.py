import ctypes
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'prime-kernels'))
import torch
from prime_kernels.lossless_bf16 import EncodedGUCache, ROWS, COLS, CHUNKS

bits = (torch.arange(ROWS * COLS, device='cuda', dtype=torch.int32) % 65536).to(torch.uint16).reshape(ROWS, COLS)
original = bits.clone()
ones = torch.ones((ROWS, COLS), device='cuda', dtype=torch.bfloat16)
weights = [bits.view(torch.bfloat16)] + [ones] * 31
with torch.inference_mode():
    cache = EncodedGUCache(weights)
assert cache.statistics()['active_layers'] == 32
table_pointer = cache.table.data_ptr()
out = torch.empty((ROWS, COLS), device='cuda', dtype=torch.bfloat16)
library = ctypes.CDLL(str(HERE / 'decode_words.so'))
library.decode_all.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int]
library.decode_all.restype = ctypes.c_int


def verify():
    layer = cache._tables[0]
    rc = library.decode_all(torch.cuda.current_stream().cuda_stream,
                            layer['plane'].data_ptr(), layer['headers'].data_ptr(),
                            layer['raw'].data_ptr(), out.data_ptr(), CHUNKS)
    assert rc == 0, rc
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint16), bits)


verify()
bits.fill_(0xC123)
cache.refresh(weights)
verify()
bits.copy_(original)
cache.refresh(weights)
verify()
result = {'passed': True, 'unique_bf16_patterns': 65536,
          'words_checked_per_pass': ROWS * COLS, 'passes': 3,
          'cache_created_in_inference_mode': True, 'refresh_called_outside_inference_mode': True,
          'stable_descriptor': cache.table.data_ptr() == table_pointer,
          'helper_source_sha256': hashlib.sha256((HERE / 'decode_words.cu').read_bytes()).hexdigest(),
          'cache': cache.statistics()}
assert result['stable_descriptor']
(HERE / 'decode-words-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'cache'}))
