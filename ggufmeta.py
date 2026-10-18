#!/usr/bin/env python3
# ggufmeta.py — print the architecture keys, parameter count and bits per weight of GGUF files by parsing the
# header by hand (gguf-dump rejects files with tensor types it does not know, e.g. ternary Q2_0 = type 42).
# Usage: python3 -I llm/ggufmeta.py FILE.gguf...
import struct, sys, os
T = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
S = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def rd(f, fmt): n=struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
def rstr(f): n=rd(f,'<Q'); return f.read(n).decode('utf-8','replace')
def rval(f, t):
    if t==8: return rstr(f)
    if t==9:
        et=rd(f,'<I'); n=rd(f,'<Q')
        if et in (8,9):
            for _ in range(n): rval(f, et)
            return f'[{n} x {T[et]}]'
        sz=struct.calcsize(S[et]); f.seek(sz*n,1); return f'[{n} x {T[et]}]'
    return rd(f,S[t])
WANT=('block_count','embedding_length','feed_forward_length','head_count','head_count_kv','key_length','expert_count','expert_used_count','expert_feed_forward_length','expert_shared_feed_forward_length','context_length','full_attention_interval','ssm.inner_size','ssm.state_size','ssm.group_count','ssm.time_step_rank','nextn_predict_layers','general.architecture','general.name','general.file_type','tokenizer.ggml.tokens')
for p in sys.argv[1:]:
    f=open(p,'rb'); assert f.read(4)==b'GGUF'; ver=rd(f,'<I'); nt=rd(f,'<Q'); nkv=rd(f,'<Q')
    print('#####', os.path.basename(p), 'size', os.path.getsize(p))
    for _ in range(nkv):
        k=rstr(f); t=rd(f,'<I'); v=rval(f,t)
        if any(w in k for w in WANT): print(' ', k, '=', v if not isinstance(v,list) else f'[{len(v)}]')
    tot=0; types={}
    for _ in range(nt):
        name=rstr(f); nd=rd(f,'<I'); dims=[rd(f,'<Q') for _ in range(nd)]; ty=rd(f,'<I'); off=rd(f,'<Q')
        n=1
        for d in dims: n*=d
        tot+=n; types[ty]=types.get(ty,0)+n
    print('  tensors', nt, 'params %.2fB'%(tot/1e9), 'bpw %.2f'%(os.path.getsize(p)*8/tot), 'types', dict(sorted(types.items(), key=lambda x:-x[1])[:4]))
