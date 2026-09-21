"""
flydoom_v3.py - Full-brain FlyWire (FAFB) connectome plays DOOM.
v3 change: swapped Janelia hemibrain (one hemisphere, no optic lobe, mirrored/faked)
for the full FlyWire FAFB brain (both hemispheres native, real neuron anatomy).

DATA: SLOP011/flywire-fafb-connectome on Hugging Face (public, no auth needed) --
      a repackaging of FlyWire FAFB v783: 139,255 neurons, 15,091,983 synaptic
      connections, per-synapse neurotransmitter probability scores (ach/gaba/glut/da/ser/oct).
      https://huggingface.co/datasets/SLOP011/flywire-fafb-connectome
      Underlying papers: Dorkenwald et al. 2023 (wiring diagram),
      Schlegel et al. 2023 (cell typing), Eckstein/Bates 2024 (neurotransmitters).

WHAT'S ACTUALLY FIXED vs v2 (hemibrain):
  - No more mirroring hack -- 'side' (left/right) is real anatomical metadata.
  - No fabricated retinotopy needed at the anatomy level (though LC->column
    assignment is still index-based, see caveats below -- FlyWire's optic lobe
    proper wasn't traversed here, only the LC/LPLC relay neurons by hemibrain_type).
  - Synapse sign is now a *continuous* score from per-synapse NT probabilities
    (ach + 0.25*(da+ser+oct)) - (gaba+glut), not a single discrete label per neuron.

RESULT (n=20 seeds, kills/episode, defend_the_center)
  flywire full-brain      1.25  <- new
  hemibrain v2 (real NT)  1.15
  random actions          1.45
  idle                    0.00
  flywire vs random:    Mann-Whitney p=0.47
  flywire vs hemibrain: Mann-Whitney p=0.45

Verdict: full brain modestly beat hemibrain (1.25 vs 1.15) but the gap is not
statistically distinguishable at n=20, and it's still not statistically
distinguishable from random action selection. More anatomy did not buy a
significant win. That is a real result, not a bug -- see caveats.

STILL NOT BIOLOGY:
  - LC receptive fields are still assigned to 8 columns by array index, not by
    the optic lobe's actual retinotopic map -- the visual periphery (lamina,
    medulla, lobula columns) was not traversed, only the LC/LPLC projection
    neurons matched by hemibrain_type name.
  - Same LIF + adaptation + homeostatic gain control scaffolding as v2, applied
    unchanged to a 6.4x larger brain -- these constants were tuned on hemibrain
    and not re-tuned here.
  - No learning anywhere.
  - Firing rate settled around 1.0% (vs 3.6% on hemibrain) with identical target;
    the full brain's extra inhibitory circuitry changes the network's effective
    gain, and this was not re-optimized.
"""


import pyarrow.parquet as pq, pandas as pd, numpy as np, scipy.sparse as sp, pickle, gc

nodes=pd.read_parquet("/home/user/flywire/nodes.parquet",
    columns=["root_id","hemibrain_type","side"])
t=nodes["hemibrain_type"].fillna(""); side=nodes["side"].fillna("")
idx={r:i for i,r in enumerate(nodes.root_id.values)}; N=len(idx)
G=lambda pat,sd=None: np.where((t.str.match(pat).values) & ((side.values==sd) if sd else True))[0]
IN={"loom":G(r"^(LPLC2|LC4|LC6|LPLC1)"),"smallobj":G(r"^(LC11|LC18|LC12)"),
    "track":G(r"^LC10"),"wide":G(r"^(LC9|LC17|LC16)")}
dnL=G(r"^(DNa02|DNa01)","left"); dnR=G(r"^(DNa02|DNa01)","right")
atk=G(r"^(DNp09|DNp02|DNp04|DNp06)"); back=G(r"^MDN")
print({k:len(v) for k,v in IN.items()}, "dnL",len(dnL),"dnR",len(dnR),"atk",len(atk),"back",len(back))
del nodes; gc.collect()

pf=pq.ParquetFile("/home/user/flywire/connections.parquet")
print("row groups:",pf.num_row_groups,"total rows:",pf.metadata.num_rows)

rows_pre=[]; rows_post=[]; rows_w=[]
cols=["pre","post","syn_count","ach","gaba","glut","da","ser","oct"]
for i in range(pf.num_row_groups):
    b=pf.read_row_group(i, columns=cols).to_pandas()
    pre=b.pre.map(idx); post=b.post.map(idx)
    m=pre.notna().values & post.notna().values
    if not m.any(): continue
    p=pre.values[m].astype(np.int32); q=post.values[m].astype(np.int32)
    sign=((b.ach.values[m]+0.25*(b.da.values[m]+b.ser.values[m]+b.oct.values[m]))
          -(b.gaba.values[m]+b.glut.values[m]))
    w=(b.syn_count.values[m].astype(np.float32))*sign.astype(np.float32)
    rows_pre.append(p); rows_post.append(q); rows_w.append(w)
    if i%50==0: print("rg",i,"/",pf.num_row_groups)
    del b; gc.collect()

pre=np.concatenate(rows_pre); post=np.concatenate(rows_post); w=np.concatenate(rows_w)
del rows_pre,rows_post,rows_w; gc.collect()
W=sp.csr_matrix((w,(pre,post)),shape=(N,N))
del pre,post,w; gc.collect()
colsum=np.asarray(np.abs(W).sum(0)).ravel()
W=W.multiply(1/np.maximum(colsum,1e-6)).tocsr()
print("N",N,"nnz",W.nnz)
sp.save_npz("/home/user/flywire/W.npz", W)
with open("/home/user/flywire/net.pkl","wb") as f:
    pickle.dump(dict(N=N,IN=IN,dnL=dnL,dnR=dnR,atk=atk,back=back),f)
print("saved")


# --- inference/harness ---

import pickle, numpy as np, scipy.sparse as sp, vizdoom as vzd, os, json, sys
with open("/home/user/flywire/net.pkl","rb") as f: net=pickle.load(f)
W=sp.load_npz("/home/user/flywire/W.npz")
N=net["N"]; IN=net["IN"]
OUT={"turn_L":net["dnL"],"turn_R":net["dnR"],"attack":net["atk"],"back":net["back"]}
SC=os.path.join(os.path.dirname(vzd.__file__),"scenarios")

class Fly:
    def __init__(s,W,n,IN,OUT,leak=.82,th=.25,target=.03,adapt=.9,wgain=2.0):
        s.W=W*wgain; s.n=n; s.IN=IN; s.OUT=OUT; s.leak=leak; s.th=th; s.target=target; s.adapt=adapt
        s.v=np.zeros(n,np.float32); s.sp=np.zeros(n,np.float32); s.a=np.zeros(n,np.float32); s.g=1.
        s.cols={k:np.arange(len(v))%8 for k,v in IN.items()}; s.h={"t":[],"a":[]}
    def step(s,d,k=4):
        for _ in range(k):
            s.v=s.leak*s.v+s.g*s.W.T.dot(s.sp)+d-s.a
            s.sp=(s.v>s.th).astype(np.float32); s.v*=(1-s.sp); s.v=np.clip(s.v,-3,5)
            s.a=.92*s.a+s.adapt*s.sp
            s.g=float(np.clip(s.g+.05*(s.target-float(s.sp.mean())),.05,3.))
    def read(s):
        tn=float(s.v[s.OUT["turn_L"]].mean()-s.v[s.OUT["turn_R"]].mean())
        ak=float(s.v[s.OUT["attack"]].mean())
        s.h["t"].append(tn); s.h["a"].append(ak)
        z=lambda k,x: 0. if len(s.h[k])<10 else (x-np.mean(s.h[k][-90:]))/(np.std(s.h[k][-90:])+1e-6)
        return z("t",tn), z("a",ak)

def encode(f,prev,b,gain=45.):
    d=np.zeros(b.n,np.float32); f=f.astype(np.float32)/255.
    cs=np.array_split(f,8,axis=1); lum=np.array([c.mean() for c in cs]); cn=np.array([c.std() for c in cs])
    lo=np.zeros(8,np.float32)
    if prev is not None:
        pc=np.array_split(prev.astype(np.float32)/255.,8,axis=1)
        lo=np.array([max(0,c.std()-p.std()) for c,p in zip(cs,pc)],np.float32)*6
    for k,g_,s_ in (("loom",2.2,lo),("smallobj",1.1,cn),("track",.9,cn),("wide",.5,lum)):
        d[b.IN[k]]=g_*s_[b.cols[k]]
    return d*gain

def mkgame(seed):
    g=vzd.DoomGame(); g.load_config(os.path.join(SC,"defend_the_center.cfg"))
    g.set_window_visible(False); g.set_screen_format(vzd.ScreenFormat.GRAY8)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120); g.set_seed(seed); g.init(); return g

def run(seed,zt=.8,za=.7,ticks=1200):
    b=Fly(W,N,IN,OUT); g=mkgame(seed); g.new_episode(); prev=None; t=0
    while not g.is_episode_finished() and t<ticks:
        fr=g.get_state().screen_buffer; b.step(encode(fr,prev,b)); prev=fr
        z1,z2=b.read(); g.make_action([bool(z1>zt),bool(z1<-zt),bool(z2>za)],2); t+=1
    k=int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)); g.close(); return k

start,end=int(sys.argv[1]),int(sys.argv[2])
out={}
for s in range(start,end):
    out[s]=run(s)
    print(s,out[s],flush=True)
path=f"/home/user/flywire/res_{start}_{end}.json"
json.dump(out, open(path,"w"))
