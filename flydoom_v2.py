"""
flydoom_v2.py - Drosophila hemibrain connectome plays DOOM (defend_the_center).
v2 change: real neurotransmitter-derived synapse signs (was pseudo-random in v1).

DATA
  connectome : hemibrain v1.2 traced adjacencies, 21,739 neurons / 3.55M synapses
               https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz
  transmitters: Eckstein & Bates et al. 2024, Zenodo 10593546, supplemental_data_3.csv
               99.6% bodyId coverage -> 36% of neurons inhibitory
               ACh=+1, GABA=-1, Glutamate=-1 (GluCl), amines=+0.25

MODEL  LIF over the real adjacency matrix, hemibrain mirrored into two hemispheres
       (it is a single hemisphere, so unmirrored there is no left/right steering),
       reciprocal inhibition between the DNa01/DNa02 pools, spike-frequency
       adaptation + homeostatic gain control (a raw wiring diagram is bistable:
       silent or seizing). Operating point 3.6% population firing.

I/O    in  LC4/LPLC2/LC6/LPLC1 looming, LC11/12/18 small object, LC10 tracking,
           LC9/16/17 wide-field  <- 8 retinotopic columns of the Doom framebuffer
       out DNa01/DNa02 graded membrane potential -> yaw (z>0.8)
           DNp02/04/06/09 escape DNs -> ATTACK (z>0.7)
           MDN -> backward (unused in this scenario)

RESULT (n=20 seeds, kills/episode)
       connectome 1.15 | random 1.45 | idle 0.00 | Mann-Whitney p=0.19 vs random
       i.e. clearly beats doing nothing, statistically indistinguishable from
       a coin flip. v1 (fake signs) scored 0.60, so real transmitters nearly
       doubled it, but not past chance.

STILL NOT BIOLOGY: LC receptive fields are assigned to columns by array index,
       not real anatomy. No optic lobe (hemibrain lacks it) - the whole early
       visual system is replaced by 8 handmade feature channels. No learning.
"""

import os, tarfile, urllib.request, numpy as np, pandas as pd, scipy.sparse as sp, vizdoom as vzd
SC=os.path.join(os.path.dirname(vzd.__file__),"scenarios"); D="./fly"
HB="https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
NT="https://zenodo.org/api/records/10593546/files/supplemental_data_3.csv/content"
SIGN={"acetylcholine":1.,"gaba":-1.,"glutamate":-1.,"dopamine":.25,"serotonin":.25,"octopamine":.25}

def build():
    os.makedirs(D,exist_ok=True)
    if not os.path.exists(D+"/hb.tar.gz"): urllib.request.urlretrieve(HB,D+"/hb.tar.gz")
    if not os.path.exists(D+"/exported-traced-adjacencies-v1.2"):
        with tarfile.open(D+"/hb.tar.gz") as t: t.extractall(D)
    if not os.path.exists(D+"/nt.csv"): urllib.request.urlretrieve(NT,D+"/nt.csv")
    B=D+"/exported-traced-adjacencies-v1.2/"
    nrn=pd.read_csv(B+"traced-neurons.csv"); con=pd.read_csv(B+"traced-total-connections.csv")
    nt=pd.read_csv(D+"/nt.csv"); m=dict(zip(nt.bodyid,nt.conf_nt))
    sign=nrn.bodyId.map(lambda b: SIGN.get(m.get(b),1.)).values.astype(np.float32)
    idx={b:i for i,b in enumerate(nrn.bodyId.values)}; N=len(idx)
    c=con[con.bodyId_pre.isin(idx)&con.bodyId_post.isin(idx)]
    pre=c.bodyId_pre.map(idx).values; post=c.bodyId_post.map(idx).values
    W=sp.csr_matrix((c.weight.values.astype(np.float32)*sign[pre],(pre,post)),shape=(N,N))
    W=W.multiply(1/np.maximum(abs(W).sum(0),1e-6)).tocsr()
    t=nrn["type"].fillna(""); G=lambda p: np.where(t.str.match(p).values)[0]
    IN={"loom":G(r"^(LPLC2|LC4|LC6|LPLC1)$"),"smallobj":G(r"^(LC11|LC18|LC12)$"),
        "track":G(r"^LC10"),"wide":G(r"^(LC9|LC17|LC16)$")}
    dn=G(r"^(DNa02|DNa01)$"); atk=G(r"^(DNp09|DNp02|DNp04|DNp06)$")
    Wb=sp.block_diag([W,W],format="csr"); NB=2*N
    OUT={"turn_L":dn,"turn_R":dn+N,"attack":np.r_[atk,atk+N],"back":np.r_[G(r"^MDN"),G(r"^MDN")+N]}
    XI=sp.csr_matrix((np.full(len(dn)*2,-.35,np.float32),(np.r_[dn,dn+N],np.r_[dn+N,dn])),shape=(NB,NB))
    return (Wb+XI).tocsr(), NB, {k:np.r_[v,v+N] for k,v in IN.items()}, OUT

class Fly:
    def __init__(s,W,n,IN,OUT,leak=.82,th=.25,target=.03,adapt=.9):
        s.W,s.n,s.IN,s.OUT,s.leak,s.th,s.target,s.adapt=W,n,IN,OUT,leak,th,target,adapt
        s.v=np.zeros(n,np.float32);s.sp=np.zeros(n,np.float32);s.a=np.zeros(n,np.float32);s.g=1.
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

def play(W,NB,IN,OUT,seed=1,zt=.8,za=.7,ticks=1200,render=False):
    b=Fly(W*2.0,NB,IN,OUT)
    g=vzd.DoomGame(); g.load_config(os.path.join(SC,"defend_the_center.cfg"))
    g.set_window_visible(render); g.set_screen_format(vzd.ScreenFormat.GRAY8)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120); g.set_seed(seed); g.init(); g.new_episode()
    prev=None; t=0
    while not g.is_episode_finished() and t<ticks:
        fr=g.get_state().screen_buffer; b.step(encode(fr,prev,b)); prev=fr
        z1,z2=b.read(); g.make_action([bool(z1>zt),bool(z1<-zt),bool(z2>za)],2); t+=1
    k=int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)); g.close(); return k

if __name__=="__main__":
    W,NB,IN,OUT=build()
    ks=[play(W,NB,IN,OUT,seed=s) for s in range(1,21)]
    print("kills/episode over 20 seeds:", ks, "mean", sum(ks)/len(ks))
