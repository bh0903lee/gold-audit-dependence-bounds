"""Reconstruct rational constraints and verify saved witnesses without experiment imports."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import json,numpy as np
O=Path(__file__).parent;R=O  # certificates are read from this directory
Z=[(0,0),(1,0),(1,1),(0,1)];W=list(product(range(4),repeat=3));X=np.array([[Z[i][0] for i in w] for w in W]);Y=np.array([[Z[i][1] for i in w] for w in W]);c=(2*X[:,0]-1)*(2*X[:,1]-1)
flow=np.array([[int(w[:2]==a)-int(w[1:]==a) for w in W] for a in product(range(4),repeat=2)],dtype=object)
obs=Y@np.array([4,2,1]);f=(1-X[:,1])*Y[:,1]
obsrows=np.array([obs==j for j in range(8)],int)
def law(seed):
 rng=np.random.default_rng(seed);sizes=rng.integers(3,9,size=3);cycles=[rng.integers(0,4,size=int(n)) for n in sizes];weights=rng.integers(1,6,size=3);out={w:Q(0) for w in W}
 for cy,wt in zip(cycles,weights):
  for i in range(len(cy)):out[tuple(int(cy[(i+k)%len(cy)]) for k in range(3))]+=Q(int(wt),int(sum(weights))*len(cy))
 return np.array([out[w] for w in W],dtype=object)
def check(cert,ae,be,au,bu,obj):
 p=np.array([Q(x) for x in cert['primal']],dtype=object);de=np.array([Q(x) for x in cert['dual_equality']],dtype=object);du=np.array([Q(x) for x in cert['dual_inequality']],dtype=object);sg=cert['sign']
 return {'primal_nonnegative':bool(all(p>=0)),'equalities':bool(all(ae@p==be)),'inequalities':bool(all(au@p<=bu)),'dual_sign':bool(all(du<=0)),'dual_slack':bool(all(sg*obj-ae.T@de-au.T@du>=0)),'zero_duality_gap':bool(sg*(obj@p)==be@de+bu@du),'reported_objective':bool(obj@p==Q(cert['objective']))}
gap=json.loads((R/'fp_gap_certificate.json').read_text());p=law(gap['seed']);results={}
for info,certs in gap['certificates'].items():
 extra=np.array([(8*X[:,1]+obs)==j for j in range(16)],int) if info=='F' else np.array([X[:,1],Y[:,0]*X[:,1],X[:,1]*Y[:,2],X[:,1]*Y[:,1]])
 ae=np.vstack([np.ones(64,dtype=int),flow,obsrows,extra]);be=np.r_[Q(1),np.zeros(16,dtype=int),obsrows@p,extra@p];au=f[None,:];bu=np.array([Q(gap['epsilon'])],dtype=object)
 results[info]=[check(t,ae,be,au,bu,c) for t in certs]
glue=json.loads((R/'fp_gluing_certificate.json').read_text());p=law(glue['seed']);h=np.zeros((2,4,2),dtype=object)
for w,v in zip(W,p):h[Z[w[0]][1],w[1],Z[w[2]][1]]+=v
hl=h.sum(2);hr=h.sum(0);ae=[np.ones(16,dtype=int)];be=[Q(1)]
for z in range(4):
 row=np.zeros((4,4),int);row[z,:]+=1;row[:,z]-=1;ae.append(row.ravel());be.append(Q(0))
 for y in range(2):
  left=np.zeros((4,4),int);right=left.copy()
  for i in range(4):
   if Z[i][1]==y:left[i,z]=1;right[z,i]=1
  ae.extend([left.ravel(),right.ravel()]);be.extend([hl[y,z],hr[z,y]])
pc=np.array([(2*Z[i][0]-1)*(2*Z[j][0]-1) for i,j in product(range(4),repeat=2)])
gchecks=[]
for cert in glue['certificates']:
 ck=check(cert,np.array(ae,dtype=object),np.array(be,dtype=object),np.zeros((1,16),int),np.array([Q(0)]),pc)
 pair=np.array([Q(v) for v in cert['primal']],dtype=object).reshape(4,4);triple=np.array([Q(v) for v in cert['triple']],dtype=object).reshape(4,4,4);hh=np.zeros((2,4,2),dtype=object)
 for w,v in zip(W,triple.ravel()):hh[Z[w[0]][1],w[1],Z[w[2]][1]]+=v
 ck.update(triple_nonnegative=bool(all(triple.ravel()>=0)),triple_normalized=bool(sum(triple.ravel())==1),left_pair=bool(np.array_equal(triple.sum(2),pair)),right_pair=bool(np.array_equal(triple.sum(0),pair)),full_audit=bool(np.array_equal(hh,h)),reported_h=bool(np.array_equal(h.ravel(),[Q(v) for v in cert['full_h']])),target=bool(c@triple.ravel()==Q(cert['objective'])))
 gchecks.append(ck)
results['gluing']=gchecks;results['all_checks_pass']=all(all(c.values()) for k,v in results.items() if k!='all_checks_pass' for c in v)
(O/'fp_certificate_verification.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if not results['all_checks_pass']:raise SystemExit('FP certificate verification failed')
