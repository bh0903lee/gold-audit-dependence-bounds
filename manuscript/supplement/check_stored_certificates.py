"""Verify the four stored exact LP certificates using only Python's standard library."""
import itertools,json
from fractions import Fraction as Q
from pathlib import Path
z=json.loads((Path(__file__).parent/'verify_theorem.json').read_text())
print('Keys:',list(z))
words=list(itertools.product(range(3),repeat=4))
seqs=[[int(c) for c in s] for s in ['220111211210','222212120022','20001']]
p=[]
for w in words:
 p.append(sum(weight*sum(tuple(seq[(i+j)%len(seq)] for j in range(4))==w for i in range(len(seq)))/len(seq) for seq,weight in zip(seqs,[Q(1,2),Q(1,4),Q(1,4)])))
for method in ['neighbor','full']:
 A=[[1]*len(words)]
 A.extend([[int(w[:-1]==u)-int(w[1:]==u) for w in words] for u in itertools.product(range(3),repeat=3)])
 obs=[[int(tuple(int(v==2) for v in w)==u) for w in words] for u in itertools.product(range(2),repeat=4)]
 A.extend(obs)
 x=[int(w[1]>0) for w in words]
 A.extend([x,[xi*int(w[0]==2) for xi,w in zip(x,words)],[xi*int(w[2]==2) for xi,w in zip(x,words)]] if method=='neighbor' else [[xi*o for xi,o in zip(x,row)] for row in obs])
 b=[sum(ai*pi for ai,pi in zip(row,p)) for row in A]
 c=[(2*int(w[0]>0)-1)*(2*int(w[1]>0)-1) for w in words]
 for name,sign,expected in [('lower',1,Q(-1,10) if method=='neighbor' else Q(2,15)),('upper',-1,Q(11,20))]:
  cert=z['L4_certificates'][method][name]
  primal=list(map(Q,cert['primal']));dual=list(map(Q,cert['dual']))
  assert len(primal)==81 and len(dual)==len(A)
  assert min(primal)>=0
  assert all(sum(ai*pi for ai,pi in zip(row,primal))==bi for row,bi in zip(A,b))
  assert all(sum(A[i][j]*dual[i] for i in range(len(A)))<=sign*c[j] for j in range(81))
  pv=sum(sign*ci*pi for ci,pi in zip(c,primal));dv=sum(bi*di for bi,di in zip(b,dual))
  assert pv==dv and sign*pv==expected and Q(cert['endpoint'])==expected
  print(method,name,expected,'PASS')
