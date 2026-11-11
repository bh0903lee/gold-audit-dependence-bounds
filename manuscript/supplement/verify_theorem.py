"""Independent verification: L=3 gluing construction and exact rational L=4 LP certificates."""
import itertools, json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta, binom

OUT = Path(__file__).resolve().parent

def setup(L, method):
    w = np.array(list(itertools.product(range(3), repeat=L)))
    x, y = (w > 0).astype(int), (w == 2).astype(int)
    A = [np.ones(len(w), dtype=int)]
    for v in itertools.product(range(3), repeat=L-1):
        A.append(np.all(w[:, :-1] == v, axis=1).astype(int)
                 - np.all(w[:, 1:] == v, axis=1).astype(int))
    obs = [np.all(y == v, axis=1).astype(int)
           for v in itertools.product(range(2), repeat=L)]
    A.extend(obs)
    if method == 'neighbor':
        A.extend([x[:, 1], x[:, 1]*y[:, 0], x[:, 1]*y[:, 2]])
    else:
        A.extend([x[:, 1]*v for v in obs])
    # Under stationarity the first-pair objective equals the objective averaged over adjacent pairs.
    return w, np.array(A), (2*x[:, 0]-1)*(2*x[:, 1]-1)

def law(sequences, weights, L):
    p = [Q(0) for _ in range(3**L)]
    for seq, weight in zip(sequences, weights):
        for i in range(len(seq)):
            ix = sum(seq[(i+j) % len(seq)] * 3**(L-1-j) for j in range(L))
            p[ix] += weight/len(seq)
    return p

def exact_certificate(A, b, c, sign):
    fit = linprog(sign*c, A_eq=A, b_eq=np.array(b, float),
                  bounds=(0, None), method='highs')
    assert fit.success
    p = [Q(float(v)).limit_denominator(1000000) for v in fit.x]
    d = [Q(float(v)).limit_denominator(1000000) for v in fit.eqlin.marginals]
    assert min(p) >= 0
    assert all(sum(int(a)*v for a, v in zip(row, p)) == rhs for row, rhs in zip(A, b))
    assert all(sum(int(a)*v for a, v in zip(col, d)) <= sign*int(ci)
               for col, ci in zip(A.T, c))
    primal = sum(sign*int(ci)*v for ci, v in zip(c, p))
    dual = sum(bi*di for bi, di in zip(b, d))
    assert primal == dual
    return dict(endpoint=str(sign*primal), primal=list(map(str, p)), dual=list(map(str, d)))

seqs = [[2,2,0,1,1,1,2,1,1,2,1,0],
        [2,2,2,2,1,2,1,2,0,0,2,2], [2,0,0,0,1]]
p = law(seqs, [Q(1,2), Q(1,4), Q(1,4)], 4)
certs = {}
for method in ['neighbor', 'full']:
    w, A, c = setup(4, method)
    b = [sum(int(a)*v for a, v in zip(row, p)) for row in A]
    certs[method] = {name:exact_certificate(A,b,c,sign)
                    for name,sign in [('lower',1), ('upper',-1)]}
assert Q(certs['full']['lower']['endpoint']) > Q(certs['neighbor']['lower']['endpoint'])

rng = np.random.default_rng(20261007)
max_glue_error = max_lp_error = 0.
for case in range(200):
    sequences = [rng.integers(0,3,size=int(rng.integers(2,20))).tolist() for _ in range(5)]
    weights = [Q(int(k),15) for k in range(1,6)]
    p = np.array(law(sequences, weights, 3), float).reshape(3,3,3)
    w,A,c = setup(3,'full')
    x,y = (w>0), (w==2)
    flat=p.ravel()
    m,ybar,a,b,h = [float(v@flat) for v in [x[:,1],y[:,1],y[:,0]*x[:,1],x[:,1]*y[:,2],y[:,0]*y[:,1]]]
    lo=max(0.,2*m-1-a-b+h);hi=min(m-ybar-a+h,m-ybar-b+h)
    formula=[max(1-4*m+4*a+4*b-4*h,4*m-3),1-4*ybar+4*min(a,b)]
    H=np.zeros((2,3,2))
    for i,j,k in itertools.product(range(3),repeat=3): H[int(i==2),j,int(k==2)]+=p[i,j,k]
    for t in [lo,(lo+hi)/2,hi]:
        K=np.array([[1-2*m+a+b-h+t,m-ybar-a+h-t,ybar-b],
                    [m-ybar-b+h-t,t,b-h],[ybar-a,a-h,h]])
        assert K.min() > -1e-12
        K=np.maximum(K,0)
        T=np.zeros((3,3,3))
        for i,j,k in itertools.product(range(3),repeat=3):
            il=[z for z in range(3) if (z==2)==(i==2)]
            kr=[z for z in range(3) if (z==2)==(k==2)]
            dl=K[il,j].sum();dr=K[j,kr].sum()
            if dl>1e-14 and dr>1e-14:
                T[i,j,k]=H[int(i==2),j,int(k==2)]*K[i,j]/dl*K[j,k]/dr
        HT=np.zeros_like(H)
        for i,j,k in itertools.product(range(3),repeat=3): HT[int(i==2),j,int(k==2)]+=T[i,j,k]
        max_glue_error=max(max_glue_error,float(abs(T.sum(0)-K).max()),float(abs(T.sum(2)-K).max()),float(abs(HT-H).max()))
    for method in ['neighbor','full']:
        _,AA,cc=setup(3,method)
        for sign,expected in [(1,formula[0]),(-1,formula[1])]:
            fit=linprog(sign*cc,A_eq=AA,b_eq=AA@flat,bounds=(0,None),method='highs')
            assert fit.success
            max_lp_error=max(max_lp_error,abs(float(cc@fit.x)-expected))
assert max_glue_error<1e-12 and max_lp_error<1e-10

# Enumerate a binomial experiment, using both ends of the SAME confidence interval.
min_coverage=1.
for n in [1,2,5,10,50,300]:
    k=np.arange(n+1);low=np.zeros(n+1);high=np.ones(n+1)
    low[1:]=beta.ppf(.025,k[1:],n-k[1:]+1)
    high[:-1]=beta.ppf(.975,k[:-1]+1,n-k[:-1])
    for prob in np.linspace(0,1,1001):
        coverage=float(binom.pmf(k,n,prob)[(low<=prob)&(prob<=high)].sum())
        min_coverage=min(min_coverage,coverage)
assert min_coverage>=.95-1e-12
result=dict(L3_laws=200,L3_gluings=600,max_glue_error=max_glue_error,max_lp_error=max_lp_error,
            cp_grid_min_coverage=min_coverage,L4_sequences=seqs,L4_weights=['1/2','1/4','1/4'],L4_certificates=certs)
(OUT/'verify_theorem.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='L4_certificates'})
print({m:{k:v['endpoint'] for k,v in cs.items()} for m,cs in certs.items()})
