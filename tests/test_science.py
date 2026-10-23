import sys, importlib, itertools, json, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from topic_experiment.common import kl,rk4
def engine(name):
    return importlib.import_module('topic_experiment.engine')
class Science(unittest.TestCase):
    def test_finite_sample_coverage_and_full_observation(self):
        m=engine('C11'); rng=np.random.default_rng(12)
        covered=0
        for stay in [.2,.5,.8]:
            x,y=m.sample_episodes(rng,6000,3,stay,[.95,.2]); ax,ay=m.sample_episodes(rng,3000,3,stay,[.95,.2])
            lo,hi=m.bounds(y,ax[:,1],ay); covered+=lo<=2*stay-1<=hi
            self.assertTrue(-1-1e-8<=lo<=hi<=1+1e-8)
        self.assertEqual(covered,3)
        x,y=m.sample_episodes(rng,30000,3,.8,[1,1]); lo,hi=m.bounds(y,x[:,1],y)
        self.assertLess(hi-lo,.25)

    def test_audit_does_not_observe_target_pair(self):
        m=engine('C11'); x,y=m.sample_episodes(np.random.default_rng(4),10000,3,.8,[.95,.2])
        self.assertTrue(np.all(y<=x))
        self.assertAlmostEqual(np.mean(x),.5,delta=.02)
        self.assertAlmostEqual(np.mean(x[:,0]==x[:,1]),.8,delta=.02)

    def test_population_bounds_nest_and_match_independent_values(self):
        m=engine('C11')
        # Reference values from an independent enumeration of the population identified set for L=3, detect [.95,.2].
        lo,hi=m.population_bounds(.2,[.95,.2],3,'marginal'); self.assertAlmostEqual(lo,-0.847,places=2); self.assertAlmostEqual(hi,-0.136,places=2)
        lo,hi=m.population_bounds(.2,[.95,.2],3,'neighbor'); self.assertAlmostEqual(lo,-0.664,places=2); self.assertAlmostEqual(hi,-0.520,places=2)
        for stay in [.2,.5,.8]:
            w={k:m.population_bounds(stay,[.95,.2],3,k) for k in ['marginal','neighbor','full']}
            for k,(lo,hi) in w.items(): self.assertTrue(lo-1e-6<=2*stay-1<=hi+1e-6)
            self.assertLessEqual(w['full'][1]-w['full'][0],w['neighbor'][1]-w['neighbor'][0]+1e-9)
            self.assertLessEqual(w['neighbor'][1]-w['neighbor'][0],w['marginal'][1]-w['marginal'][0]+1e-9)
        self.assertAlmostEqual(m.population_law(.5,[.95,.2],4).sum(),1.)

    def test_bernstein_tighter_for_rare_events_and_valid(self):
        m=engine('C11'); rng=np.random.default_rng(5)
        rare=(rng.random(5000)<.02).astype(float)
        self.assertLess(m.radius(rare,.005,'empirical_bernstein'),m.radius(rare,.005,'hoeffding'))
        half=(rng.random(5000)<.5).astype(float)
        self.assertGreater(m.radius(half,.005,'empirical_bernstein'),0.)
        # Empirical coverage of the Bernstein radius on a Bernoulli mean (one-feature check).
        hits=0
        for _ in range(400):
            v=(rng.random(500)<.1).astype(float); hits+=abs(v.mean()-.1)<=m.radius(v,.05,'empirical_bernstein')
        self.assertGreaterEqual(hits,380)
        x,y=m.sample_episodes(rng,4000,3,.5,[.95,.2]); ax,ay=m.sample_episodes(rng,500,3,.5,[.95,.2])
        for moments in ['marginal','neighbor','full']:
            for interval in ['hoeffding','empirical_bernstein']:
                lo,hi=m.bounds(y,ax[:,1],ay,moments=moments,interval=interval); self.assertTrue(-1-1e-8<=lo<=hi<=1+1e-8)
        a=m.bounds(y,ax[:,1],ay,cross=True); b=m.bounds(y,ax[:,1],ay,moments='neighbor'); self.assertEqual(a,b)

    def test_exact_binomial_and_decomposition(self):
        m=engine('C11'); rng=np.random.default_rng(9)
        v=(rng.random(400)<.1).astype(float); lo,hi=m.confidence_limits(v,.01,'exact_binomial')
        self.assertTrue(lo<=v.mean()<=hi); self.assertLess(hi-lo,1.)
        # Exact coverage of a single Clopper-Pearson interval, computed by summing binomial probabilities.
        from scipy.stats import binom
        for N in [1,5,50,300]:
            for p in np.linspace(.01,.99,41):
                cov=sum(binom.pmf(k,N,p) for k in range(N+1) if (lambda lh:lh[0]<=p<=lh[1])(m.confidence_limits(np.r_[np.ones(k),np.zeros(N-k)],.05,'exact_binomial')))
                self.assertGreaterEqual(cov,.95-1e-9)
        x,y=m.sample_episodes(rng,3000,3,.2,[.95,.2]); ax,ay=m.sample_episodes(rng,300,3,.2,[.95,.2])
        pobs,paud=m.population_moments(.2,[.95,.2],3,'neighbor'); plo,phi=m.population_bounds(.2,[.95,.2],3,'neighbor')
        both=m.bounds(y,ax[:,1],ay,moments='neighbor',interval='hoeffding'); eo=m.bounds(y,ax[:,1],ay,moments='neighbor',interval='hoeffding',exact_obs=pobs); ea=m.bounds(y,ax[:,1],ay,moments='neighbor',interval='hoeffding',exact_audit=paud)
        for lo,hi in [both,eo,ea]: self.assertLessEqual(lo,plo+1e-6); self.assertGreaterEqual(hi,phi-1e-6)
        self.assertLessEqual(eo[1]-eo[0],both[1]-both[0]+1e-9); self.assertLessEqual(ea[1]-ea[0],both[1]-both[0]+1e-9)
        exact=m.bounds(y,ax[:,1],ay,moments='neighbor',interval='hoeffding',exact_obs=pobs,exact_audit=paud)
        self.assertAlmostEqual(exact[0],plo,places=5); self.assertAlmostEqual(exact[1],phi,places=5)

    def test_bernstein_is_two_sided(self):
        m=engine('C11'); v=np.r_[np.ones(30),np.zeros(70)]
        ln=np.log(4/.05); expected=np.sqrt(2*np.var(v,ddof=1)*ln/100)+7*ln/(3*99)
        self.assertAlmostEqual(m.radius(v,.05,'empirical_bernstein'),expected,places=12)

if __name__=="__main__": unittest.main(verbosity=2)
