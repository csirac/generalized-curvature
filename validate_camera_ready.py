"""Meaningful scalar-oracle, feasibility, rounding and small exact validation."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import itertools, json, time, subprocess, hashlib
from pathlib import Path
import numpy as np
from objectives import ExperimentalDesign, CoverageCost, FeatureSelection
from objectives.maxcut import MaxCut, make_erdos_renyi
from algorithms.batch_oracle import neighbor_values, sampled_gradient
from algorithms.mcg import pipage_round, measured_continuous_greedy
from opt_finder import find_best_solution


def main():
    start=time.perf_counter(); rng=np.random.default_rng(1741); checks=0
    instances=[ExperimentalDesign.synthetic(n_exp=8,d=3,cost_scale=.1,seed=1),
               CoverageCost.random_bipartite(n_vertices=8,n_items=12,cost_scale=.2,seed=1),
               FeatureSelection.synthetic(p=8,n_groups=2,within_corr=.9,cost_scale=.1,seed=1),
               MaxCut(make_erdos_renyi(8,.4,1))]
    for f in instances:
        for _ in range(20):
            S=frozenset(np.flatnonzero(rng.random(f.n)<.4))
            for component in (['f','g'] if hasattr(f,'g_evaluate') else ['f']):
                oracle=f.evaluate if component=='f' else f.g_evaluate
                for remove,es in [(True,sorted(S)),(False,sorted(set(range(f.n))-S))]:
                    got=neighbor_values(f,S,es,remove=remove,component=component)
                    expected=[oracle(S-{e} if remove else S|{e}) for e in es]
                    assert np.allclose(got,expected,atol=1e-10,rtol=1e-10),(type(f),got,expected)
                    checks+=len(es)
    x=np.array([.2,.8,.4,.5,.1,.3,.6,.1]); k=3
    mean=np.zeros(len(x))
    f=instances[-1]; vals=[]
    for _ in range(20000):
        S=pipage_round(x,k,rng); assert len(S)<=k
        mean[list(S)]+=1; vals.append(f.evaluate(S))
    mean/=20000
    assert np.max(abs(mean-x))<.015
    multilinear=0
    for bits in itertools.product([0,1],repeat=f.n):
        mask=np.array(bits);S=frozenset(np.flatnonzero(mask))
        multilinear+=np.prod(np.where(mask,x,1-x))*f.evaluate(S)
    assert np.mean(vals)>=multilinear-.03
    exact=[]
    for f in instances:
        O=find_best_solution(f,3,exact_threshold=10);opt=f.evaluate(O)
        runs=[]
        for seed in range(10):
            f.reset_queries();r=measured_continuous_greedy(f,3,T=50,m=50,seed=seed)
            assert len(r['solution'])<=3 and 0<=r['value']<=opt+1e-9
            assert np.allclose(r['value'],f.evaluate(r['solution']))
            # One extra validation query above.
            assert f.query_counts['f']==50*50*(f.n+1)+3
            runs.append(r['value']/opt)
        exact.append(dict(objective=type(f).__name__,opt=opt,mean_ratio=float(np.mean(runs)),min_ratio=min(runs)))
    out=dict(parameters=dict(T=50,m=50,n=8,k=3,rounding_repetitions=20000),seeds=list(range(10)),
             git_commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True).stdout.strip() or 'uncommitted',
             source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('algorithms').glob('*.py')},
             wall_time=time.perf_counter()-start,scalar_checks=checks,rounding_max_error=float(max(abs(mean-x))),
             rounding_expected_value=multilinear,rounding_observed_value=float(np.mean(vals)),exact=exact)
    Path(RESULTS / 'camera_ready/mcg_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
