#!/usr/bin/env python3
"""Reproducible camera-ready runs; legacy results are never overwritten.

Run from experiments/ with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 and nice -n 10.
Example: python3 run_camera_ready.py --stage scale --n 1000 --workers 12
Results are one atomic JSON per instance; --only-missing safely resumes cells.
"""
from release_paths import RESULTS, FIGURES, ORIGINAL
import argparse, concurrent.futures, hashlib, itertools, json, os, subprocess, time
from pathlib import Path
import numpy as np
from objectives import ExperimentalDesign, DOptimalDesign, CoverageCost, FeatureSelection
from objectives.maxcut import MaxCut, make_planted_distractor, make_erdos_renyi
from algorithms.greedy_pruning import greedy_pruning
from algorithms.distorted_greedy import distorted_greedy
from algorithms.standard_greedy import standard_greedy
from algorithms.random_greedy import random_greedy
from algorithms.local_search import local_search
from algorithms.mcg import measured_continuous_greedy
from compute_removal_cert import compute_removal_marginal_cert
from opt_finder import find_best_solution
from curvature import guarantee_curvature, guarantee_harshaw

OUT=(RESULTS / 'camera_ready')
COSTS1=dict(design=[0,.03,.06,.10,.15,.20,.28],coverage=[0,.5,1,2,3.5,5,8],features=[0,.05,.1,.2,.3,.5,.8])
COSTS2=dict(design=[0,.02,.05,.10,.15,.22,.30],coverage=[0,.5,1.5,3,5,8,12],features=[0,.03,.08,.15,.25,.4,.6])
# Same fixed five cost levels across sizes, chosen before any scale results.
COSTSS=dict(design=[0,.02,.05,.10,.15],coverage=[0,1.5,5,8,12],features=[0,.08,.25,.4,.6])
NAMES=dict(design='experimental_design',coverage='coverage_cost',features='feature_selection')


def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple,set,frozenset,np.ndarray)):return [clean(v) for v in obj]
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.bool_):return bool(obj)
    return obj


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def provenance():
    paths=list(Path('algorithms').glob('*.py'))+list(Path('objectives').glob('*.py'))
    paths += [Path(x) for x in ['run_camera_ready.py','curvature.py','compute_removal_cert.py','opt_finder.py']]
    return dict(git_commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True).stdout.strip() or 'uncommitted',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def factory(p):
    n,cs,seed=p['n'],p['cost_scale'],p['seed']
    if p['app']=='design':
        return (DOptimalDesign if p.get('design_objective')=='D-optimality' else ExperimentalDesign).synthetic(n_exp=n,d=p['d'],condition_number=5,cost_scale=cs,noise_var=1,seed=seed)
    if p['app']=='coverage':
        return CoverageCost.random_bipartite(n_vertices=n,n_items=p['n_items'],edge_prob=p['edge_prob'],cost_scale=cs,seed=seed)
    if p['app']=='features':
        return FeatureSelection.synthetic(p=n,n_groups=p['n_groups'],within_corr=p['within_corr'],cost_scale=cs,noise_var=1,seed=seed)
    graph_seed=seed*1000+n*100+p['k']
    graph=make_planted_distractor(n,p['k'],graph_seed) if p['app']=='planted_distractor' else make_erdos_renyi(n,.3,graph_seed)
    return MaxCut(graph)


def exact_opt(f,k):
    if not isinstance(f,MaxCut):return find_best_solution(f,k,exact_threshold=f.n+1)
    masks=np.arange(1<<f.n,dtype=np.uint32);vals=np.zeros(len(masks))
    for u in range(f.n):
        for v in range(u+1,f.n):
            if f._adj[u,v]:vals+=((masks>>u)^(masks>>v))&1
    # This vectorized reference computes a value for every subset.
    f.record_queries('f',len(masks))
    vals[np.bitwise_count(masks)>k]=-np.inf
    mask=int(np.argmax(vals))
    return frozenset(e for e in range(f.n) if mask&(1<<e))


def alpha_for_certificate(f):
    if isinstance(f,DOptimalDesign):
        return f.alpha_g_exact,'exact full-set removal / singleton ratio'
    if isinstance(f,FeatureSelection):
        # Exact CC curvature of the normalized monotone submodular logdet g.
        M=np.eye(f.n)+f._cov/f._noise_var
        inverse=np.linalg.inv(M)
        singleton=.5*np.log(np.diag(M))
        removal=-.5*np.log(np.diag(inverse))
        return float(np.clip(1-np.min(removal/singleton),0,1)), 'exact full-set removal / singleton ratio'
    if isinstance(f,CoverageCost):return 1.0,'universal upper bound for normalized monotone submodular g'
    return float(f.alpha_g_exact),'legacy eigenvalue diagnostic; A-optimality is not submodular'


def trajectory_diagnostic(f,trajectory,O):
    ratios=[];nonpositive=0;valueO=f.evaluate(O)
    for step in trajectory:
        A=frozenset(step['set']);B=A-O
        if not B:continue
        denom=f.evaluate(B)
        if denom<=1e-12:
            nonpositive+=1;continue
        ratios.append(1-(f.evaluate(O|A)-valueO)/denom)
    return dict(curvature=max([0.0]+ratios) if nonpositive==0 else None,
                nonpositive_denominators=nonpositive,admissible_steps=len(ratios),
                interpretation='trajectory diagnostic; theorem only for admissible GP/submodular instances')


def measured(f,fn):
    f.reset_queries();t=time.perf_counter();res=fn();elapsed=time.perf_counter()-t
    res['queries']=f.query_counts;res['wall_time']=elapsed
    return res


def run_instance(p):
    t=time.perf_counter();f=factory(p);k=p['k'];algs={}
    algs['GP']=measured(f,lambda:greedy_pruning(f,k,trace=True))
    if hasattr(f,'g_evaluate'):
        algs['DG']=measured(f,lambda:distorted_greedy(f,k,trace=True))
    algs['SG']=measured(f,lambda:standard_greedy(f,k,trace=True))
    rg=[measured(f,lambda s=s:random_greedy(f,k,seed=s,trace=True)) for s in p['rg_subseeds']]
    algs['RG']=dict(value=float(np.mean([r['value'] for r in rg])),
                    queries={q:float(np.mean([r['queries'][q] for r in rg])) for q in ['f','g','ell']},
                    wall_time=float(np.mean([r['wall_time'] for r in rg])),runs=rg)
    algs['LS']=measured(f,lambda:local_search(f,k,max_iters=200,trace=True))
    mcg=[measured(f,lambda s=s:measured_continuous_greedy(f,k,T=p['T'],m=p['m'],seed=s)) for s in p['mcg_subseeds']]
    algs['MCG']=dict(value=float(np.mean([r['value'] for r in mcg])),
                     queries={q:float(np.mean([r['queries'][q] for r in mcg])) for q in ['f','g','ell']},
                     wall_time=float(np.mean([r['wall_time'] for r in mcg])),runs=mcg)
    reference_runs=[]
    if p['stage'] in ['tier1','features','maxcut']:
        f.reset_queries();O=exact_opt(f,k);normalizer=f.evaluate(O);reference_kind='exact OPT'
        reference_queries=f.query_counts
    else:
        candidates=[r for a,r in algs.items() if a not in ['RG','MCG']]+rg+mcg
        initial=max(candidates,key=lambda r:r['value'])['solution']
        reference_runs.append(measured(f,lambda:local_search(f,k,initial=initial,max_iters=200,trace=True)))
        for seed in [10000,11000,12000]:
            rr=measured(f,lambda:random_greedy(f,k,seed=seed,trace=True))
            reference_runs.append(rr)
            reference_runs.append(measured(f,lambda:local_search(f,k,initial=rr['solution'],max_iters=200,trace=True)))
        best=max(candidates+reference_runs,key=lambda r:r['value']);O=best['solution'];normalizer=best['value']
        reference_kind='best known, non-certified'
        reference_queries={q:sum(r['queries'][q] for r in reference_runs) for q in ['f','g','ell']}
    post=time.perf_counter();f.reset_queries()
    for name,r in algs.items():
        r['ratio']=r['value']/normalizer if normalizer>1e-12 else None
        if name in ['RG','MCG']:
            for rr in r['runs']:
                rr['ratio']=rr['value']/normalizer if normalizer>1e-12 else None
                traj=rr.get('trajectory',[dict(set=rr['solution'])])
                rr['diagnostic']=trajectory_diagnostic(f,traj,O)
            valid=[rr['diagnostic']['curvature'] for rr in r['runs'] if rr['diagnostic']['curvature'] is not None]
            r['diagnostic_curvature']=float(np.mean(valid)) if len(valid)==len(r['runs']) else None
            if name=='MCG':r['diagnostic_scope']='rounded terminal set only; no discrete greedy trajectory for MCG'
        else:
            r['diagnostic']=trajectory_diagnostic(f,r['trajectory'],O)
            r['diagnostic_curvature']=r['diagnostic']['curvature']
    certificate=None
    if hasattr(f,'g_evaluate'):
        alpha,alpha_source=alpha_for_certificate(f)
        certificate=compute_removal_marginal_cert(f,algs['GP']['trajectory'],alpha)
        certificate.update(alpha_g=alpha,alpha_source=alpha_source,
                           certified=not isinstance(f,ExperimentalDesign) or isinstance(f,DOptimalDesign),
                           interpretation='formal GP certificate' if not isinstance(f,ExperimentalDesign) or isinstance(f,DOptimalDesign) else 'diagnostic only: not submodular')
        gO=f.g_evaluate(O);ellO=f.ell_evaluate(O);rho=ellO/gO if gO>0 else None
    else:rho=None
    out=dict(parameters=p,seeds=dict(instance=p['seed'],RG=p['rg_subseeds'],MCG=p['mcg_subseeds'],reference_RG=[10000,11000,12000]),
             **p['provenance'],wall_time=time.perf_counter()-t,algorithms=algs,
             normalizer=dict(value=normalizer,solution=sorted(O),kind=reference_kind,queries=reference_queries),
             reference_runs=reference_runs,certificate=certificate,rho=rho,
             harshaw=guarantee_harshaw(rho) if rho is not None else None,
             postprocessing_queries=f.query_counts,postprocessing_wall_time=time.perf_counter()-post,
             gp_query_bound=f.n*k+2*k*k,gp_query_bound_pass=algs['GP']['queries']['f']<=f.n*k+2*k*k)
    # Reuse, and explicitly name, the submitted RG/SG results.
    if p['stage']=='tier1' and p['app']!='design' and (RESULTS/(NAMES[p['app']]+'.json')).exists() and not p.get('smoke'):
        legacy=json.loads((RESULTS/(NAMES[p['app']]+'.json')).read_text())
        row=next(r for r in legacy if r['cost_scale']==p['cost_scale'])
        old=next(r for r in row['per_seed'] if r['seed']==p['seed'])
        out['legacy_RG_shrinking_pool']=old['random_greedy']
        out['legacy_SG']=old['standard_greedy']
        out['legacy_GP']=old['greedy_pruning']
    filename=cell_name(p)
    write_json(OUT/filename,out)
    return filename,out['wall_time'],out['gp_query_bound_pass']


def cell_name(p):
    return f"{p['stage']}_{p['app']}_n{p['n']}_k{p['k']}_c{p['cost_scale']:g}_s{p['seed']}.json"


def cells(stage,n=None,T=None,m=None):
    prov=provenance();configs=[]
    if stage=='maxcut':
        from run_maxcut import CONFIGS
        configs=[dict(app=app,n=nn,k=k,cost_scale=0,seed=s) for app,nn,k in CONFIGS for s in range(20)]
    else:
        apps=['features'] if stage=='features' else ['design','coverage','features']
        for app in apps:
            nn,kk=(20,5) if stage in ['tier1','features'] else ((n,{500:20,1000:30,2000:50}[n]) if stage=='scale' else dict(design=(200,20),coverage=(300,30),features=(100,15))[app])
            costs=COSTS1[app] if stage in ['tier1','features'] else (COSTSS[app] if stage=='scale' else COSTS2[app])
            ns=10 if stage in ['tier1','features'] else (3 if stage=='scale' else 5)
            for cs in costs:
                for seed in range(ns):
                    p=dict(app=app,n=nn,k=kk,cost_scale=cs,seed=seed)
                    if app=='design':
                        grid=json.loads((OUT/'dopt_validation.json').read_text())
                        check=next(x for x in grid['sizes'] if x['n']==nn)
                        if not check['passed']:raise ValueError('D-opt validation failed')
                        p.update(d=5 if nn==20 else 10,condition_number=5,noise_var=1,
                                 design_objective='D-optimality',old_cost_scale=cs,
                                 cost_multiplier=check['cost_multiplier'],
                                 cost_scale=cs*check['cost_multiplier'],cost_rule=grid['cost_rule'])
                    if app=='coverage':p.update(n_items=2*nn,edge_prob=.2 if nn==20 else .05)
                    if app=='features':p.update(n_groups=3 if stage=='features' else (4 if nn==20 else 10),within_corr=.9 if stage=='features' else .7,noise_var=1,cross_group_noise=.05)
                    configs.append(p)
    for p in configs:
        p.update(stage=stage,T=T or 50,m=m or (16 if stage=='scale' else 50),
                 rg_subseeds=[1000*s for s in range(10)],mcg_subseeds=[17000,18000,19000],provenance=prov)
    return configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['tier1','tier2','features','scale','maxcut'],required=True)
    parser.add_argument('--n',type=int,choices=[500,1000,2000])
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--only-missing',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--app')
    parser.add_argument('--smoke',action='store_true',help='One n=8 synthetic coverage cell; not paper settings')
    parser.add_argument('--T',type=int);parser.add_argument('--m',type=int)
    args=parser.parse_args()
    if not 1<=args.workers<=12:parser.error('Use at most 12 child processes')
    if args.stage=='scale' and args.n is None:parser.error('--n required for scale')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        jobs=[dict(stage=args.stage,app='coverage',n=8,k=2,cost_scale=.1,seed=0,
                   n_items=16,edge_prob=.2,T=3,m=3,rg_subseeds=[0],mcg_subseeds=[17000],
                   provenance=provenance(),smoke=True)]
    else:
        jobs=cells(args.stage,args.n,args.T,args.m)
    if args.app:jobs=[p for p in jobs if p['app']==args.app]
    if args.only_missing:jobs=[p for p in jobs if not (OUT/cell_name(p)).exists()]
    if args.limit:jobs=jobs[:args.limit]
    print(f'jobs={len(jobs)}, workers={args.workers}',flush=True)
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_instance,p):p for p in jobs}
        for future in concurrent.futures.as_completed(futures):
            try: print(future.result(),flush=True)
            except Exception as exc:
                print('ERROR',cell_name(futures[future]),repr(exc),flush=True);errors.append(str(exc))
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
