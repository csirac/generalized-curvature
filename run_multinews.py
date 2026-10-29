#!/usr/bin/env python3
"""Read-only source port, greedy reproduction gate, then counted baselines.

Run from experiments/. No LLM, network, ROUGE computation or source writes.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_paths import RESULTS, FIGURES, ORIGINAL
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from multinews.data import load_multinews, build_example
from multinews.similarity import tfidf_similarity
from multinews.original import greedy_best_prefix, trajectory_curvature, heuristic_best_known
from multinews.objective import GCLinObjective
from algorithms.batch_oracle import neighbor_values
from run_camera_ready import (write_json, provenance, measured, trajectory_diagnostic,
    greedy_pruning, distorted_greedy, standard_greedy, random_greedy,
    local_search, measured_continuous_greedy)

OUT=Path(RESULTS / 'camera_ready/multinews')
from release_paths import CACHE, ORIGINAL
LAMBDAS=[.1,.25,.5,.75,1.,1.5]

def prep_one(job):
    qi,example,all_data,old=job
    start=time.perf_counter()
    built=build_example(example,all_data,n_target=200,chunk_words=150,seed=42)
    sim,weights=tfidf_similarity(built['passages'],query=built['query'])
    assert built['example_idx']==old['example_idx']
    np.savez(CACHE/f'example_{qi}.npz',sim=sim,weights=weights)
    rows=[]
    for lam in LAMBDAS:
        f=GCLinObjective(sim,weights,lam)
        f.reset_queries();t=time.perf_counter()
        S,traj=greedy_best_prefix(set(range(f.n)),10,f)
        val=f(S);elapsed=time.perf_counter()-t;counts=f.query_counts
        f.reset_queries();c,_=trajectory_curvature(f,traj)
        original=old['per_lambda'][str(lam)]
        errors=[]
        if sorted(S)!=original['selected']:errors.append('selected')
        if not np.isclose(val,original['f_greedy'],rtol=1e-12,atol=1e-12):errors.append('f_greedy')
        if not np.isclose(c,original['c_hat_traj'],rtol=1e-12,atol=1e-12):errors.append('c_hat_traj')
        # Independent scalar/batch and decomposition checks on each instance.
        rng=np.random.default_rng(42+qi);batch_error=0.0
        for _ in range(10):
            A=frozenset(rng.choice(f.n,size=8,replace=False))
            for remove,es in [(True,sorted(A)),(False,[e for e in range(f.n) if e not in A][:8])]:
                for component in ['f','g']:
                    oracle=f.evaluate if component=='f' else f.g_evaluate
                    batch=neighbor_values(f,A,es,remove=remove,component=component)
                    scalar=[oracle(A-{e} if remove else A|{e}) for e in es]
                    batch_error=max(batch_error,float(np.max(np.abs(batch-scalar))))
        if batch_error>1e-10:errors.append('batch')
        mono_min=float(np.min(f.a+f.costs-f.B.sum(axis=1)))
        if mono_min< -1e-10 or np.min(f.B)<0:errors.append('decomposition')
        rows.append(dict(lambda_value=lam,errors=errors,
            value_error=abs(val-original['f_greedy']),curvature_error=abs(c-original['c_hat_traj']),
            batch_error=batch_error,min_g_marginal=mono_min,
            original_greedy=dict(solution=sorted(S),value=val,trajectory=traj,queries=counts,
                wall_time=elapsed,original_c_hat_traj=c,rouge=original.get('rouge'))))
    output=dict(qi=qi,example_idx=built['example_idx'],n=len(sim),n_source=built['n_source'],
                wall_time=time.perf_counter()-start,cells=rows)
    write_json(OUT/f'reproduction_{qi}.json',output)
    return qi,rows

def reproduce(workers):
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    data=load_multinews();old=json.loads(ORIGINAL.read_text())['results']
    jobs=[(qi,e,data,old[qi]) for qi,e in enumerate(data[:100])]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for qi,rows in pool.map(prep_one,jobs):
            results.extend(rows);print('reproduced',qi,flush=True)
    errors=[r for r in results if r['errors']]
    summary=dict(cells=len(results),matched=len(results)-len(errors),errors=errors,
        max_value_error=max(r['value_error'] for r in results),
        max_curvature_error=max(r['curvature_error'] for r in results),
        max_batch_error=max(r['batch_error'] for r in results),
        tolerance=dict(rtol=1e-12,atol=1e-12),selected_match='exact integer lists',
        verdict='PASS' if not errors and len(results)==600 else 'FAIL',
        source_manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('multinews').glob('*.py')})
    write_json(OUT.parent/'multinews_reproduction.json',summary)
    print(json.dumps(summary),flush=True)
    if summary['verdict']!='PASS':raise SystemExit(1)

def run_one(job):
    qi,lam,prov=job;start=time.perf_counter()
    matrix=np.load(CACHE/f'example_{qi}.npz')
    f=GCLinObjective(matrix['sim'],matrix['weights'],lam);k=10
    repro=json.loads((OUT/f'reproduction_{qi}.json').read_text())
    prefix=next(c['original_greedy'] for c in repro['cells'] if c['lambda_value']==lam)
    algs={'Prefix':prefix}
    algs['GP']=measured(f,lambda:greedy_pruning(f,k,trace=True))
    algs['DG']=measured(f,lambda:distorted_greedy(f,k,trace=True))
    algs['SG']=measured(f,lambda:standard_greedy(f,k,trace=True))
    algs['LS']=measured(f,lambda:local_search(f,k,max_iters=200,trace=True))
    for name,seeds,fn in [('RG',[1000*s for s in range(10)],random_greedy),
                          ('MCG',[17000,18000,19000],measured_continuous_greedy)]:
        runs=[measured(f,lambda seed=seed:fn(f,k,seed=seed,**(dict(T=50,m=50) if name=='MCG' else dict(trace=True)))) for seed in seeds]
        algs[name]=dict(value=float(np.mean([r['value'] for r in runs])),
            queries={q:float(np.mean([r['queries'][q] for r in runs])) for q in ['f','g','ell']},
            wall_time=float(np.mean([r['wall_time'] for r in runs])),runs=runs)
    candidates=[rr for r in algs.values() for rr in r.get('runs',[r])]
    reference_runs=[]
    # Repeat the original twenty restart procedure and top-k check.
    f.reset_queries();t=time.perf_counter()
    val,S=heuristic_best_known(set(range(f.n)),k,f,set(prefix['solution']),prefix['value'],
                               n_restarts=20,seed=42+qi*100+int(lam*1000))
    reference_runs.append(dict(solution=sorted(S),value=val,queries=f.query_counts,
                               wall_time=time.perf_counter()-t,kind='original 20 heuristic restarts'))
    top=frozenset(np.argsort(matrix['weights'])[::-1][:k])
    reference_runs.append(measured(f,lambda:dict(solution=top,value=f.evaluate(top),kind='original topk')))
    old=json.loads(ORIGINAL.read_text())['results'][qi]['per_lambda'][str(lam)]
    old_reproduced=max(r['value'] for r in reference_runs)
    if not np.isclose(old_reproduced,old['best_known_val'],rtol=1e-10,atol=1e-10):
        raise ValueError(('original reference mismatch',qi,lam,old_reproduced,old['best_known_val']))
    initial=max(candidates+reference_runs,key=lambda r:r['value'])['solution']
    reference_runs.append(measured(f,lambda:local_search(f,k,initial=initial,max_iters=200,trace=True)))
    for seed in [10000,11000,12000]:
        rr=measured(f,lambda:random_greedy(f,k,seed=seed,trace=True));reference_runs.append(rr)
        reference_runs.append(measured(f,lambda:local_search(f,k,initial=rr['solution'],max_iters=200,trace=True)))
    best=max(candidates+reference_runs,key=lambda r:r['value']);O=frozenset(best['solution']);norm=best['value']
    f.reset_queries()
    for name,r in algs.items():
        r['ratio']=r['value']/norm if norm>1e-12 else None
        for rr in r.get('runs',[r]):
            rr['ratio']=rr['value']/norm if norm>1e-12 else None
            rr['diagnostic']=trajectory_diagnostic(f,rr.get('trajectory',[dict(set=rr['solution'])]),O)
        valid=[rr['diagnostic']['curvature'] for rr in r.get('runs',[r])]
        r['diagnostic_curvature']=float(np.mean(valid)) if all(v is not None for v in valid) else None
        if name=='MCG':r['diagnostic_scope']='rounded terminal set only'
    result=dict(parameters=dict(stage='multinews',app='multinews',qi=qi,
        example_idx=repro['example_idx'],n=f.n,k=k,lambda_value=lam,seed=42,T=50,m=50,
        rg_subseeds=[1000*s for s in range(10)],mcg_subseeds=[17000,18000,19000]),
        **prov,algorithms=algs,reference_runs=reference_runs,
        normalizer=dict(value=norm,solution=sorted(O),kind='best known, non-certified'),
        original_reference_value=old['best_known_val'],postprocessing_queries=f.query_counts,
        gp_query_bound=f.n*k+2*k*k,gp_query_bound_pass=algs['GP']['queries']['f']<=f.n*k+2*k*k,
        wall_time=time.perf_counter()-start,
        decomposition='ell_e=max(0,sum_j lambda(s_ej+s_je)-a_e); g=f+ell',
        rouge_scope='original greedy selections only; no baseline ROUGE computed')
    write_json(OUT/f'cell_{qi}_lambda{lam:g}.json',result)
    return qi,lam,result['wall_time']

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['reproduce','run']);p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int)
    args=p.parse_args();assert 1<=args.workers<=12
    if args.mode=='reproduce':return reproduce(args.workers)
    gate=json.loads((OUT.parent/'multinews_reproduction.json').read_text())
    assert gate['verdict']=='PASS' and gate['matched']==600
    prov=provenance()
    for path in [Path('run_multinews.py'),*Path('multinews').glob('*.py')]:
        prov['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    jobs=[(qi,lam,prov) for qi in range(100) for lam in LAMBDAS if not (OUT/f'cell_{qi}_lambda{lam:g}.json').exists()]
    if args.limit:jobs=jobs[:args.limit]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_one,jobs):print(result,flush=True)

if __name__=='__main__':main()
