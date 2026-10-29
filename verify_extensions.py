"""Independent audits for the D-optimal and Multi-News additions."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from objectives import DOptimalDesign,ExperimentalDesign
from multinews.objective import GCLinObjective
from run_multinews import CACHE,ORIGINAL
from run_camera_ready import trajectory_diagnostic

def verify_extensions(rows,check_values):
    errors=[];checks=0
    def check(condition,message):
        nonlocal checks
        checks+=1
        if not condition:errors.append(message)
    root=(RESULTS / 'camera_ready')
    validation=json.loads((root/'dopt_validation.json').read_text())
    for v in validation['sizes']:
        check(v['passed'] and v['triples']>=10000 and v['small_S_triples']>=5000,'D-opt validation insufficient')
        f=DOptimalDesign.synthetic(n_exp=v['n'],d=v['d'],condition_number=5,seed=0)
        old=ExperimentalDesign.synthetic(n_exp=v['n'],d=v['d'],condition_number=5,seed=0)
        ratio=np.mean([f.g_evaluate(frozenset([e])) for e in range(f.n)])/np.mean([old.g_evaluate(frozenset([e])) for e in range(f.n)])
        check(abs(v['cost_multiplier']-ratio)<1e-13,'D-opt multiplier changed')
        check(abs(v['alpha_g']-f.alpha_g_exact)<1e-13,'D-opt alpha mismatch')
    design=[r for r in rows if r['parameters']['app']=='design']
    check(len(design)==150,'Missing D-opt cells')
    for r in design:
        p=r['parameters'];v=next(v for v in validation['sizes'] if v['n']==p['n'])
        check(p.get('design_objective')=='D-optimality','A-opt remains in current results')
        check(abs(p['cost_scale']-p['old_cost_scale']*v['cost_multiplier'])<1e-14,'Retuned D-opt cost')
        check(r['certificate']['certified'],'D-opt certificate not certified')
    reproduction=json.loads((root/'multinews_reproduction.json').read_text())
    check(reproduction['verdict']=='PASS' and reproduction['matched']==600,'Multi-News reproduction failed')
    check(reproduction.get('selected_match')=='exact integer lists','Missing exact selected-list checks')
    for path,sha in reproduction['source_manifest'].items():
        check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,'Read-only source changed: '+path)
    original=json.loads(ORIGINAL.read_text())['results']
    files=sorted((root/'multinews').glob('cell_*.json'))
    check(len(files)==600,'Missing Multi-News baseline cells')
    seen=set();sources=set();max_error=0.0
    for path in files:
        r=json.loads(path.read_text());p=r['parameters'];qi=p['qi'];lam=p['lambda_value'];key=(qi,lam)
        check(key not in seen,'Duplicate Multi-News cell');seen.add(key)
        check(set(r['algorithms'])=={'GP','Prefix','DG','SG','RG','LS','MCG'},'Missing Multi-News algorithm')
        check(r['gp_query_bound_pass'] and r['algorithms']['GP']['queries']['f']<=p['n']*p['k']+2*p['k']**2,'Multi-News GP query bound')
        old=original[qi]['per_lambda'][str(lam)];prefix=r['algorithms']['Prefix']
        check(prefix['solution']==old['selected'],'Original selected list mismatch')
        check(abs(prefix['value']-old['f_greedy'])<1e-12,'Original value mismatch')
        check(abs(prefix['original_c_hat_traj']-old['c_hat_traj'])<1e-12,'Original curvature mismatch')
        check(prefix['rouge']==old.get('rouge'),'Original ROUGE changed')
        for a in set(r['algorithms'])-{'Prefix'}:check('rouge' not in r['algorithms'][a],'New ROUGE computed')
        check(r['normalizer']['value']+1e-10>=old['best_known_val'],'Reference omits original heuristic')
        sources.update(r['source_sha256'].items())
        f=None
        if check_values:
            matrix=np.load(CACHE/f'example_{qi}.npz');f=GCLinObjective(matrix['sim'],matrix['weights'],lam)
        O=frozenset(r['normalizer']['solution']);norm=r['normalizer']['value']
        for a,result in r['algorithms'].items():
            runs=result.get('runs',[result])
            if a in ['RG','MCG']:check(len(runs)==(10 if a=='RG' else 3),'Sub-seed count mismatch')
            for run in runs:
                S=frozenset(run['solution'])
                check(len(S)<=p['k'] and S.issubset(range(p['n'])),'Multi-News infeasible set')
                check(all(isinstance(run['queries'][q],int) and run['queries'][q]>=0 for q in ['f','g','ell']),'Invalid oracle counts')
                if a=='MCG':
                    x=np.array(run['fractional'])
                    check(len(x)==p['n'] and min(x)>=-1e-9 and max(x)<=1+1e-9 and sum(x)<=p['k']+1e-8,'Multi-News MCG feasibility')
                    check(run['queries']['f']==50*50*(p['n']+1)+2,'Multi-News MCG count mismatch')
                if f:
                    err=abs(f.evaluate(S)-run['value']);max_error=max(max_error,err)
                    check(err<1e-8,'Multi-News value mismatch '+str(path)+' '+a)
                    diagnostic=trajectory_diagnostic(f,run.get('trajectory',[dict(set=S)]),O)['curvature']
                    stored=run['diagnostic']['curvature']
                    check((diagnostic is None and stored is None) or (diagnostic is not None and stored is not None and abs(diagnostic-stored)<1e-8),'Multi-News diagnostic mismatch')
                check(abs(run['ratio']-run['value']/norm)<1e-12,'Multi-News ratio mismatch')
            check(abs(result['value']-np.mean([rr['value'] for rr in runs]))<1e-12,'Multi-News mean value mismatch')
        if f:
            check(abs(f.evaluate(O)-norm)<1e-8,'Multi-News reference value mismatch')
            for rr in r['reference_runs']:
                S=frozenset(rr['solution']);check(len(S)<=p['k'],'Infeasible reference restart')
                check(abs(f.evaluate(S)-rr['value'])<1e-8,'Reference restart value mismatch')
                check(norm+1e-10>=rr['value'],'Reference not maximum')
    check(seen=={(qi,l) for qi in range(100) for l in [.1,.25,.5,.75,1.,1.5]},'Multi-News grid mismatch')
    mapping={}
    for path,sha in sources:
        file=Path(path)
        check(file.exists() and hashlib.sha256(file.read_bytes()).hexdigest()==sha,
              'Source differs from release checkout '+path)
        mapping[path+':'+sha]='release checkout'
    return dict(checks=checks,errors=errors,dopt_cells=len(design),multinews_cells=len(files),
        max_multinews_value_error=max_error,source_commit_map=mapping)
