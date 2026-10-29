"""Audit experiment completeness, output values, provenance and manuscript scope."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import argparse
import collections
import difflib
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from run_camera_ready import factory, cells, cell_name, clean, write_json
from summarize_camera_ready import load


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-values',action='store_true')
    args=parser.parse_args();start=time.perf_counter();rows=load();errors=[];checks=0
    required=[]
    for stage in ['tier1','tier2','features','maxcut']:
        required += [cell_name(p) for p in cells(stage)]
    for n in [1000,500,2000]:
        required += [cell_name(p) for p in cells('scale',n)]
    actual={Path(r['_file']).name for r in rows}
    missing=sorted(set(required)-actual);extra=sorted(actual-set(required))
    if missing:errors.append(f'Missing {len(missing)} cells')
    if extra:errors.append(f'Unexpected {len(extra)} cells')
    ls_capped=0;source_pairs=set();changes=[]
    for r in rows:
        p=r['parameters'];n,k=p['n'],p['k']
        checks+=1
        if not all(key in r for key in ['parameters','seeds','git_commit','wall_time','source_sha256']):
            errors.append(r['_file']+': missing provenance')
        source_pairs.update(r['source_sha256'].items())
        if r['gp_query_bound']!=n*k+2*k*k or r['algorithms']['GP']['queries']['f']>n*k+2*k*k:
            errors.append(r['_file']+': GP count violation')
        f=factory(p) if args.check_values else None
        for alg,result in r['algorithms'].items():
            runs=result.get('runs',[result])
            for run in runs:
                S=frozenset(run['solution']);checks+=1
                if len(S)>k or not S.issubset(range(n)):
                    errors.append(r['_file']+': infeasible '+alg)
                if args.check_values and not np.isclose(f.evaluate(S),run['value'],rtol=1e-9,atol=1e-8):
                    errors.append(r['_file']+': value mismatch '+alg)
                if alg=='MCG':
                    x=np.array(run['fractional']);checks+=1
                    if len(x)!=n or min(x)<-1e-9 or max(x)>1+1e-9 or x.sum()>k+1e-8:
                        errors.append(r['_file']+': fractional infeasibility')
                    expected=p['T']*p['m']*(n+1)+2
                    if run['queries']['f']!=expected:
                        errors.append(r['_file']+': MCG count mismatch')
            if 'runs' in result and not np.isclose(result['value'],np.mean([rr['value'] for rr in runs])):
                errors.append(r['_file']+': stochastic mean mismatch')
        if not r['algorithms']['LS']['converged']:ls_capped+=1
        if args.check_values and not np.isclose(f.evaluate(frozenset(r['normalizer']['solution'])),r['normalizer']['value'],rtol=1e-9,atol=1e-8):
            errors.append(r['_file']+': reference value mismatch')
        if 'legacy_GP' in r and not np.isclose(r['legacy_GP']['value'],r['algorithms']['GP']['value'],rtol=1e-9,atol=1e-8):
            changes.append(r['_file'])
        cert=r.get('certificate')
        if cert and p['app']=='design' and (not cert['certified'] or p.get('design_objective')!='D-optimality'):
            errors.append(r['_file']+': invalid design certificate label')
        ratio=r['algorithms']['GP']['ratio']
        if cert and cert['certified'] and ratio is not None and ratio+1e-8<cert['guarantee']:
            errors.append(r['_file']+': certificate exceeds observed ratio')

    mapping={}
    for file,sha in sorted(source_pairs):
        path=Path(file)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            errors.append('Source hash differs from release checkout: '+file)
        else:mapping[file+':'+sha]='release checkout'
    from verify_extensions import verify_extensions
    ext=verify_extensions(rows,args.check_values)
    checks+=ext['checks'];errors.extend(ext['errors'])
    out=dict(parameters=dict(check_values=args.check_values,expected_cells=len(required)),seeds=sorted({r['parameters']['seed'] for r in rows}),
             git_commit=git('rev-parse','HEAD').decode().strip(),wall_time=time.perf_counter()-start,
             verdict='PASS' if not errors else 'FAIL',checks=checks,instances=len(rows),missing=missing,extra=extra,
             errors=errors,legacy_GP_changes=changes,ls_capped=ls_capped,source_commit_map=mapping,
             extensions=ext)
    write_json(Path(RESULTS / 'camera_ready/verification.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['source_commit_map','main_tex_hunks']},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
