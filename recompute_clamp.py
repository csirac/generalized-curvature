"""Reproduce the Gap-1 corrections without overwriting submitted results."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import json, subprocess, time, hashlib
from pathlib import Path
import numpy as np
from curvature import guarantee_curvature
from plotting import plot_cost_sweep, plot_curvature_tightness

def main():
    start=time.perf_counter()
    out=(RESULTS / 'camera_ready'); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name, short in [('experimental_design','design'),('coverage_cost','coverage'),('feature_selection','features')]:
        source=RESULTS/(name+'.json')
        data=json.loads(source.read_text())
        for row in data:
            old=row['curv_guarantee_mean']
            values=[]
            for seed in row['per_seed']:
                if 'empirical_c_g' in seed:
                    seed['curv_guarantee']=guarantee_curvature(seed['empirical_c_g'])
                    values.append(seed['curv_guarantee'])
            row['curv_guarantee_mean']=float(np.mean(values))
            row['curv_guarantee_std']=float(np.std(values))
            rows.append(dict(application=name,cost_scale=row['cost_scale'],old=old,new=row['curv_guarantee_mean']))
        payload=dict(parameters=dict(operation='per-seed max(1,c) clamp',source=str(source)),
                     seeds=list(range(10)),git_commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True).stdout.strip() or 'uncommitted',
                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),wall_time=time.perf_counter()-start,results=data)
        (out/(name+'_clamped.json')).write_text(json.dumps(payload,indent=2)+'\n')
        if short!='features':
            plot_cost_sweep(data,str(FIGURES / 'fig_t1_')+short+'_sweep.pdf',title=name.replace('_',' ').title())
            plot_curvature_tightness(data,str(FIGURES / 'fig_t1_')+short+'_tightness.pdf',title=name.replace('_',' ').title())
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
