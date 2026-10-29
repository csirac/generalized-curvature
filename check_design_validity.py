"""Reproduce a diminishing-returns violation in the submitted design setup."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from objectives.experimental_design import ExperimentalDesign


def main():
    start=time.perf_counter()
    f=ExperimentalDesign.synthetic(n_exp=20,d=5,condition_number=5,seed=0)
    S=frozenset({12});e,j=17,10
    sets=[S,S|{e},S|{j},S|{e,j}]
    vals=[f.g_evaluate(A) for A in sets]
    independent=[]
    for A in sets:
        X=f._X[sorted(A)]
        M=np.eye(5)+X.T@X
        independent.append(float(5-np.trace(solve(M,np.eye(5),assume_a='pos'))))
    assert np.allclose(vals,independent,atol=1e-12,rtol=0)
    small=vals[1]-vals[0];large=vals[3]-vals[2]
    assert large-small>1e-3
    source=Path('objectives/experimental_design.py')
    result=dict(parameters=dict(n_exp=20,d=5,condition_number=5,noise_var=1,cost_scale=0),
        seeds=[0],git_commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True).stdout.strip() or 'uncommitted',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),wall_time=time.perf_counter()-start,
        S=[12],e=e,j=j,values=vals,independent_cholesky_values=independent,
        marginal_small=small,marginal_large=large,violation=large-small,
        conclusion='Not submodular; the violation uses sets of size at most 3, within k=5.')
    Path(RESULTS / 'camera_ready/design_validity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
