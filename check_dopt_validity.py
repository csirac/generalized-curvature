"""Freeze D-opt cost normalization and validate BEFORE running algorithms."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import json
import time
from pathlib import Path
import numpy as np
from objectives import ExperimentalDesign, DOptimalDesign
from algorithms.batch_oracle import neighbor_values

def main():
    path = Path(RESULTS / 'camera_ready/dopt_validation.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise SystemExit('Frozen validation already exists; do not retune the grid.')
    sizes = []
    for n in [20, 200, 500, 1000, 2000]:
        d = 5 if n == 20 else 10
        old = ExperimentalDesign.synthetic(n_exp=n, d=d, condition_number=5, seed=0)
        f = DOptimalDesign.synthetic(n_exp=n, d=d, condition_number=5, seed=0)
        singles = [f.g_evaluate(frozenset([e])) for e in range(n)]
        old_singles = [old.g_evaluate(frozenset([e])) for e in range(n)]
        r = float(np.mean(singles) / np.mean(old_singles))
        rng = np.random.default_rng(20261028+n)
        sub_violation = mono_violation = 0.0
        minimum_marginal = float('inf')
        batch_error = 0.0
        for i in range(10000):
            order = rng.permutation(n)
            ns = int(rng.integers(0, min(n-1, 5))) if i % 2 == 0 else int(rng.integers(0,n-1))
            nt = int(rng.integers(ns+1,n))
            S, T, e = frozenset(order[:ns]), frozenset(order[:nt]), int(order[nt])
            ds = f.g_evaluate(S|{e})-f.g_evaluate(S)
            dt = f.g_evaluate(T|{e})-f.g_evaluate(T)
            sub_violation = max(sub_violation, dt-ds)
            mono_violation = max(mono_violation, -ds, -dt)
            minimum_marginal = min(minimum_marginal, ds, dt)
            if i < 100:
                for A, remove in [(S,False),(T|{e},True)]:
                    val = neighbor_values(f,A,[e],remove=remove,component='g')[0]
                    expected = f.g_evaluate(A-{e} if remove else A|{e})
                    batch_error = max(batch_error,abs(val-expected))
            # Validation need not retain large random subsets in the cache.
            if i % 100 == 0:f._cache.clear()
        full=f.g_evaluate(frozenset(range(n)))
        removals=[full-f.g_evaluate(frozenset(range(n))-{e}) for e in range(n)]
        direct_alpha=float(1-min(np.array(removals)/singles))
        entry=dict(n=n,d=d,seed=0,triples=10000,small_S_triples=5000,
            cost_multiplier=r,mean_singleton_dopt=float(np.mean(singles)),
            mean_singleton_aopt=float(np.mean(old_singles)),alpha_g=f.alpha_g_exact,
            alpha_direct=direct_alpha,max_submodularity_violation=sub_violation,
            max_monotonicity_violation=mono_violation,minimum_marginal=minimum_marginal,
            max_batch_value_error=batch_error,
            passed=max(sub_violation,mono_violation,batch_error,abs(direct_alpha-f.alpha_g_exact))<1e-10)
        sizes.append(entry);print(json.dumps(entry),flush=True)
    result=dict(cost_rule='For each tier/size multiply every old design cost by mean singleton D-opt / mean singleton A-opt on seed 0; fixed before all algorithms; no retuning.',
        frozen_unix=time.time(),sizes=sizes,passed=all(x['passed'] for x in sizes))
    path.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
