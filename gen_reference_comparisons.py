"""Regenerate Figure 2(a) and the illustrative parameter comparison table.

Partial-monotonicity formulas: Mualem and Feldman (NeurIPS 2022),
Theorems 4.1, 4.2, 6.1: https://arxiv.org/abs/2202.03051.
These curves assume uniform query weights, not the TF-IDF instance weights.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from release_paths import RESULTS, FIGURES
from objectives import DOptimalDesign
from curvature import guarantee_curvature, guarantee_harshaw


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    lam=np.linspace(0,1,401)
    m=np.minimum(1,2*(1-lam))
    curvature=np.ones_like(lam)
    np.divide(-np.expm1(-2*lam),2*lam,out=curvature,where=lam>0)
    fig,ax=plt.subplots(figsize=(5,3.5))
    ax.plot(lam,m*(1-1/np.e),label='Greedy (mon-ratio)')
    ax.plot(lam,m*(1-1/np.e)+(1-m)/np.e,label='Random Greedy (mon-ratio)')
    ax.plot(lam,curvature,label='Greedy (curvature)')
    ax.set(xlabel=r'$\lambda$',ylabel=r'$f(S)/f(O)$',xlim=(0,1),ylim=(0,1.02))
    ax.legend();fig.tight_layout();fig.savefig(FIGURES/'PMovieRec.pdf');plt.close(fig)
    alpha=DOptimalDesign.synthetic(n_exp=20,d=5,condition_number=5,seed=0).alpha_g_exact
    configs=[('D-opt. design (low)',alpha,.1),('D-opt. design (med)',alpha,.4),('D-opt. design (high)',alpha,.6),
             ('Feature selection',.5,.3),('Feature selection',.5,.5),('Vertex cover',1.,.5)]
    rows=['Application,alpha_g,rho,HFWK,Curvature']
    for name,a,rho in configs:
        rows.append(f'{name},{a:.9f},{rho:g},{guarantee_harshaw(rho):.3f},{guarantee_curvature(a/(1-rho)):.3f}')
    (RESULTS/'comparison.csv').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':main()
