"""GCLin adapter; original scalar evaluation retained for reproduction.

Let a_e=sum_i w_i s_ie and B_ej=lambda(s_ej+s_je), B_ee=0.
f(S)=sum a_e - sum_{e<j in S} B_ej is submodular since B>=0.
Choose ell_e=max(0,sum_j B_ej-a_e), g=f+ell. The smallest marginal
of g for element e is a_e+ell_e-sum_j B_ej>=0, so g is normalized,
monotone and submodular and ell is nonnegative modular. Thus DG applies.
The decomposition is fixed by the instance, before running any algorithms.
"""
import numpy as np
from objectives.base import DecomposableObjective
from .original import GraphCutLin

class GCLinObjective(DecomposableObjective):
    def __init__(self, sim, weights, lam):
        self.original=GraphCutLin(sim,weights,lam)
        self.a=weights @ sim
        self.B=lam*(sim+sim.T)
        np.fill_diagonal(self.B,0)
        self.costs=np.maximum(0,self.B.sum(axis=1)-self.a)
        self._signatures=[(self.original.sim[e].tobytes(),self.original.qw[e])
                          for e in range(self.n)]

    def equivalent(self,e,j):return self._signatures[e]==self._signatures[j]

    @property
    def n(self):return self.original.n

    def evaluate(self,S):return self.original(S)

    def __call__(self,S):return self.evaluate(frozenset(S))

    def g_evaluate(self,S):return self.original(S)+sum(self.costs[list(S)])

    def ell_evaluate(self,S):return float(sum(self.costs[list(S)]))

    def singleton(self,e):return self.evaluate(frozenset([e]))

    def neighbor_values_exact(self,S,elements,remove,component,base):
        gain=self.a[elements]-self.B[np.ix_(elements,sorted(S))].sum(axis=1)
        if component=='g':gain=gain+self.costs[elements]
        return base+(-1 if remove else 1)*gain
