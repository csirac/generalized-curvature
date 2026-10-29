"""Abstract base class for submodular objective functions."""

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np


class ObjectiveFunction(ABC):
    """Interface for set functions f: 2^V -> R.

    Elements are integers 0..n-1.  Sets are represented as sorted tuples
    or frozensets.  Implementations must provide `evaluate` and `n`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Instrument concrete overrides, including inherited decomposable f.
        for name, component in [('evaluate', 'f'), ('g_evaluate', 'g'),
                                ('ell_evaluate', 'ell')]:
            method = cls.__dict__.get(name)
            if method is None or getattr(method, '__isabstractmethod__', False):
                continue
            def wrap(method, component):
                @wraps(method)
                def counted(self, *args, **kwargs):
                    depth = getattr(self, '_oracle_depth', 0)
                    if depth == 0:
                        self.record_queries(component, 1)
                    self._oracle_depth = depth + 1
                    try:
                        return method(self, *args, **kwargs)
                    finally:
                        self._oracle_depth = depth
                return counted
            setattr(cls, name, wrap(method, component))

    def reset_queries(self):
        """Reset before each algorithm; post-processing is counted separately."""
        self._queries = dict(f=0, g=0, ell=0)

    def record_queries(self, component, count):
        """Also used by vectorized batches: one query per returned set value."""
        if not hasattr(self, '_queries'):
            self.reset_queries()
        self._queries[component] += int(count)

    @property
    def query_counts(self):
        return dict(getattr(self, '_queries', dict(f=0, g=0, ell=0)))

    @property
    @abstractmethod
    def n(self) -> int:
        """Ground set size."""

    @abstractmethod
    def evaluate(self, S: frozenset) -> float:
        """Evaluate f(S)."""

    # ---- convenience wrappers ----

    def marginal(self, e: int, S: frozenset) -> float:
        """Marginal gain of adding element e to set S."""
        if e in S:
            return 0.0
        return self.evaluate(S | {e}) - self.evaluate(S)

    @property
    def ground_set(self) -> list:
        return list(range(self.n))


class DecomposableObjective(ObjectiveFunction, ABC):
    """f = g - ℓ  where g is monotone submodular and ℓ is modular.

    Subclasses provide g_evaluate and ell_evaluate separately.
    This lets the distorted greedy (which needs g and ℓ apart) run
    on the same instance.
    """

    @abstractmethod
    def g_evaluate(self, S: frozenset) -> float:
        """Evaluate the monotone submodular part g(S)."""

    @abstractmethod
    def ell_evaluate(self, S: frozenset) -> float:
        """Evaluate the modular cost part ℓ(S)."""

    def evaluate(self, S: frozenset) -> float:
        return self.g_evaluate(S) - self.ell_evaluate(S)

    def g_marginal(self, e: int, S: frozenset) -> float:
        if e in S:
            return 0.0
        return self.g_evaluate(S | {e}) - self.g_evaluate(S)

    def ell_element(self, e: int) -> float:
        """Per-element cost ℓ({e}).  (ℓ is modular, so ℓ(S) = sum ℓ(e).)"""
        return self.ell_evaluate(frozenset([e]))
