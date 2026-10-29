"""Portable output and data locations; environment overrides work in child processes."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('GC_RESULTS', ROOT / 'results'))
FIGURES = RESULTS / 'figures'
CACHE = Path(os.environ.get('GC_CACHE', ROOT / 'cache' / 'multinews_matrices'))
DATASET = Path(os.environ.get('MULTINEWS_DATA', ROOT / 'data' / 'multinews_validation.json'))
ORIGINAL = Path(os.environ.get('GC_EXP19', RESULTS / 'exp19_lambda_sweep.json'))
