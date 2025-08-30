# src/utils/splits.py
import numpy as np
from sklearn.model_selection import KFold

class PurgedKFoldEmbargo(KFold):
    """
    Purges overlapping observations and applies an embargo around test folds.
    Assumes X is indexed by time (DatetimeIndex) and y shares the index.
    """
    def __init__(self, n_splits=5, embargo_fraction=0.01, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.embargo_fraction = embargo_fraction

    def split(self, X, y=None, groups=None):
        idx = np.arange(len(X))
        for train_idx, test_idx in super().split(idx, y, groups):
            n = len(idx)
            embargo = int(self.embargo_fraction * n)
            test_start, test_end = test_idx[0], test_idx[-1]
            # Purge overlap (here we assume label horizon; for triple-barrier you can add a mapping of t0->t1)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end+1] = False
            # Embargo
            lo = max(0, test_start - embargo)
            hi = min(n, test_end + 1 + embargo)
            train_mask[lo:test_start] = False
            train_mask[test_end+1:hi] = False
            yield idx[train_mask], test_idx
