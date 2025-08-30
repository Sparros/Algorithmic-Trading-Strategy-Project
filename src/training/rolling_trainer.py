# src/training/rolling_trainer.py
import optuna
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss
from .metrics import aucpr_brier

class RollingTrainer:
    def __init__(self, splitter, make_pipeline_fn, n_trials=30, random_state=42):
        self.splitter = splitter
        self.make_pipeline_fn = make_pipeline_fn
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params_ = None
        self.model_ = None
        self.calibrator_ = None

    def _objective(self, trial, X, y):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        pipe = self.make_pipeline_fn()
        pipe.set_params(clf__C=C)
        # Purged CV scoring on AUCPR
        oof_pred = np.zeros(len(y), dtype=float)
        for tr, te in self.splitter.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            proba = pipe.predict_proba(X.iloc[te])[:, 1]
            oof_pred[te] = proba
        score = average_precision_score(y, oof_pred)
        trial.set_user_attr("brier", brier_score_loss(y, oof_pred))
        return score

    def fit(self, X, y):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)
        self.best_params_ = study.best_params

        # Refit on full data with best params
        pipe = self.make_pipeline_fn()
        pipe.set_params(**{f"clf__{k}": v for k, v in self.best_params_.items()})
        pipe.fit(X, y)
        self.model_ = pipe

        # Calibrate on last fold’s train/test split (simple and avoids reuse-of-test)
        last = list(self.splitter.split(X, y))[-1]
        tr, te = last
        from .calibration import calibrate
        self.calibrator_ = calibrate(pipe, X.iloc[te], y.iloc[te], method="isotonic")
        return self

    def predict_proba(self, X):
        return self.calibrator_.predict_proba(X)[:, 1]
