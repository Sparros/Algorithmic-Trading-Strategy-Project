# Pseudocode; adapt to your repo objects
from utils.labels import triple_barrier_labels
from utils.splits import PurgedKFoldEmbargo
from modeling.pipeline import make_baseline_pipeline
from training.rolling_trainer import RollingTrainer
from execution.sizer import prob_to_weight, volatility_target, apply_turnover_penalty
from execution.costs import cost_model

# 0) Load features X, prices, returns r
# X: DataFrame[T x F]; prices: Series[T]; r: returns aligned to X *future* bar
# 1) Labels
vol = r.rolling(20).std().shift(1)  # simple rolling vol
y = triple_barrier_labels(prices=prices, vol=vol, horizon=20, up_mult=1.0, dn_mult=1.0)

# 2) Splitter
splitter = PurgedKFoldEmbargo(n_splits=5, embargo_fraction=0.02)

# 3) Train + calibrate
trainer = RollingTrainer(splitter=splitter, make_pipeline_fn=make_baseline_pipeline, n_trials=25)
trainer.fit(X.loc[y.index], y)

# 4) Predict proba OOS (walk-forward loop recommended; here, simple demo)
p = trainer.predict_proba(X)

# 5) Sizing
w = prob_to_weight(pd.Series(p, index=X.index), w_max=0.5)
w = volatility_target(w, returns=r, target_vol=0.10, lookback=63)
w = apply_turnover_penalty(w, kappa=0.5)

# 6) PnL after costs
trades = w.diff().fillna(w)
gross = (w.shift(1) * r).fillna(0.0)
costs = cost_model(trades, price=prices, spread_bps=5, fee_bps=1, slippage_bps=2)
net = gross - costs
