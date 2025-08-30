# src/modeling/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def make_baseline_pipeline(model=None):
    # Start simple; you can swap in XGBoost/CatBoost later
    if model is None:
        model = LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced")
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", model),
    ])
    return pipe
