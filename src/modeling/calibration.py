# src/modeling/calibration.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def calibrate(model, X_train, y_train, method="isotonic", cv="prefit"):
    if cv != "prefit":
        raise ValueError("Expect prefit model; pass cv='prefit'")
    calib = CalibratedClassifierCV(model, cv="prefit", method=method)
    calib.fit(X_train, y_train)
    return calib
