import pandas as pd
from src.qa.validators import assert_monotonic

def test_monotonic():
    idx = pd.to_datetime(["2020-01-01","2020-01-02"])
    assert_monotonic(idx)
