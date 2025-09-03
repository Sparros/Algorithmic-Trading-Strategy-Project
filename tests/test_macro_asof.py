import pandas as pd
from src.transforms.macro_asof import AsOfMacroMerger

def test_asof_publish_lag():
    stock = pd.DataFrame({"Close_X": [10,11,12]}, index=pd.to_datetime(["2020-01-02","2020-01-03","2020-01-06"]))
    macro = pd.DataFrame({"CPIAUCSL": [100]}, index=pd.to_datetime(["2020-01-03"]))
    merger = AsOfMacroMerger(publish_lag_days={"CPIAUCSL": 1}, tolerance_days=10)
    out = merger.transform(stock, macro)
    # CPI published Jan 3 with +1BD lag → first available Jan 6
    assert pd.isna(out.loc["2020-01-03","CPIAUCSL"])
    assert out.loc["2020-01-06","CPIAUCSL"] == 100
