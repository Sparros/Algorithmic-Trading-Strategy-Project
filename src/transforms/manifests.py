# src/transforms/manifests.py
from __future__ import annotations
import json, hashlib, platform, sys
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

def _df_hash(df: pd.DataFrame, n=1_000) -> str:
    # sample for speed but keep deterministic (head + tail)
    head = df.head(n).to_csv(index=True).encode()
    tail = df.tail(n).to_csv(index=True).encode()
    h = hashlib.sha256(head + tail).hexdigest()
    return h

@dataclass
class ManifestWriter:
    out_json_path: str

    def write(self, *,
              source_paths: list[str] | None,
              inputs_hash: dict[str, str],
              rows: int, cols: int,
              config: dict | None = None):
        meta = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "python": sys.version,
            "platform": platform.platform(),
            "inputs": inputs_hash,
            "shape": {"rows": rows, "cols": cols},
            "config": config or {}
        }
        with open(self.out_json_path, "w") as f:
            json.dump(meta, f, indent=2)

def hash_df_map(**dfs) -> dict[str,str]:
    return {name: _df_hash(df) for name, df in dfs.items() if df is not None}
