"""A worked 1-D example: interpolate SIDER from a DataSAIL split to a random one.

Run it from a directory holding ``sider.csv`` with ``ID`` and ``smiles``
columns. The block at the top builds the two endpoint directories from that
file; uncomment it the first time, then leave it commented so later runs reuse
the same endpoints.

    python examples/sider_example.py
"""
from pathlib import Path

from continuous_datasail import continuous_split_1D

if __name__ == "__main__":
    # import pandas as pd
    # import numpy as np

    # df = pd.read_csv("sider.csv")
    # df["random"] = np.random.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1], size=len(df))
    # df["ID"] = [f"D{i:04d}" for i in range(len(df))]

    # sr = Path("sider_random")
    # sr.mkdir(parents=True, exist_ok=True)
    # sd = Path("sider_datasail")
    # sd.mkdir(parents=True, exist_ok=True)
    # for col, p in [("random", sr), ("datasail", sd)]:
    #     for split in ["train", "val", "test"]:
    #         tmp = df[df[col] == split][["ID", "smiles", col]]
    #         print(col, split, tmp.shape)
    #         tmp.rename(columns={col: "split"}, inplace=True)
    #         tmp.to_csv(p / (split + ".csv"), index=False)

    continuous_split_1D(
        start_dir=Path("sider_datasail"),
        end_dir=Path("sider_random"),
        out_dir=Path("sider_steps"),
        datatype="M",
        id_col="ID",
        value_col="smiles",
        # 0 to 100, not 0 to 1: passing fractions explores only the first
        # percent of the interpolation and yields near-identical splits.
        percentages=list(range(0, 101, 10)),
        similarity="ecfp",
        folds=None,
        plots=True,
        curve=True,
        weights=None,
        num_clusters=None,
        ranking_method="leakage",
        ranking_seed=None,
        cache_path=None,
        k_method="leakage",
        dataset=None,
    )
