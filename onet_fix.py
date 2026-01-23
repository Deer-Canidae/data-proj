#!/usr/bin/env -S uv run

import pandas as pd

df_input = pd.read_csv(
    "./data/onet_classification/Onet_association.csv",
    sep=";",
    encoding="utf-8-sig",
    low_memory=False,
)

df_output = pd.DataFrame()

df_output["commodity_code"] = df_input["Commodity Code"]
df_output["example"] = df_input["Example"]

df_output.to_csv(
    "./data/onet_classification/Onet_association_fixed.csv",
    sep=";",
    index=False,
    encoding="utf-8",
)
