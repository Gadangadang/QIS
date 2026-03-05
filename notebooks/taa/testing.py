import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib

# Add project root to path
sys.path.append(os.path.abspath('../../'))

import core.taa.features.price
import core.taa.features.relative
import core.taa.features.pipeline

importlib.reload(core.taa.features.price)
importlib.reload(core.taa.features.relative)
importlib.reload(core.taa.features.pipeline)

from core.taa.features.pipeline import FeaturePipeline


def convert_df(df):
    # 1. Clean up headers: Extract "Close", "High", etc., from the first row
    sub_headers = df.iloc[0].tolist()
    
    # 2. Identify the ticker name from the original column headers (e.g., "ACWI-US")
    ticker_name = df.columns[1].split('.')[0]
    
    # 3. Remove the first two rows (which contain the sub-header labels and the "date" text)
    df_clean = df.iloc[2:].copy()
    
    # 4. Set the column names temporarily to match sub_headers
    # The first column index 0 is 'date'
    sub_headers[0] = 'date'
    df_clean.columns = sub_headers
    
    # 5. Format 'date' column and set it as the index
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.set_index('date')
    
    # 6. Ensure numeric columns are floats
    df_clean = df_clean.apply(pd.to_numeric)
    
    # 7. Create a MultiIndex for columns with the ticker at the top level
    df_clean.columns = pd.MultiIndex.from_product([[ticker_name], df_clean.columns])
    
    # Display the result
    print(df_clean)
    return df_clean



pipeline = FeaturePipeline(use_factset=True)

print("read inn data")
assets_df = convert_df(pd.read_csv("assets.csv"))
bench_df = convert_df(pd.read_csv("bench.csv"))
print("data loaded")
print(assets_df.head(), bench_df.head())
print("run feats")

pipeline.gen_feats(assets_df, bench_df)
#print("feats ran")
