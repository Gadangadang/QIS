import pandas as pd

df = pd.read_csv('Dataset/gc_futures_2000_2025.csv')
df.columns = df.columns.str.strip()
print('Before:', df.columns.tolist())
print('Length:', len(df.columns))

if 'Price' in df.columns and len(df.columns) > 6:
    df = df.drop(columns=['Price'])
    print('Dropped Price column')

print('After:', df.columns.tolist())
print(df[df['Date']=='2020-01-02'][['Date','Close','Volume']])
