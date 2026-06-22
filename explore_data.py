import pandas as pd

df = pd.read_csv("BBCNews.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("one document:")
print(df.iloc[0])

print(df.iloc[0]["descr"])

df["char_count"] = df["descr"].str.len()

print(df["char_count"].describe())