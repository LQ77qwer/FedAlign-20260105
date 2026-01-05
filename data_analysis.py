import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("default_pre.csv")

# 假设列名就是 LIMIT_BAL（或你实际用的名字）
scaler = MinMaxScaler(feature_range=(0, 1))
df["LIMIT_BAL"] = scaler.fit_transform(df[["LIMIT_BAL"]])

df.to_csv("default_pre_limit_nonneg.csv", index=False)
