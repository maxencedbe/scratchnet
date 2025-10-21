import pandas as pd
import numpy as np

np.random.seed(42)

n = 150
X = np.random.normal(0, 1, (n, 4))
y = np.repeat([0, 1, 2], n // 3)

X[y == 0] += np.array([2, 0, 0, 0])
X[y == 1] += np.array([0, 2, 0, 0])
X[y == 2] += np.array([0, 0, 2, 0])

df = pd.DataFrame(X, columns=["sep_len", "sep_wid", "pet_len", "pet_wid"])
df["flower_class"] = y
df.to_csv("custom_datasets/flowers.csv", index=False)