import pandas as pd
import numpy as np

np.random.seed(42)

n = 500
surface = np.random.uniform(30, 200, n)
rooms = np.random.randint(1, 8, n)
distance = np.random.uniform(0, 30, n)

price = 1000 * surface + 15000 * rooms - 500 * distance + np.random.normal(0, 10000, n)

df = pd.DataFrame({
    "surface_m2": surface,
    "rooms": rooms,
    "distance_km": distance,
    "price_eur": price
})
df.to_csv("custom_datasets/houses.csv", index=False)