import pandas as pd
import numpy as np
import os

os.makedirs("Dataset", exist_ok=True)
np.random.seed(42)
n_samples = 1500

# Generate Historical/Weather parameters
data = {
    "Annual_Rainfall": np.random.uniform(1200, 4000, n_samples),
    "Cloud_Visibility": np.random.uniform(1.0, 10.0, n_samples),
    "Seasonal_Rainfall": np.random.uniform(300, 1500, n_samples)
}

df = pd.DataFrame(data)
# Logic: High seasonal rainfall combined with low cloud visibility equals flood event (1)
df["Flood_Occurred"] = ((df["Seasonal_Rainfall"] * 0.7 + df["Annual_Rainfall"] * 0.3) / df["Cloud_Visibility"] > 280).astype(int)

# Split and save into the structure datasets
df.to_csv("Dataset/Historical_Flood_Data.csv", index=False)
df[["Annual_Rainfall", "Seasonal_Rainfall"]].to_csv("Dataset/Rainfall_Data.csv", index=False)
df[["Cloud_Visibility"]].to_csv("Dataset/Weather_Data.csv", index=False)

print("Datasets generated successfully inside 'Dataset/' folder!")