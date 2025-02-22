import pickle
import pandas as pd

# Suppose you have computed the class distribution as a pandas DataFrame named df_class
# For illustration, create a dummy DataFrame:
df_class = pd.DataFrame({
    "attack_cat": ["0", "1"],
    "Count": [5000, 1500],
    "dataset": ["original", "original"]
})
df_class.to_csv("class_distribution.csv", index=False)
