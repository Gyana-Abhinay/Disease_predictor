import pandas as pd

# Load Parquet file
df = pd.read_parquet("test-00000-of-00001.parquet", engine="pyarrow")

# Save as CSV
df.to_csv("test-00000-of-00001.csv", index=False)

print("Conversion complete. CSV file saved.")