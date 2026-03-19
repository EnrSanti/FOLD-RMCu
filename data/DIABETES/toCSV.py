import pandas as pd

# Load feature data
X = pd.read_csv("secom.data", sep="\s+", header=None)

# Load labels
y = pd.read_csv("secom_labels.data", sep="\s+", header=None)
y.columns = ["label", "date"]

# Load feature names
with open("secom.names") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Assign feature names
X.columns = feature_names

# Merge
df = pd.concat([X, y], axis=1)

# Save CSV
df.to_csv("secom_full.csv", index=False)

print("CSV created: secom_full.csv")
