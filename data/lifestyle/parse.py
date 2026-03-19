import pandas as pd
import numpy as np

df = pd.read_csv("user_data.csv", dtype=str)  # read everything as string to catch weird values
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
# Define suspicious patterns
suspicious_values = {"?","??", "NaN", "nan", "NAN", "", "None", "null", "NULL"}

# Check for exact matches
mask_exact = df.isin(suspicious_values)

# Check for actual NaN values (missing)
mask_nan = df.isna()

# Combine masks
mask = mask_exact | mask_nan

# Get rows containing any suspicious value
rows_with_issues = df[mask.any(axis=1)]

print("Number of problematic rows:", len(rows_with_issues))

# Optional: show where the issues are
problem_cells = mask.stack()
print(problem_cells[problem_cells])