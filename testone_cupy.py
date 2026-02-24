import cupy as cp

# --------------------------------------------------
# 1) Create example data
# --------------------------------------------------
X = cp.array([
    [1.2, 9.0],
    [1.3, 2.0],
    [1.1, 5.0],
    [1.2, 1.0],
    [5.4, 7.0],
    [1.2, 3.0],
    [3.5, 8.0],
    [3.2, 4.0],
], dtype=cp.float32)

rows, cols = X.shape

print("Original:")
print(X)

# --------------------------------------------------
# 2) Transpose to make rows contiguous in memory
# --------------------------------------------------
XT = X.T  # shape (cols, rows)

# --------------------------------------------------
# 3) Argsort each row
# --------------------------------------------------
perm_T = cp.argsort(XT, axis=1)
sorted_XT = cp.take_along_axis(XT, perm_T, axis=1)

# --------------------------------------------------
# 4) Compute reverse index (original -> sorted)
# --------------------------------------------------
reverse_index_T = cp.empty_like(perm_T)
cols_T, rows_T = XT.shape
for j in range(cols_T):
    reverse_index_T[j, perm_T[j]] = cp.arange(rows_T)

# --------------------------------------------------
# 6) Print results
# --------------------------------------------------
print("\nSorted (by columns):")
print(sorted_XT)

print("\nReverse index:")
print(reverse_index_T)

# Verify condition: sorted[reverse_index[i,j], j] == X[i,j]

col_indices = cp.arange(cols_T)[:, None]  # shape (cols, 1)
reconstructed_XT = sorted_XT[cp.arange(cols_T)[:, None], reverse_index_T]

print("\nReconstructed (should equal original):")
print(reconstructed_XT)