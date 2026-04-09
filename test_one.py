import numpy as np
from numba import cuda

# -----------------------------
# Trivial GPU evaluation function
# -----------------------------
@cuda.jit(device=True)
def evaluate_dev(items, embedded, categorical_cols):
    return embedded[0] > 0  # Simple test: covered if > 0

# -----------------------------
# Kernels (from your code)
# -----------------------------
@cuda.jit
def generate_mask_and_count_manual_idx(items, embedded_data_original, categorical_cols, index, len_index, total_count_out):
    tid_in_block = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    gw = cuda.gridDim.x
    tid = bx * 32 + tid_in_block
    stride = gw * 32
    local_count = 0
    for i in range(tid, len_index, stride):
        idx_val = index[i]
        print("Thread", tid, "processing index", idx_val, "value accessed:", embedded_data_original[idx_val][0])
        if evaluate_dev(items, embedded_data_original[idx_val], categorical_cols):
            local_count += 1
            print("Thread", tid, "covers index", embedded_data_original[idx_val][0])
        else:
            index[i] = -1
    warp_count = local_count
    warp_count += cuda.shfl_down_sync(0xffffffff, warp_count, 16)
    warp_count += cuda.shfl_down_sync(0xffffffff, warp_count, 8)
    warp_count += cuda.shfl_down_sync(0xffffffff, warp_count, 4)
    warp_count += cuda.shfl_down_sync(0xffffffff, warp_count, 2)
    warp_count += cuda.shfl_down_sync(0xffffffff, warp_count, 1)
    if tid_in_block == 0:
        total_count_out[bx] = warp_count

@cuda.jit
def compact_warps_kernel(index, index_size, prefix_sum_counts, out_compacted, index_sizes_dev, last):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    tid = bx * 32 + tx
    out_compacted[tid]=-1
    last_index = index_sizes_dev[last]
    val = -1
    if tid < index_size and tid <= last_index:
        val = index[tid]
    is_valid = (val != -1)
    mask = cuda.ballot_sync(0xffffffff, is_valid)
    if is_valid:
        warp_base_offset = prefix_sum_counts[bx]
        lower_mask = (1 << tx) - 1
        local_offset = cuda.popc(mask & lower_mask)
        out_compacted[warp_base_offset + local_offset] = val
    
@cuda.jit
def warp_prefix_sum_inplace(arr, total_out,max_per_block, index_write):
    tid = cuda.threadIdx.x
    n = arr.shape[0]
    print("length of arr:", n)
    local_sum = 0
    local_last = -1
    for i in range(tid, n, 32):
        val = arr[i]
        max=max_per_block[i]
        local_sum += val
        if val != -1:
            local_last = max
    scan_val = local_sum
    for delta in (1, 2, 4, 8, 16):
        tmp = cuda.shfl_up_sync(0xffffffff, scan_val, delta)
        if tid >= delta:
            scan_val += tmp
    base = scan_val - local_sum
    running = base
    for i in range(tid, n, 32):
        val = arr[i]
        arr[i] = running
        running += val
    last_idx = local_last
    print("Thread", tid, "local last index:", local_last)
    # Warp-wide max reduction
    for offset in [16, 8, 4, 2, 1]:
        tmp = cuda.shfl_down_sync(0xffffffff, last_idx, offset)
        last_idx = max(last_idx, tmp)

    if tid == 31:
        total_out[index_write] = scan_val
    if tid == 0:
        total_out[index_write + 2] = last_idx

# -----------------------------
# GPU pipeline with intermediate prints
# -----------------------------
def cover_on_gpu_debug(items_dev, embedded_data_original_dev, categorical_cols_dev,
                       index_e_plus_dev, index_e_minus_dev, out_compacted_minus, out_compacted_plus,
                       size_plus, size_minus, index_sizes_dev,max_per_block):
    
    blocks_no_minus = (size_minus + 31) // 32
    blocks_no_plus = (size_plus + 31) // 32
    
    block_counts_device_plus = cuda.device_array(blocks_no_minus, dtype=np.int32)
    block_counts_device_minus = cuda.device_array(blocks_no_plus, dtype=np.int32)

    # -----------------------------
    # Generate mask + counts
    # -----------------------------

    print("index_e_minus:", index_e_minus_dev.copy_to_host())
    generate_mask_and_count_manual_idx[blocks_no_minus, 32](items_dev, embedded_data_original_dev, categorical_cols_dev, index_e_minus_dev, size_minus, block_counts_device_minus
    )
    cuda.synchronize()

    print("After generate_mask_and_count_manual_idx:")
    print("index_e_minus: (DA SOMMARE)", index_e_minus_dev.copy_to_host())
    print("block_counts_device_minus:", block_counts_device_minus.copy_to_host())

    # -----------------------------
    # Warp prefix sum
    # -----------------------------
    warp_prefix_sum_inplace[1, 32](block_counts_device_minus, index_sizes_dev, max_per_block,0)
    cuda.synchronize()

    print("After warp_prefix_sum_inplace:")
    print("index_sizes_dev:", index_sizes_dev.copy_to_host())

    # -----------------------------
    # Compact warps
    # -----------------------------
    compact_warps_kernel[blocks_no_minus, 32](
        index_e_minus_dev, size_minus, block_counts_device_minus, out_compacted_minus, index_sizes_dev, 0
    )
    cuda.synchronize()

    print("After compact_warps_kernel:")
    print("out_compacted_minus:", out_compacted_minus.copy_to_host())

    # -----------------------------
    # Return final sizes
    # -----------------------------
    host_counts = index_sizes_dev.copy_to_host()
    size_minus = int(host_counts[0])
    size_plus = int(host_counts[1])
    return size_plus, size_minus

# -----------------------------
# Minimal test shell
# -----------------------------
if __name__ == "__main__":
    items = np.array([], dtype=np.float32)
    embedded_data_original = np.array([
    [1], [-10], [2], [3], [-10], [-10], [4], [5],
    [-10], [6], [-10], [7], [8], [-10], [-10], [9],
    [10], [-10], [11], [-10], [-10], [12], [13], [-10],
    [14], [-10], [15], [-10], [-10], [16], [17], [-10],
    [18], [-10], [19], [-10], [20], [-10], [21], [-10]
    ], dtype=np.float32)
    categorical_cols = np.zeros((3,), dtype=np.int32)
    
    index_e_plus = np.array(list(range(40)), dtype=np.int32)
    index_e_minus = np.array(list(range(40)), dtype=np.int32)
    
    out_compacted_plus = cuda.device_array(64, dtype=np.int32)
    out_compacted_minus = cuda.device_array(64, dtype=np.int32)
    index_sizes = cuda.device_array(4, dtype=np.int32)

    max_per_block = cuda.device_array(4, dtype=np.int32)
    
    items_dev = cuda.to_device(items)
    embedded_dev = cuda.to_device(embedded_data_original)
    categorical_dev = cuda.to_device(categorical_cols)
    index_plus_dev = cuda.to_device(index_e_plus)
    index_minus_dev = cuda.to_device(index_e_minus)
    
    size_plus, size_minus = cover_on_gpu_debug(
        items_dev, embedded_dev, categorical_dev,
        index_plus_dev, index_minus_dev,
        out_compacted_minus, out_compacted_plus,
        size_plus=40, size_minus=40,
        index_sizes_dev=index_sizes, max_per_block=max_per_block
    )
    
    print("Final Result sizes:", size_minus)