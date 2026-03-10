import math
from numba import cuda, float64, int32
import numba
import numpy as np
from timeit import default_timer as timer
from algo import *
import cupy as cp
import cupy as cupy #yes to unify
import re
import ast
global_kernel_call=0
def cover_gpu_dev(rule, embedded_data_original, i,categorical_cols, placeholder_nums):
    example_x=embedded_data_original[i]
    return evaluate_gpu(rule, example_x, categorical_cols, placeholder_nums)

def preprocess_gpu(data):
    """
    Preprocess without transposing: 
    - Sort each column
    - Create reverse index per column
    Returns:
        sorted_data: sorted values per column (same shape as data)
        reverse_index: mapping from original row -> sorted position (same shape)
    """
    cpData = cp.array(data, dtype=cp.int32)  # move to GPU

    # --------------------------------------------------
    # 1) Argsort each column (axis=0)
    # --------------------------------------------------
    perm = cp.argsort(cpData, axis=0)  # shape (rows, cols)
    
    # --------------------------------------------------
    # 2) Gather sorted values per column
    # --------------------------------------------------
    sorted_data = cp.take_along_axis(cpData, perm, axis=0)  # same shape as data

    # --------------------------------------------------
    # 3) Create reverse index per column
    # reverse_index[row, col] = index in sorted column
    # --------------------------------------------------
    reverse_index = cp.argsort(perm, axis=0)  # same shape as data

    return sorted_data, reverse_index
def embed_data_global(data):
    
    num_cols = len(data[0])
    categorical_cols = []
    range_per_col=[0]*num_cols
    # Detect categorical columns
    for col in range(num_cols):
        for row in data:
            val = row[col]
            if isinstance(val, str) and val != '?':
                categorical_cols.append(col)
                break
    #print("Categorical columns:", categorical_cols)

    # Collect all unique values across all categorical columns
  
    mapping={}
    for col in categorical_cols:
        col_uniques = set()
        for row in data:
            val = row[col]
            col_uniques.add(val)
            mapping[col]={val: i for i, val in enumerate(sorted(col_uniques))}
        range_per_col[col]=len(col_uniques)

    

    minus_1_col=len(data[0])-1
    if(minus_1_col in categorical_cols):
        mapping[-1]=mapping[minus_1_col]
    # Create a global mapping: str -> unique int
    #print(global_uniques)
    #print(categorical_cols)

    # Detect numeric columns
    numeric_cols = [col for col in range(num_cols) if col not in categorical_cols]

    

    # Encode the data
    encoded_data,placeholder_numeric, rev_map_numeric,cols_float,range_per_col = encode_data_global_with_placeholder(data, mapping, categorical_cols,num_cols,numeric_cols,range_per_col)

    return encoded_data, mapping, categorical_cols, placeholder_numeric, rev_map_numeric,cols_float,range_per_col


def encode_data_global_with_placeholder(data, mapping, categorical_cols,num_cols,numeric_cols,range_per_col):
    encoded = []

    # Find a safe placeholder for each numeric column
    placeholder_numeric = [0]*num_cols
    map_numeric = [[]]*num_cols
    rev_map_numeric = [[]]*num_cols
    max_values_array=[0]*num_cols
    cols_float=[]
    
    for col in numeric_cols:
        for row in data:
            val = row[col]

            # Skip missing
            if val == '?':
                continue

            # Strict float check (exclude ints)
            if not float(val).is_integer() and not col in cols_float:
                cols_float.append(col)

    #print("cols float" + str(cols_float))
    for col in numeric_cols:
        if(col in cols_float):
            """
            data: list of rows
            col: column index

            Returns:
            mapping: dict {float_value -> int_rank}
            reverse_map: list where reverse_map[rank] = float_value
            """

            # 1) Copy column (without modifying dataset)
            column_copy = [row[col] for row in data]

            # 2) Keep only numeric values (ignore '?')
            numeric_vals = [
                v for v in column_copy
                if v != '?'
            ]

            unique_vals = set(numeric_vals)
            range_per_col[col]=len(unique_vals)

            sorted_unique = sorted(unique_vals)
            #print(f"sorted_col {col}: {str(sorted_unique)}")

            mapping_float = {val: idx for idx, val in enumerate(sorted_unique)} #float -> int
            reverse_map_float = {idx: val for idx, val in enumerate(sorted_unique)} #int -> float
            map_numeric[col]=mapping_float
            rev_map_numeric[col]=reverse_map_float
            #print("mapping_float"+str(mapping_float))
            max_val=max(reverse_map_float)
            placeholder_numeric[col] = max_val + 1
            max_values_array[col]=max_val
            
        else:
            column_copy = [row[col] for row in data]
            numeric_vals = [v for v in column_copy if v != '?']
            unique_vals = set(numeric_vals)
            range_per_col[col]=len(unique_vals)
            sorted_unique = sorted(unique_vals)
            mapping_int = {val: idx for idx, val in enumerate(sorted_unique)} #float -> int
            reverse_map_int = {idx: val for idx, val in enumerate(sorted_unique)} #int -> float
            map_numeric[col]=mapping_int
            rev_map_numeric[col]=reverse_map_int
            max_val=max(reverse_map_int)
            placeholder_numeric[col] = max_val + 1
            max_values_array[col]=max_val


    for row in data:
        new_row = list(row)
        for col in range(len(row)):
            val = new_row[col]
            if col in categorical_cols:
                new_row[col] = mapping[col][val]
            else:
                # aggiungi float mapping
                if val == '?':
                    new_row[col] = placeholder_numeric[col]
                else:
                    new_row[col] = map_numeric[col][val]

        encoded.append(new_row)

    #print("Encoded data:", encoded)
    #print("Placeholders for '?':", placeholder_numeric)
    #print(encoded)
    return encoded,placeholder_numeric, rev_map_numeric,cols_float, range_per_col

def foldrmGPU(data, ratio=0.5):
    ret = []
    # Accumulators for timings
    overall_most = 0
    overall_split = 0
    overall_learn = 0
    overall_covers1 = 0
    overall_setop = 0
    total_loops = 0
    overall_best_item=0
    overall_covers=0 
    overall_fold = 0 
    total_time = 0 
    learn_rule_loops = 0
    reverse_index_T=[]

    embedded_data,mapping,categorical_cols,placeholder_nums, rev_map_numeric,float_cols,max_range_cols=embed_data_global(data)

    #print(mapping) col & str -> int (0,1.. n)
    #i just need to keep track of the size of mapping (n) and a list of strings
    #reverse_map = {v: k for k, v in mapping.items()}

    reverse_map = {}




    minus_1_col=len(data[0])-1
    
    print("categorical_cols")
    print(categorical_cols)
    if(minus_1_col in categorical_cols):
        categorical_cols.append(-1)
        categorical_cols.remove(minus_1_col)

    # Initialize an array of zeros
    categorical_mask = np.zeros(len(data[0]), dtype=int)

    # Set the specified indices to 1
    categorical_mask[categorical_cols] = 1
    categorical_mask_dev = cuda.to_device(np.array(categorical_mask, dtype=np.int32))
    
    
    for key, value in mapping.items():
        reverse_map[key] = {}
        for key_, value_ in value.items():
            reverse_map[key][value_] = key_ #array of strings

    #print("categorical cols"+str(categorical_cols))
    #print("last col"+str(minus_1_col))
    begin_preprocess = timer()

    original_sorted_dev, reverse_index_T_dev = preprocess_gpu(embedded_data)
    
    cp.cuda.Stream.null.synchronize()

    end_preprocess = timer()
    overall_preprocess = end_preprocess - begin_preprocess
    #orignal_training_data=data
    original_data_indexes = list(range(len(data)))
    embedded_data_original=embedded_data
    embedded_data_original_dev = cuda.to_device(
        np.array(embedded_data_original, dtype=np.int32)  # or np.int32 if ints
    )

    categorical_cols_dev = cuda.to_device(np.array(categorical_cols, dtype=np.int32))
    placeholder_nums_dev = cuda.to_device(np.array(placeholder_nums, dtype=np.int32))

    while len(original_data_indexes) > 0:
        total_loops += 1

        start_most = timer()
        l = most_gpu(embedded_data_original, original_data_indexes) #CPU

        #print(l)
        end_most = timer()
        overall_most += end_most - start_most
        
        start_split = timer()

        #invece degli elementi prendo gli indici
        index_e_plus, index_e_minus = split_data_by_item_gpu_dev(embedded_data_original, l,categorical_cols,placeholder_nums, original_data_indexes) #CPU but indexes
 
        #move
        #index_e_plus_gpu  = cp.asarray(index_e_plus)
        #index_e_minus_gpu = cp.asarray(index_e_minus)
        
        end_split = timer()
        overall_split += end_split - start_split

        start_learn = timer()
        rule,best_item, coversTime,foldTime,timeTotal,loops = learn_rule_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus , reverse_index_T,categorical_cols,categorical_cols_dev,categorical_mask_dev, placeholder_nums,placeholder_nums_dev, max_range_cols,embedded_data_original_dev,[], ratio)
        '''                                                   
        print("type emmbedded data "+str(type(embedded_data_original)))
        print("index e plus "+str(type(index_e_plus)))
        print("reverse_index_T "+str(type(reverse_index_T)))
        print("reverse map"+ str(type(reverse_map)))
        print("categorical_cols "+str(type(categorical_cols)))
        print("placeholder_nums "+str(type(placeholder_nums))) 
        '''
        overall_best_item+=best_item
        overall_covers+=coversTime
        overall_fold+=foldTime
        total_time+=timeTotal
        learn_rule_loops+=loops
        end_learn = timer()
        overall_learn += end_learn - start_learn
        
        start_covers1 = timer()
        
        e_tp_index = [i for i in index_e_plus if not cover_gpu_dev(rule, embedded_data_original, i,categorical_cols, placeholder_nums)]
        
        #print("etp:"+str(e_tp))
        #print("etp_index:"+str(e_tp_index))

        end_covers1 = timer()
        overall_covers1 += end_covers1 - start_covers1

        if len(e_tp_index) == len(index_e_plus):
            break

        start_setop = timer()
        
        e_tn_index =  [i for i in index_e_minus if not cover_gpu_dev(rule, embedded_data_original, i,categorical_cols, placeholder_nums)]
        
        original_data_indexes = e_tp_index + e_tn_index

        #print("----------\n")
        #print("remaining original data indexes "+str(original_data_indexes))
        #print("remaining embedded_data "+str(embedded_data))

        end_setop = timer()

        overall_setop += end_setop - start_setop
        
        # Append rule with selected literal
        #print("to print -> " + str(rule_to_print))
        rule = l, rule[1], rule[2], rule[3]
        rule=remap_to_cat_rule((rule), categorical_cols, reverse_map, placeholder_nums, rev_map_numeric,float_cols)
        ret.append(rule)
        
        #print("rule -> "+str(rule))

    # Total time spent
    #print("all rules")
    #print(ret)
    total_time = overall_most + overall_split + overall_learn + overall_covers1 + overall_setop

    print(f"Timing summary after {total_loops} loops:")
    print(f"most:        {overall_most:.4f}s ({100 * overall_most/total_time:.1f}%)")
    print(f"split_data:  {overall_split:.4f}s ({100 * overall_split/total_time:.1f}%)")
    print(f"learn_rule:  {overall_learn:.4f}s ({100 * overall_learn/total_time:.1f}%)")
    
    print(f"----learn_rule summary after {learn_rule_loops} loops:")
    print(f"----best_item: {overall_best_item:.4f}s ({100 * overall_best_item/total_time:.1f}%)")
    print(f"----cover:     {overall_covers:.4f}s ({100 * overall_covers/total_time:.1f}%)")
    print(f"----fold:      {overall_fold:.4f}s ({100 * overall_fold/total_time:.1f}%)")
    print(f"----Total:     {total_time:.4f}s")

    print(f"cover check: {overall_covers1:.4f}s ({100 * overall_covers1/total_time:.1f}%)")
    print(f"set op:      {overall_setop:.4f}s ({100 * overall_setop/total_time:.1f}%)")
    print(f"Total:       {total_time:.4f}s")

    print(f"Time preprocessing: {overall_preprocess:.4f}s")
    return ret


def remap_to_cat_rule(obj, categorical_cols, reverse_map,placeholder_nums, reverse_map_numeric, float_cols):
    
    #print("ramapping on obj" + str(obj))
    # Case 1: literal (INT, OP, VALUE)
    VALID_OPS=[0,1,2,3] # <= > == !=
    MAPPED_OPS=['<=','>','==','!=']
    if (
        isinstance(obj, tuple)
        and len(obj) == 3
        and isinstance(obj[0], int)
        and isinstance(obj[1], int)
        and obj[1] in VALID_OPS
    ):
        col, op, val = obj
        #print("base canse \n")
        if col in categorical_cols:
            val = reverse_map[col][val]
        elif (val == placeholder_nums[col]):
            val = '?'
        else:
            val=reverse_map_numeric[col][val]
        
        m_op=MAPPED_OPS[op]
        return (col, m_op, val)

    # Case 2: tuple (general)
    if isinstance(obj, tuple):
        #print("TUPLE calling it on \n", [x for x in obj])
        return tuple(
            remap_to_cat_rule(x, categorical_cols, reverse_map,placeholder_nums,reverse_map_numeric, float_cols)
            for x in obj
        )

    # Case 3: list
    if isinstance(obj, list):
        
        #print("TUPLE calling it on \n", [x for x in obj])
        return [
            remap_to_cat_rule(x, categorical_cols, reverse_map,placeholder_nums, reverse_map_numeric, float_cols)
            for x in obj
        ]

    # Case 4: anything else
    return obj
                    #fixed size            #can be fixed size   #can be fixed size  #can be fixed size   #fixed size       #fixed size      #can be fixed size                                       
def learn_rule_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,   index_e_plus,       index_e_minus,       rev_index,          categorical_cols, categorical_cols_dev,categorical_mask, placeholder_nums, placeholder_nums_dev,max_range_cols,embedded_data_original_dev,used_items=[], ratio=0.5):
    items = []
    learn_rule_loops = 0

    # Timing accumulators
    overall_best_item = 0
    overall_covers = 0
    overall_fold = 0

    while True:
        learn_rule_loops += 1

        # ===== best_item timing =====
        start_best_item = timer()
        #print("*****************************\n")
        #print("POS: "+str(data_pos))
        #print("----------------\n")
        #print("\n NEG: "+str(data_neg))
        #print("*****************************\n")
        
        t = best_item_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus,categorical_cols_dev, categorical_mask, placeholder_nums_dev,max_range_cols, embedded_data_original_dev,used_items + items)
        end_best_item = timer()
        overall_best_item += end_best_item - start_best_item 

        items.append(t)
        rule = -1, items, [], 0

        # ===== cover timing =====
        start_cover_pos_neg = timer()
        #gets rows

        
        index_e_plus = [i for i in index_e_plus if cover_gpu_dev(rule, embedded_data_original, i,categorical_cols,placeholder_nums)]
        index_e_minus = [i for i in index_e_minus  if cover_gpu_dev(rule, embedded_data_original, i,categorical_cols, placeholder_nums)]
        
        end_cover_pos_neg = timer()
        overall_covers += end_cover_pos_neg - start_cover_pos_neg

        # Check termination conditions
        if t[0] == -1 or len(index_e_minus) <= len(index_e_plus) * ratio:
            if t[0] == -1:
                rule = -1, items[:-1], [], 0

            if len(index_e_minus) > 0 and t[0] != -1:
                # ===== fold timing =====
                start_fold = timer()
                #SISTEMA
                ab = fold_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_minus,index_e_plus ,rev_index,categorical_cols,categorical_cols_dev,categorical_mask,placeholder_nums,placeholder_nums_dev,max_range_cols, embedded_data_original_dev, used_items + items,  ratio)
                end_fold = timer()
                overall_fold += end_fold - start_fold
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break

    # Total time for profiling
    total_time = overall_best_item + overall_covers + overall_fold
    #print("returned rule: " +str(rule))
    return rule, overall_best_item, overall_covers,overall_fold,total_time,learn_rule_loops

def best_item_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus,categorical_cols_dev,categorical_mask_dev, placeholder_nums_dev,max_range_cols, embedded_data_original_dev, used_items=[]):

    ret = -1, 0, 0
    if len(index_e_plus) == 0 and len(index_e_minus) == 0:
        return ret
    
    n = len(embedded_data_original[index_e_plus[0]]) if len(index_e_plus) > 0 else len(embedded_data_original[index_e_minus[0]]) #prende la lunghezza di una riga
    best = cp.float32(-1e20)


    n_max=max(max_range_cols)
    
    # 2) 1D lists of indices -> int32 arrays
    index_e_plus_dev  = cuda.to_device(np.array(index_e_plus, dtype=np.int32))
    index_e_minus_dev = cuda.to_device(np.array(index_e_minus, dtype=np.int32))

    #max_range_cols_dev   = cuda.to_device(np.array(max_range_cols, dtype=np.int32))

    # 4) Numeric placeholders -> float32 array


    used_items_arr = np.zeros((len(used_items), 3), dtype=cupy.float64)

    for j, (col, cmp, val) in enumerate(used_items):
        used_items_arr[j, 0] = col       # column index as float32 (or int32 if you like)
        used_items_arr[j, 1] = cmp       # comparator as float32 (or int32)
        used_items_arr[j, 2] = val       # value (float)

    used_items_dev = cuda.to_device(used_items_arr)

    thread_per_block=32

    blocks_grid=n-1
    
    return_vals_dev = cuda.device_array(3*(n-1), dtype=cupy.float64)

    global global_kernel_call
    #3333000 il num totale di esempi 4+ |pos+neg|=330000 + |cats+vals|=330000 
    neg_dev = cuda.device_array(n_max*(n-1), dtype=np.int32)
    pos_dev = cuda.device_array(n_max*(n-1), dtype=np.int32)
    vals_dev = cuda.device_array(n_max*(n-1), dtype=np.int32)
    cats_dev = cuda.device_array(n_max*(n-1), dtype=np.int32)
    aux_sorted_dev = cuda.device_array(n_max, dtype=np.int32)
    #crea [len,len,len,len, pos, neg, vals,cats]

    
    #for each example check a literal providing the most IG 

        
    best_ig_gpu[blocks_grid,thread_per_block](categorical_mask_dev, aux_sorted_dev,  embedded_data_original_dev,original_sorted_dev, reverse_index_T_dev,index_e_plus_dev, index_e_minus_dev, categorical_cols_dev,placeholder_nums_dev, global_kernel_call,return_vals_dev, pos_dev, neg_dev, vals_dev,cats_dev,n_max,used_items_dev)
        
        
    global_kernel_call+=1
        
        
        
    cuda.synchronize()

        #print("ig1 on host: ",host_arr[0], "ig2 on host: ", host_arr[1])
    return_vals_host = return_vals_dev.copy_to_host()  # returns a NumPy array
    n_triplets = len(return_vals_host) // 3

    for i in range(n_triplets):
        idx = i * 3
        ig = return_vals_host[idx]
        r  = return_vals_host[idx + 1]
        v  = return_vals_host[idx + 2]
        
        v=int(v) if v != -1e20 else v
        r=int(r)
        # Do something with your values
        if best < ig:
            best = ig
            ret = i, r, v

    return ret

def fold_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus, rev_index,categorical_cols,categorical_cols_dev,categorical_mask,placeholder_nums,placeholder_nums_dev, max_range_cols,embedded_data_original_dev,used_items=[], ratio=0.5):
    ret = []
    while len(index_e_plus) > 0:
        #print("fold gpu")
        rule,_,_,_,_,_ = learn_rule_gpu(embedded_data_original,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus, rev_index,categorical_cols,categorical_cols_dev, categorical_mask,placeholder_nums,placeholder_nums_dev,max_range_cols,embedded_data_original_dev,used_items, ratio)
        data_fn = [i for i in index_e_plus if not cover_gpu_dev(rule, embedded_data_original,i, categorical_cols, placeholder_nums)]
        if len(index_e_plus) == len(data_fn):
            break
        index_e_plus = data_fn
        ret.append(rule)
    return ret


                #fixed size (can be >>) #non fixed size (int arrays) #int  #fixed size array #fixed size array #non fixed size array
#@cuda.jit           # copy once                 #   merge                          # copy once          copy once         copy once

@cuda.jit
def best_ig_gpu(categorical_mask_dev, aux_sorted, embedded_data_original_dev,original_sorted_dev, reverse_index_T_dev,index_e_plus, index_e_minus, categorical_cols, placeholder_nums, kenerl_num,return_vals_dev,pos, neg, unique_vals_present,unique_cats_present,n_cols,used_items=[]):
    
    xp, xn, cp, cn = 0, 0, 0, 0

    bests_sm = cuda.shared.array(shape=32, dtype=float64)
    v_sm = cuda.shared.array(shape=32, dtype=float64)
    r_sm = cuda.shared.array(shape=32, dtype=int32)
    #outer loop is on the columns
    tid = cuda.threadIdx.x
    offset=cuda.blockIdx.x
    i=offset
    
    is_categorical = categorical_mask_dev[offset] #leva l'offse
    for j in range(tid,n_cols,32):
        pos[offset*n_cols+j] = 0
        neg[offset*n_cols+j] = 0
        unique_vals_present[offset*n_cols+j] = 0
        unique_cats_present[offset*n_cols+j] = 0
        #print("thread ",tid, "block index: ",cuda.blockIdx.x,"setting index: ",offset*n_cols+j)

    #for j in range(tid,index_e_plus,32):
    #    index=reverse_index_T_dev[index, ]
    #    aux_sorted[j]= 


    #formalmente non ok ---------------------------------------------------
    #elements_per_th = (max_range_cols + 32 - 1) // 32
    # ... (Codice precedente: reset pos/neg/unique con tid loop) ...
    cuda.syncwarp()

    # --- FASE 1: Processa INDEX_E_PLUS (per cp, xp) ---
    cp, xp = warp_process_sorted_column(
        embedded_data_original_dev, index_e_plus, 
        unique_cats_present, unique_vals_present, pos, 
        offset * n_cols, len(index_e_plus), i, 
        is_categorical, placeholder_nums[i]
    )

    # --- FASE 2: Processa INDEX_E_MINUS (per cn, xn) ---
    # Nota: passiamo 'neg' invece di 'pos', ma gli 'unique' sono gli stessi
    cn, xn = warp_process_sorted_column(
        embedded_data_original_dev, index_e_minus, 
        unique_cats_present, unique_vals_present, neg, 
        offset * n_cols, len(index_e_minus), i, 
        is_categorical, placeholder_nums[i]
    )

    # Sincronizziamo: tutti i thread devono aver finito di marcare unique_...
    cuda.syncwarp()

    #end of formalmente non ok ---------------------------------------------------
    #------

    num_vals = warp_compact_indices(unique_vals_present, offset, n_cols)
            
    # Now unique_vals_present[:num_vals] contains the indices of present values

    num_cats = warp_compact_indices(unique_cats_present, offset, n_cols)

    mask = 0xffffffff
    carry_p = 0.0
    carry_n = 0.0
    off_base = offset * n_cols

    for chunk_start in range(0, num_vals, 32):
        position_index = chunk_start + tid
        
        # 1. Create the active mask for this chunk
        # This tells shfl_up_sync exactly which threads are providing data
        active_mask = cuda.ballot_sync(mask, position_index < num_vals)
        
        # 2. Indirect Load 
        # We use a 'safe' index (0) for inactive threads but set their values to 0.0
        if(position_index < num_vals):
        
            # Use a ternary to pick the index; inactive threads just point to base
            # This prevents illegal memory access while keeping execution linear
            load_idx = off_base + unique_vals_present[off_base + position_index] 
            
            p_val = pos[load_idx] 
            n_val = neg[load_idx] 

            # 3. Intra-Warp Scan (Kogge-Stone) 
            # Using active_mask ensures synchronization only among threads with work
            shift = 1
            while shift < 32:
                p_left = cuda.shfl_up_sync(active_mask, p_val, shift)
                n_left = cuda.shfl_up_sync(active_mask, n_val, shift)
                if tid >= shift:
                    p_val += p_left
                    n_val += n_left
                shift *= 2

            # 4. Add the carry from the PREVIOUS chunk
            p_val += carry_p
            n_val += carry_n

            pos[load_idx] = p_val
            neg[load_idx] = n_val

        last_thread_in_chunk = min(31, (num_vals - chunk_start) - 1)
        
        # Broadcast the total sum from that specific last active thread
        carry_p = cuda.shfl_sync(mask, p_val, last_thread_in_chunk)
        carry_n = cuda.shfl_sync(mask, n_val, last_thread_in_chunk)

   
   
    bests_sm[tid]= -1e20
    v_sm[tid] = -1e20
    r_sm[tid] = 0
    num_used = len(used_items)  
    for x_i in range(tid, num_vals, 32):
        x=unique_vals_present[offset*n_cols+x_i]
        skip = 0  # boolean flag
        for j in range(num_used):

            if int(used_items[j, 0]) == i and int(used_items[j, 1]) in (0,1) and int(used_items[j, 2]) == x:
                skip = 1
                break

        if skip:
            continue
        x_index=offset*n_cols+x
        ig = gain_device(pos[x_index], xp - pos[x_index] + cp, xn - neg[x_index] + cn, neg[x_index]) #su gpu, tempi assurdi causa data transfer

        if bests_sm[tid] < ig:
            bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, x, 0
        
        ig = gain_device(xp - pos[x_index], pos[x_index] + cp, neg[x_index] + cn, xn - neg[x_index])

        
        if bests_sm[tid] < ig:
            bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, x, 1

    for c_i in range(tid, num_cats, 32):
        c=unique_cats_present[offset*n_cols+c_i]
        skip = 0  # boolean flag
        
        for j in range(num_used):
            if int(used_items[j, 0]) == i and int(used_items[j, 1]) in (2,3) and int(used_items[j, 2]) == c:
                skip = 1
                break

        if skip:
            continue
        
        c_index=c+offset*n_cols

        ig = gain_device(pos[c_index], cp - pos[c_index] + xp, cn - neg[c_index] + xn, neg[c_index])
        
        if bests_sm[tid] < ig:
            bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, c, 2
        
        ig = gain_device(cp - pos[c_index] + xp, pos[c_index], neg[c_index], cn - neg[c_index] + xn)

        if bests_sm[tid] < ig:
            bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, c, 3
            
    #cuda.syncthreads()

    #shuffle
    mask = 0xffffffff
    best = bests_sm[tid]
    v    = v_sm[tid]
    r    = r_sm[tid]

    offset = 16
    while offset > 0:
        other_best = cuda.shfl_down_sync(mask, best, offset)
        other_v    = cuda.shfl_down_sync(mask, v, offset)
        other_r    = cuda.shfl_down_sync(mask, r, offset)

        if other_best > best:
            best = other_best
            v = other_v
            r = other_r

        offset //= 2


    if(tid==0):
        return_vals_dev[cuda.blockIdx.x * 3 + 0] = best
        return_vals_dev[cuda.blockIdx.x * 3 + 1] = r
        return_vals_dev[cuda.blockIdx.x * 3 + 2] = v


# GPU gain function
@cuda.jit(device=True)
def gain_device(tp, fn, tn, fp):
    # Force double precision
    tp = float(tp)
    fn = float(fn)
    tn = float(tn)
    fp = float(fp)

    if tp + tn < fp + fn:
        result=-1e20
        #print("returning early ")
        return result

    tot_p = tp + fp
    tot_n = tn + fn
    tot = tot_p + tot_n
    ret = 0.0  # float64 by default

    if tp > 0.0:
        ret += tp / tot * math.log(tp / tot_p)
    if fp > 0.0:
        ret += fp / tot * math.log(fp / tot_p)
    if tn > 0.0:
        ret += tn / tot * math.log(tn / tot_n)
    if fn > 0.0:
        ret += fn / tot * math.log(fn / tot_n)
    
    return math.floor(ret / 1e-10) * 1e-10
    
@cuda.jit(device=True)
def warp_compact_indices(unique_vals_present, offset, n_cols):
    tid = cuda.threadIdx.x
    off_base = offset * n_cols
    mask = 0xffffffff
    
    # Questo manterrà il numero totale di elementi trovati (il vecchio num_vals)
    total_found = 0
    
    # Cicliamo su n_cols in chunk da 32
    for chunk_start in range(0, n_cols, 32):
        j = chunk_start + tid
        
        # 1. Ogni thread controlla il suo elemento nel chunk attuale
        is_present = False
        if j < n_cols:
            is_present = (unique_vals_present[off_base + j] == 1)
        
        # 2. Otteniamo la maschera dei bit per questo chunk
        ballot = cuda.ballot_sync(mask, is_present)
        
        # 3. Calcoliamo l'indice di destinazione LOCALE al chunk e aggiungiamo il totale precedente
        # (Quanti '1' ci sono prima di me in QUESTO chunk + quanti ne abbiamo trovati PRIMA)
        lower_mask = (1 << tid) - 1
        dest_idx = total_found + cuda.popc(ballot & lower_mask)
        
        # Salviamo il valore di j prima di rischiare di sovrascrivere l'array
        val_to_store = j
        
        # Sincronizziamo il warp per assicurarci che tutti abbiano letto 
        # i valori corretti prima di iniziare a scrivere
        cuda.syncwarp()
        
        # 4. Scrittura (Scatter)
        if is_present:
            unique_vals_present[off_base + dest_idx] = val_to_store
            
        # 5. Aggiorniamo il totale cumulativo per il prossimo chunk
        # Ogni thread nel warp aggiorna il suo 'total_found' con il numero di bit nel ballot
        total_found += cuda.popc(ballot)
        
        # Sincronizziamo di nuovo prima del prossimo chunk per evitare race conditions
        # sulla memoria globale 'unique_vals_present'
        cuda.syncwarp()

    return total_found


@cuda.jit(device=True)
def warp_reduce_sum(val):
    """Sum across all threads in a warp"""
    mask = 0xffffffff
    for offset in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(mask, val, offset)
    return cuda.shfl_sync(mask, val, 0)

@cuda.jit(device=True)
def warp_process_sorted_column(original_data, index_list,
                               unique_cats, unique_vals, counts_hist,
                               off_base, len_indices, col_idx,
                               is_categorical, placeholder_val):
    tid = cuda.threadIdx.x

    d_sm = cuda.shared.array(32, dtype=int32)
    c_count = 0
    x_count = 0

    # Process in warp-sized chunks
    for chunk_start in range(0, len_indices, 32):
        mask = 0xffffffff
        pos_in_list = chunk_start + tid
        active = pos_in_list < len_indices
        active_mask = cuda.ballot_sync(mask, active)

        # Load value or placeholder
        d = -1
        if active:
            idx = index_list[pos_in_list]
            d = original_data[idx, col_idx]
            d_sm[tid]=d
        else:
            d_sm[tid] = 2147483647
        # --- Bitonic sort for 32 threads ---
        # Bitonic sort inside a warp (32 threads)
        if(tid==0):
            for i in range(1, 32):
                key = d_sm[i]
                j = i - 1
                while j >= 0 and d_sm[j] > key:
                    d_sm[j + 1] = d_sm[j]
                    j -= 1
                d_sm[j + 1] = key
        
        cuda.syncwarp()
        if active:
            d = d_sm[tid]
            same_mask = cuda.match_any_sync(active_mask, d)
            count = cuda.popc(same_mask)
            mask_before = same_mask & ((1 << tid) - 1)

            # First thread has no bits set before it
            is_first = mask_before == 0
            if(is_first and d!=2147483647):
                if is_categorical or d == placeholder_val:
                    unique_cats[off_base + d] = 1
                    c_count += count
                else:
                    unique_vals[off_base + d] = 1
                    x_count += count
                counts_hist[off_base + d] += count #pos/neg
        cuda.syncwarp()

    return warp_reduce_sum(c_count), warp_reduce_sum(x_count)

def split_data_by_item_gpu_dev(embedded_data, l,categorical_cols,placeholder_nums, original_data_indexes):
    data_pos, data_neg = [], []

    for i in original_data_indexes:
        x=embedded_data[i]
        if evaluate_gpu(l, x,categorical_cols, placeholder_nums):
            data_pos.append(i) #lui aggiungeva righe io aggiungo INDICI DELLE COLLONE IN sorted_T
        else:
            data_neg.append(i)
    return data_pos, data_neg


#literal/col and row of the dataset
def evaluate_gpu(item, dataset_example, categorical_cols, placeholder_nums):

    if len(item) == 0:
        return 0  # automatically false

    # -------------------------
    # Simple literal case
    # -------------------------
    if len(item) == 3:
        i, r, v = item
        val = dataset_example[i]

        if i in categorical_cols:
            if r == 2:
                return val == v
            elif r == 3:
                return val != v
            else:
                return False

        elif val != placeholder_nums[i]:
            if r == 0:
                return val <= v
            elif r == 1:
                return val > v
            else:
                return False

        else:  # val == placeholder
            if r == 2:
                return val == v
            elif r == 3:
                return val != v
            else:
                return False

    # -------------------------
    # Complex rule case
    # -------------------------
    # item structure assumed:
    # [?, positive_literals, negative_literals, flag]

    # If flag == 0 → conjunction must hold
    if item[3] == 0 and len(item[1]) > 0:
        for sub in item[1]:
            if len(sub) == 3:
                i, r, v = sub
                val = dataset_example[i]

                if i in categorical_cols:
                    if r == 2:
                        cond = val == v
                    elif r == 3:
                        cond = val != v
                    else:
                        cond = False

                elif val != placeholder_nums[i]:
                    if r == 0:
                        cond = val <= v
                    elif r == 1:
                        cond = val > v
                    else:
                        cond = False

                else:
                    if r == 2:
                        cond = val == v
                    elif r == 3:
                        cond = val != v
                    else:
                        cond = False

                if not cond:
                    return 0

            else:
                if not evaluate_gpu(sub, dataset_example, categorical_cols, placeholder_nums):
                    return 0

    # Negative literals (any must NOT hold)
    if len(item[2]) > 0:
        for sub in item[2]:
            if len(sub) == 3:
                i, r, v = sub
                val = dataset_example[i]

                if i in categorical_cols:
                    if r == 2:
                        cond = val == v
                    elif r == 3:
                        cond = val != v
                    else:
                        cond = False

                elif val != placeholder_nums[i]:
                    if r == 0:
                        cond = val <= v
                    elif r == 1:
                        cond = val > v
                    else:
                        cond = False

                else:
                    if r == 2:
                        cond = val == v
                    elif r == 3:
                        cond = val != v
                    else:
                        cond = False

                if cond:
                    return 0

            else:
                if evaluate_gpu(sub, dataset_example, categorical_cols, placeholder_nums):
                    return 0
    return 1

def most_gpu(data, original_data_indexes, i=-1):
    tab = dict()
    
    for row_index in original_data_indexes:
        d=data[row_index]
        if d[i] not in tab:
            tab[d[i]] = 0
        tab[d[i]] += 1
    
    y, n = 0, 0
    for t in tab:
        if n <= tab[t]:
            y, n = t, tab[t]
    return i, 2, y
