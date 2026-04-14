import math
from numba import cuda, float64, int32
import numba
import numpy as np
from timeit import default_timer as timer
from algo import *
import cupy as cp
import re
import ast


####################################### PREPROCESSING ##########################################

#From the dataset (matrix) return a matrix of the same size with only integer values and a dictionary provinding a mapping (and reverse mapping) between the two matrices

def embed_data_global(data):
    timer_t = timer()
    num_cols = len(data[0])
    placeholder_numeric = [0]*num_cols
    categorical_cols = []
    range_per_col=[0]*num_cols

    # Detect categorical columns
    for col in range(num_cols):
        val = data[0][col]
        if isinstance(val, str):
            categorical_cols.append(col)

    # Collect all unique values across all categorical columns
  
    mapping={}
    data_np = np.array(data)
    for col in categorical_cols:
        # np.unique restituisce già i valori ordinati
        uniques = np.unique(data_np[:, col])
        
        mapping[col] = {val: i for i, val in enumerate(uniques)}
        
        n_uniques = len(uniques)
        range_per_col[col] = n_uniques
        placeholder_numeric[col] = n_uniques + 1

    
    minus_1_col=len(data[0])-1
    if(minus_1_col in categorical_cols):
        mapping[-1]=mapping[minus_1_col]
    # Create a global mapping: str -> unique int
    #print(global_uniques)
    #print(categorical_cols)

    # Detect numeric columns
    numeric_cols = [col for col in range(num_cols) if col not in categorical_cols]

    
    # Encode the data
    timer2 = timer()
    print(f"Time for global embedding 1: {timer2 - timer_t:.4f}s")
    encoded_data,placeholder_numeric, rev_map_numeric,range_per_col = encode_data_global_with_placeholder(data,data_np,placeholder_numeric, mapping, categorical_cols,num_cols,numeric_cols,range_per_col)
    timer3 = timer()
    print(f"Time for global embedding 2: {timer3 - timer2:.4f}s")
    return encoded_data, mapping, categorical_cols, placeholder_numeric, rev_map_numeric,range_per_col

def encode_data_global_with_placeholder(data,data_np,placeholder_numeric, mapping, categorical_cols,num_cols,numeric_cols,range_per_col):


    # Find a safe placeholder for each numeric column
    map_numeric = [[]]*num_cols
    rev_map_numeric = [[]]*num_cols
    max_values_array=[0]*num_cols
    
    timer_part1 = timer()

    #print("cols float" + str(cols_float))
    for col in numeric_cols:
        column_copy = [row[col] for row in data]
        numeric_vals = [v for v in column_copy]
        unique_vals = set(numeric_vals)
        range_per_col[col]=len(unique_vals)

        sorted_unique = sorted(unique_vals)

        mapping_direct = {val: idx for idx, val in enumerate(sorted_unique)} #float -> int
        reverse_map = {idx: val for idx, val in enumerate(sorted_unique)} #int -> float
        map_numeric[col]=mapping_direct
        rev_map_numeric[col]=reverse_map
        max_val=max(reverse_map)
        placeholder_numeric[col] = max_val + 1
        max_values_array[col]=max_val

    encoded = np.empty_like(data_np, dtype=np.int64)

    timer_part2 = timer()
    print(f"encoding 2, part 1 took:{timer_part1 - timer_part2:.4f}s")

    timer_part1 = timer()
    # categorical columns
    for col in categorical_cols:
        col_data = data_np[:, col]
        encoded[:, col] = np.array([mapping[col][v] for v in col_data])

    # numeric columns
    for col in numeric_cols:
        col_data = data_np[:, col]
        encoded[:, col] = np.array([map_numeric[col][float(v)] for v in col_data])
    
    timer_part2 = timer()
    print(f"encoding 2, part 1 took:{timer_part1 - timer_part2:.4f}s")
    
    return encoded,placeholder_numeric, rev_map_numeric, range_per_col

#######################################                ##########################################
####################################### POSTPROCESSING ##########################################

#From the learned rules in the hypotesis i translate back the int values back to strings and floats

def remap_to_cat_rule(obj, categorical_cols, reverse_map, reverse_map_numeric):
    
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
        else:
            val=reverse_map_numeric[col][val]
        
        m_op=MAPPED_OPS[op]
        return (col, m_op, val)

    # Case 2: tuple (general)
    if isinstance(obj, tuple):
        #print("TUPLE calling it on \n", [x for x in obj])
        return tuple(
            remap_to_cat_rule(x, categorical_cols, reverse_map,reverse_map_numeric)
            for x in obj
        )

    # Case 3: list
    if isinstance(obj, list):
        
        #print("TUPLE calling it on \n", [x for x in obj])
        return [
            remap_to_cat_rule(x, categorical_cols, reverse_map, reverse_map_numeric)
            for x in obj
        ]

    # Case 4: anything else
    return obj


#######################################                ##########################################
#######################################  MAIN METHODS  ##########################################

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

    begin_preprocess = timer()
    embedded_data,mapping,categorical_cols,fst_unused_num, rev_map_numeric,max_range_cols=embed_data_global(data)


    #print(mapping) col & str -> int (0,1.. n)
    #i just need to keep track of the size of mapping (n) and a list of strings
    #reverse_map = {v: k for k, v in mapping.items()}

    reverse_map = {}




    minus_1_col=len(data[0])-1
    
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
    fst_unused_num_dev = cuda.to_device(np.array(fst_unused_num, dtype=np.int32))

    num_blocks=len(embedded_data_original[0])-1
    
    neg_dev = cuda.device_array(max(max_range_cols)*num_blocks, dtype=np.int32)
    pos_dev = cuda.device_array(max(max_range_cols)*num_blocks, dtype=np.int32)
    vals_dev = cuda.device_array(max(max_range_cols)*num_blocks, dtype=np.int32)
    cats_dev = cuda.device_array(max(max_range_cols)*num_blocks, dtype=np.int32)
    index_sizes_dev = cuda.device_array(2, dtype=np.int32) #e+ e-
    post_time=0
    while len(original_data_indexes) > 0:
        total_loops += 1

        start_most = timer()
        l = most_(embedded_data_original, original_data_indexes) #CPU

        #print(l)
        end_most = timer()
        overall_most += end_most - start_most
        
        start_split = timer()

        #invece degli elementi prendo gli indici
        index_e_plus, index_e_minus = split_data_by_item_(embedded_data_original, l,categorical_cols, original_data_indexes) #CPU but indexes
 
        #move
        #index_e_plus_gpu  = cp.asarray(index_e_plus)
        #index_e_minus_gpu = cp.asarray(index_e_minus)
        
        end_split = timer()
        overall_split += end_split - start_split

        start_learn = timer()
        rule,best_item, coversTime,foldTime,timeTotal,loops = learn_rule_(embedded_data_original,index_e_plus, index_e_minus , categorical_cols,categorical_cols_dev,categorical_mask_dev, fst_unused_num_dev, max_range_cols,embedded_data_original_dev,neg_dev, pos_dev, vals_dev, cats_dev, index_sizes_dev ,[], ratio)

        overall_best_item+=best_item
        overall_covers+=coversTime
        overall_fold+=foldTime
        total_time+=timeTotal
        learn_rule_loops+=loops
        end_learn = timer()
        overall_learn += end_learn - start_learn
        
        start_covers1 = timer()
        
        e_tp_index = [i for i in index_e_plus if not cover_(rule, embedded_data_original, i,categorical_cols)]
        
        #print("etp:"+str(e_tp))
        #print("etp_index:"+str(e_tp_index))

        end_covers1 = timer()
        overall_covers1 += end_covers1 - start_covers1

        if len(e_tp_index) == len(index_e_plus):
            break

        start_setop = timer()
        
        e_tn_index =  [i for i in index_e_minus if not cover_(rule, embedded_data_original, i,categorical_cols)]
        
        original_data_indexes = e_tp_index + e_tn_index

        #print("----------\n")
        #print("remaining original data indexes "+str(original_data_indexes))
        #print("remaining embedded_data "+str(embedded_data))

        end_setop = timer()

        overall_setop += end_setop - start_setop
        
        # Append rule with selected literal
        #print("to print -> " + str(rule_to_print))
        rule = l, rule[1], rule[2], rule[3]
        
        begin_post = timer()
        rule=remap_to_cat_rule((rule), categorical_cols, reverse_map, rev_map_numeric)
        end_post = timer()
        post_time = end_post - begin_post + post_time
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


    print(f"cover check: {overall_covers1:.4f}s ({100 * overall_covers1/total_time:.1f}%)")
    print(f"set op:      {overall_setop:.4f}s ({100 * overall_setop/total_time:.1f}%)")
    print(f"Total:       {total_time:.4f}s")

    print(f"Time preprocessing: {overall_preprocess:.4f}s")
    
    print(f"Time postprocessing: {post_time:.4f}s")
    return ret


def cover_(rule, embedded_data_original, i,categorical_cols):
    example_x=embedded_data_original[i]
    return evaluate_(rule, example_x, categorical_cols)

def cover_on_gpu(items_dev, embedded_data_original_dev, categorical_cols_dev,index_e_plus_dev,index_e_minus_dev,size_plus,size_minus,index_sizes_dev):
    
    #SI, TEMPORANEAMENTE SOLO CON 2 BLOCCHI, con più blocchi servono 2 kernel diversi lanciati uno dopo l'altro
    update_e_plus_min_dev[2,32](index_sizes_dev,items_dev, embedded_data_original_dev, categorical_cols_dev,index_e_plus_dev,size_plus,index_e_minus_dev,size_minus)
    host_counts = index_sizes_dev.copy_to_host()

    size_plus = int(host_counts[0])
    size_minus = int(host_counts[1])
    return size_plus,size_minus


def evaluate_(item, dataset_example, categorical_cols):

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
        else:
            if r == 0:
                return val <= v
            elif r == 1:
                return val > v
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

                else:
                    if r == 0:
                        cond = val <= v
                    elif r == 1:
                        cond = val > v
                    else:
                        cond = False

              
                if not cond:
                    return 0

            else:
                if not evaluate_(sub, dataset_example, categorical_cols):
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

                else:
                    if r == 0:
                        cond = val <= v
                    elif r == 1:
                        cond = val > v
                    else:
                        cond = False


                if cond:
                    return 0

            else:
                if evaluate_(sub, dataset_example, categorical_cols):
                    return 0
    return 1


def learn_rule_(embedded_data_original,  index_e_plus,       index_e_minus,                categorical_cols, categorical_cols_dev,categorical_mask, fst_unused_num_dev,max_range_cols,embedded_data_original_dev,neg_dev, pos_dev, vals_dev, cats_dev,index_sizes_dev ,used_items=[], ratio=0.5):
    items = []
    items_3array=[]
    learn_rule_loops = 0

    # Timing accumulators
    overall_best_item = 0
    overall_covers = 0
    overall_fold = 0
    while True:
        learn_rule_loops += 1

        # ===== best_item timing =====
        start_best_item = timer()
        
        if len(index_e_plus) != 0 or len(index_e_minus) != 0:
            index_e_plus_dev  = cuda.to_device(np.array(index_e_plus, dtype=np.int32))
            index_e_minus_dev = cuda.to_device(np.array(index_e_minus, dtype=np.int32))

        t,t_arr = best_item_gpu(index_e_plus_dev,index_e_minus_dev, embedded_data_original, index_e_plus, index_e_minus, categorical_mask, fst_unused_num_dev,max_range_cols, embedded_data_original_dev,neg_dev, pos_dev, vals_dev, cats_dev ,used_items + items)

        end_best_item = timer()
        overall_best_item += end_best_item - start_best_item 

        items.append(t)
        items_3array.append(t_arr)
        items_np = np.array(items_3array, dtype=np.float64)
        items_dev = cuda.to_device(items_np)
        rule = -1, items, [], 0
        # ===== cover timing =====
        start_cover_pos_neg = timer()
        #gets rows

        

        if(len(index_e_plus)+len(index_e_minus)>5000):
            n_valid_plus,n_valid_minus=cover_on_gpu(items_dev, embedded_data_original_dev, categorical_cols_dev,index_e_plus_dev,index_e_minus_dev,len(index_e_plus),len(index_e_minus), index_sizes_dev)
            

            # 2. Taglia l'array direttamente sulla GPU (lo slicing in Numba non copia dati)
            # e POI copia solo la parte utile sull'host
            if(n_valid_plus>0):
                index_e_plus = index_e_plus_dev[:n_valid_plus].copy_to_host().tolist()
            else:
                index_e_plus=[]
            if(n_valid_minus>0):
                index_e_minus = index_e_minus_dev[:n_valid_minus].copy_to_host().tolist()
            else:
                index_e_minus=[]
        else:
            index_e_plus = [i for i in index_e_plus if cover_(rule, embedded_data_original, i,categorical_cols)]
            index_e_minus = [i for i in index_e_minus  if cover_(rule, embedded_data_original, i,categorical_cols)]
        

        end_cover_pos_neg = timer()
        overall_covers += end_cover_pos_neg - start_cover_pos_neg

        # Check termination conditions
        if t[0] == -1 or len(index_e_minus) <= len(index_e_plus) * ratio:
            if t[0] == -1:
                rule = -1, items[:-1], [], 0

            if len(index_e_minus) > 0 and t[0] != -1:
                # ===== fold timing =====
                start_fold = timer()
                ab = fold_gpu(embedded_data_original,index_e_minus,index_e_plus ,categorical_cols,categorical_cols_dev,categorical_mask,fst_unused_num_dev,max_range_cols, embedded_data_original_dev, neg_dev, pos_dev, vals_dev, cats_dev ,index_sizes_dev,used_items + items,  ratio)
                end_fold = timer()
                overall_fold += end_fold - start_fold
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break

    # Total time for profiling
    total_time = overall_best_item + overall_covers + overall_fold
    return rule, overall_best_item, overall_covers,overall_fold,total_time,learn_rule_loops

def best_item_gpu(index_e_plus_dev,index_e_minus_dev,embedded_data_original,index_e_plus, index_e_minus,categorical_mask_dev, fst_unused_num_dev,max_range_cols, embedded_data_original_dev, pos_dev,neg_dev, vals_dev,cats_dev , used_items=[]):

    ret = -1, 0, 0
    ret_arr = [-1, 0, 0]
    if len(index_e_plus) == 0 and len(index_e_minus) == 0:
        return ret,ret_arr
    
    n = len(embedded_data_original[index_e_plus[0]]) if len(index_e_plus) > 0 else len(embedded_data_original[index_e_minus[0]]) #prende la lunghezza di una riga
    best = cp.float32(-1e20)

    

    #max_range_cols_dev   = cuda.to_device(np.array(max_range_cols, dtype=np.int32))

    # 4) Numeric placeholders -> float32 array

    n_max=max(max_range_cols)
    used_items_arr = np.zeros((len(used_items), 3), dtype=cp.float64)

    for j, (col, cmp, val) in enumerate(used_items):
        used_items_arr[j, 0] = col       # column index as float32 (or int32 if you like)
        used_items_arr[j, 1] = cmp       # comparator as float32 (or int32)
        used_items_arr[j, 2] = val       # value (float)

    used_items_dev = cuda.to_device(used_items_arr)

    thread_per_block=32
    blocks_grid=n-1
    
    return_vals_dev = cuda.device_array(3*(n-1), dtype=cp.float64)


    
    #blocchi piccoli ma è tutto intrawarp con molte colonne già scala, con poche tocca aumentare il numero di blocchi o lanciare con + th
    best_ig_dev[blocks_grid,thread_per_block](categorical_mask_dev,  embedded_data_original_dev,index_e_plus_dev, index_e_minus_dev,fst_unused_num_dev, return_vals_dev, pos_dev, neg_dev, vals_dev,cats_dev,n_max,used_items_dev)
        
        
        
        
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
        if best < ig:
            best = ig
            ret = i, r, v
            ret_arr = [i,r,v]
    return ret , ret_arr

def fold_gpu(embedded_data_original, index_e_plus, index_e_minus, categorical_cols,categorical_cols_dev,categorical_mask,placeholder_nums_dev, max_range_cols,embedded_data_original_dev,neg_dev, pos_dev, vals_dev, cats_dev ,index_sizes_dev,used_items=[], ratio=0.5):
    ret = []
    while len(index_e_plus) > 0:
        rule,_,_,_,_,_ = learn_rule_(embedded_data_original,index_e_plus, index_e_minus, categorical_cols,categorical_cols_dev, categorical_mask,placeholder_nums_dev,max_range_cols,embedded_data_original_dev,neg_dev, pos_dev, vals_dev, cats_dev ,index_sizes_dev,used_items, ratio)
        data_fn = [i for i in index_e_plus if not cover_(rule, embedded_data_original,i, categorical_cols)]
        if len(index_e_plus) == len(data_fn):
            break
        index_e_plus = data_fn
        ret.append(rule)
    return ret


                #fixed size (can be >>) #non fixed size (int arrays) #int  #fixed size array #fixed size array #non fixed size array

def split_data_by_item_(embedded_data, l,categorical_cols, original_data_indexes):
    data_pos, data_neg = [], []

    for i in original_data_indexes:
        x=embedded_data[i]
        if evaluate_(l, x,categorical_cols):
            data_pos.append(i) #lui aggiungeva righe io aggiungo INDICI DELLE COLLONE IN sorted_T
        else:
            data_neg.append(i)
    return data_pos, data_neg


def most_(data, original_data_indexes, i=-1):
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

#######################################               ##########################################
#######################################    KERNELS    ##########################################


@cuda.jit
def best_ig_dev(categorical_mask_dev, embedded_data_original_dev,index_e_plus, index_e_minus, fst_unused_num,return_vals_dev,pos, neg, unique_vals_present,unique_cats_present,n_cols,used_items=[]):
    
    xp, xn, cp, cn = 0, 0, 0, 0

    bests_sm = cuda.shared.array(shape=32, dtype=float64)
    v_sm = cuda.shared.array(shape=32, dtype=float64)
    r_sm = cuda.shared.array(shape=32, dtype=int32)
    #outer loop is on the columns
    tid = cuda.threadIdx.x
    block_id=cuda.blockIdx.x

    base_block_index=block_id*n_cols
    is_categorical = categorical_mask_dev[block_id] #leva l'offse

    for j in range(tid,n_cols,32):
        pos[base_block_index+j] = 0
        neg[base_block_index+j] = 0
        unique_vals_present[base_block_index+j] = 0
        unique_cats_present[base_block_index+j] = 0

    cuda.syncwarp()
    
    bests_sm[tid]= -1e20
    v_sm[tid] = -1e20
    r_sm[tid] = 0
    num_used = len(used_items)  
    
    #nel codice seriale controllava le cose dentro un for... almeno qua si controllano 1 volta (le colonne sono di un singolo tipo se il dataset è pulito)
    if(is_categorical):

        # --- FASE 1: Processa INDEX_E_PLUS (per cp, xp) ---
        cp = warp_process_sorted_column(
            embedded_data_original_dev, index_e_plus, 
            unique_cats_present, pos, 
            block_id * n_cols, len(index_e_plus), block_id, fst_unused_num[block_id]
        )

        # --- FASE 2: Processa INDEX_E_MINUS (per cn, xn) ---
        cn = warp_process_sorted_column(
            embedded_data_original_dev, index_e_minus, 
            unique_cats_present, neg, 
            block_id * n_cols, len(index_e_minus), block_id, fst_unused_num[block_id]
        )
        cuda.syncwarp()
        num_cats = warp_compact_indices_dev(unique_cats_present, block_id, n_cols)

        for c_i in range(tid, num_cats, 32):
            c=unique_cats_present[base_block_index+c_i]
            skip = 0  # boolean flag
            
            for j in range(num_used):
                if int(used_items[j, 0]) == block_id and int(used_items[j, 1]) in (2,3) and int(used_items[j, 2]) == c:
                    skip = 1
                    break

            if skip:
                continue
            
            c_index=c+base_block_index

            ig = gain_dev(pos[c_index], cp - pos[c_index] + xp, cn - neg[c_index] + xn, neg[c_index])
            
            if bests_sm[tid] < ig:
                bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, c, 2
            
            ig = gain_dev(cp - pos[c_index] + xp, pos[c_index], neg[c_index], cn - neg[c_index] + xn)

            if bests_sm[tid] < ig:
                bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, c, 3
                
    else:
        # --- FASE 1: Processa INDEX_E_PLUS (per cp, xp) ---
        xp = warp_process_sorted_column(
            embedded_data_original_dev, index_e_plus, 
            unique_vals_present, pos, 
            block_id * n_cols, len(index_e_plus), block_id, fst_unused_num[block_id]
        )

        # --- FASE 2: Processa INDEX_E_MINUS (per cn, xn) ---
        xn = warp_process_sorted_column(
            embedded_data_original_dev, index_e_minus, 
            unique_vals_present, neg, 
            block_id * n_cols, len(index_e_minus), block_id, fst_unused_num[block_id]
        )

        cuda.syncwarp()
        num_vals = warp_compact_indices_dev(unique_vals_present, block_id, n_cols)

        mask = 0xffffffff
        carry_p = 0.0
        carry_n = 0.0
        off_base = block_id * n_cols
        for chunk_start in range(0, num_vals, 32):
            position_index = chunk_start + tid
            
            # This tells shfl_up_sync exactly which threads are providing data
            active_mask = cuda.ballot_sync(mask, position_index < num_vals)
            
            # 2. Indirect Load 
            if(position_index < num_vals):
            
                # Use a ternary to pick the index; inactive threads just point to base
                load_idx = off_base + unique_vals_present[off_base + position_index] 
                
                p_val = pos[load_idx] 
                n_val = neg[load_idx] 

                # intra-Warp Scan 
                
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

    
    
    
        for x_i in range(tid, num_vals, 32):
            x=unique_vals_present[base_block_index+x_i]
            skip = 0  # boolean flag
            for j in range(num_used):

                if int(used_items[j, 0]) == block_id and int(used_items[j, 1]) in (0,1) and int(used_items[j, 2]) == x:
                    skip = 1
                    break

            if skip:
                continue
            x_index=base_block_index+x
            ig = gain_dev(pos[x_index], xp - pos[x_index] + cp, xn - neg[x_index] + cn, neg[x_index]) #su gpu, tempi assurdi causa data transfer

            if bests_sm[tid] < ig:
                bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, x, 0
            
            ig = gain_dev(xp - pos[x_index], pos[x_index] + cp, neg[x_index] + cn, xn - neg[x_index])

            
            if bests_sm[tid] < ig:
                bests_sm[tid],  v_sm[tid],  r_sm[tid] = ig, x, 1
   

    # Sincronizziamo: tutti i thread devono aver finito di marcare unique_...
    cuda.syncwarp()
 
    
    best = bests_sm[tid]
    v    = v_sm[tid]
    r    = r_sm[tid]
    offset = 16
    
    while offset > 0:
        other_best = cuda.shfl_down_sync(mask, best, offset)
        other_v    = cuda.shfl_down_sync(mask, v, offset)
        other_r    = cuda.shfl_down_sync(mask, r, offset)

        if other_best > best or (other_best==best and other_v<v):
            best = other_best
            v = other_v
            r = other_r

        offset //= 2


    if(tid==0):
        return_vals_dev[cuda.blockIdx.x * 3 + 0] = best
        return_vals_dev[cuda.blockIdx.x * 3 + 1] = r
        return_vals_dev[cuda.blockIdx.x * 3 + 2] = v


@cuda.jit(device=True)
def gain_dev(tp, fn, tn, fp):
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
    
    return math.floor(ret / 1e-8) * 1e-8
    
@cuda.jit(device=True)
def warp_compact_indices_dev(unique_vals_present, offset, n_cols):
    tid = cuda.threadIdx.x
    off_base = offset * n_cols
    mask = 0xffffffff
    
    #il num totale di el trovati
    total_found = 0
    
    #chunk da 32
    for chunk_start in range(0, n_cols, 32):
        j = chunk_start + tid
        
        #ogni thread controlla il suo elemento 
        is_present = False
        if j < n_cols:
            is_present = (unique_vals_present[off_base + j] == 1)
        
        ballot = cuda.ballot_sync(mask, is_present)
        
        #l'indice di destinazione LOCALE al chunk e aggiungiamo il totale precedente
        #Quanti '1' ci sono prima di me in QUESTO chunk + quanti ne abbiamo trovati PRIMA
        lower_mask = (1 << tid) - 1
        dest_idx = total_found + cuda.popc(ballot & lower_mask)
        
        
        cuda.syncwarp()
        
        #scrittura (Scatter)
        if is_present:
            unique_vals_present[off_base + dest_idx] = j
            
        total_found += cuda.popc(ballot)
        
        cuda.syncwarp()

    return total_found


@cuda.jit(device=True)
def warp_reduce_sum_dev(val):
    #Sum across all threads in a warp
    mask = 0xffffffff
    for offset in (16, 8, 4, 2, 1):
        val += cuda.shfl_down_sync(mask, val, offset)
    return cuda.shfl_sync(mask, val, 0)

@cuda.jit(device=True)
def warp_process_sorted_column(original_data, index_list,
                               uniques, counts_hist,
                               off_base, len_indices, col_idx,
                               unused_val):
    tid = cuda.threadIdx.x

    global_count = 0
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
        else:
            d=unused_val
            
        if active:
            same_mask = cuda.match_any_sync(active_mask, d)
            count = cuda.popc(same_mask)
            mask_before = same_mask & ((1 << tid) - 1)

            # First thread has no bits set before it
            is_first = mask_before == 0
            if(is_first):
                uniques[off_base + d] = 1
                global_count += count
                counts_hist[off_base + d] += count #pos/neg
        cuda.syncwarp()

    return warp_reduce_sum_dev(global_count)

@cuda.jit(device=True)
def evaluate_dev(items, dataset_example, categorical_cols):
    n_items = items.shape[0]
    if n_items == 0:
        return 0

    for idx in range(n_items):
        i = int(items[idx, 0])
        r = items[idx, 1]
        v = items[idx, 2]

        val = dataset_example[i]
        is_categorical = False
        for c_idx in range(len(categorical_cols)):
            if categorical_cols[c_idx] == i:
                is_categorical = True
                break

        if is_categorical:
            if r == 2:
                cond = val == v
            elif r == 3:
                cond = val != v
            else:
                cond = False
        else:
            if r == 0:
                cond = val <= v
            elif r == 1:
                cond = val > v
            else:
                cond = False

        if not cond:
            return 0

    return 1

#molto temporanamente solo con due blocchi, con più blocchi servono 2 lanci di kernel diversi
@cuda.jit
def update_e_plus_min_dev(index_sizes,items, embedded_data_original, categorical_cols,index_pos,len_index_pos,index_neg,len_index_neg):
    #molto temporanamente solo con due blocchi, con più blocchi servono 2 lanci di kernel diversi    
    tid = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    total_found = 0
    if(block_id==0):
        for chunk_start in range(0, len_index_pos, 32):
            pos_in_list = chunk_start + tid
            mask = 0xffffffff
            active = pos_in_list < len_index_pos
            active_mask = cuda.ballot_sync(mask, active)
            # Load value or placeholder
            remove = -1
            i=-1
            if active:
                remove=0
                i=index_pos[pos_in_list]
                covered=evaluate_dev(items,embedded_data_original[i],categorical_cols)
                if(not covered):
                    remove=1
            ballot = cuda.ballot_sync(active_mask, remove==0)
            lower_mask = (1 << tid) - 1
            dest_idx = total_found + cuda.popc(ballot & lower_mask)
            cuda.syncwarp()

            if(remove==0): #keep
                index_pos[dest_idx]=i
            total_found += cuda.popc(ballot)
            cuda.syncwarp()
    else:
        for chunk_start in range(0, len_index_neg, 32):
            pos_in_list = chunk_start + tid
            mask = 0xffffffff
            active = pos_in_list < len_index_neg
            active_mask = cuda.ballot_sync(mask, active)
            # Load value or placeholder
            remove = -1
            i=-1
            if active:
                remove=0
                i=index_neg[pos_in_list]
                covered=evaluate_dev(items,embedded_data_original[i],categorical_cols)
                if(not covered):
                    remove=1
            ballot = cuda.ballot_sync(active_mask, remove==0)
            lower_mask = (1 << tid) - 1
            dest_idx = total_found + cuda.popc(ballot & lower_mask)
            cuda.syncwarp()

            if(remove==0): #keep
                index_neg[dest_idx]=i
            total_found += cuda.popc(ballot)
            cuda.syncwarp()
    if(tid==0):
        index_sizes[block_id]=total_found