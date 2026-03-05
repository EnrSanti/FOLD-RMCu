import math
from numba import cuda, float64
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
    cpData = cp.array(data, dtype=cp.float32)

    dataT = cpData.T  # shape (cols, rows)
    #print(dataT)
    #print("\n")
    # --------------------------------------------------
    # 3) Argsort each row
    # --------------------------------------------------
    perm_T = cp.argsort(dataT, axis=1)
    sorted_T = cp.take_along_axis(dataT, perm_T, axis=1)
    #print(sorted_T)

    reverse_index_T = cp.empty_like(perm_T)
    cols_T, rows_T = dataT.shape
    reverse_index_T = cp.argsort(perm_T, axis=1)
    #for j in range(cols_T):
    #    reverse_index_T[j, perm_T[j]] = cp.arange(rows_T)

    #print("\nSorted (by columns):")
    #print(sorted_T)

    #print("\nReverse index:")
    #print(reverse_index_T)

    return sorted_T, reverse_index_T

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
    if(minus_1_col in categorical_cols):
        categorical_cols.append(-1)
    
    
    for key, value in mapping.items():
        reverse_map[key] = {}
        for key_, value_ in value.items():
            reverse_map[key][value_] = key_ #array of strings

    #print("categorical cols"+str(categorical_cols))
    #print("last col"+str(minus_1_col))
    original_sorted_T_dev, reverse_index_T_dev = preprocess_gpu(embedded_data)
    #orignal_training_data=data
    original_data_indexes = list(range(len(data)))
    embedded_data_original=embedded_data

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
        rule,best_item, coversTime,foldTime,timeTotal,loops = learn_rule_gpu(embedded_data_original,index_e_plus, index_e_minus , reverse_index_T,categorical_cols, placeholder_nums, max_range_cols,[], ratio)
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
def learn_rule_gpu(embedded_data_original,   index_e_plus,       index_e_minus,       rev_index,          categorical_cols, placeholder_nums, max_range_cols,used_items=[], ratio=0.5):
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
        
        t = best_item_gpu(embedded_data_original,index_e_plus, index_e_minus,categorical_cols, placeholder_nums,max_range_cols, used_items + items)
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
                ab = fold_gpu(embedded_data_original,index_e_minus,index_e_plus ,rev_index,categorical_cols,placeholder_nums,max_range_cols, used_items + items, ratio)
                end_fold = timer()
                overall_fold += end_fold - start_fold
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break

    # Total time for profiling
    total_time = overall_best_item + overall_covers + overall_fold
    #print("returned rule: " +str(rule))
    return rule, overall_best_item, overall_covers,overall_fold,total_time,learn_rule_loops

def best_item_gpu(embedded_data_original,index_e_plus, index_e_minus,categorical_cols, placeholder_nums,max_range_cols, used_items=[]):

    ret = -1, 0, 0
    if len(index_e_plus) == 0 and len(index_e_minus) == 0:
        return ret
    
    n = len(embedded_data_original[index_e_plus[0]]) if len(index_e_plus) > 0 else len(embedded_data_original[index_e_minus[0]]) #prende la lunghezza di una riga
    best = cp.float32(-1e20)
    
    #for each example check a literal providing the most IG
    for i in range(n - 1): # 0..n-1 # form 20 to 500 blocks
        # launch 1 block
        thread_per_block=1
        blocks_grid=1
        # 1) 2D list of data -> 2D float32 array
        embedded_data_original_dev = cuda.to_device(
            np.array(embedded_data_original, dtype=np.int32)  # or np.int32 if ints
        )

        # 2) 1D lists of indices -> int32 arrays
        index_e_plus_dev  = cuda.to_device(np.array(index_e_plus, dtype=np.int32))
        index_e_minus_dev = cuda.to_device(np.array(index_e_minus, dtype=np.int32))

        # 3) 1D lists of column indices -> int32 arrays
        categorical_cols_dev = cuda.to_device(np.array(categorical_cols, dtype=np.int32))
        max_range_cols_dev   = cuda.to_device(np.array(max_range_cols, dtype=np.int32))

        # 4) Numeric placeholders -> float32 array
        placeholder_nums_dev = cuda.to_device(np.array(placeholder_nums, dtype=np.int32))


        used_items_arr = np.zeros((len(used_items), 3), dtype=cupy.float64)

        for j, (col, cmp, val) in enumerate(used_items):
            used_items_arr[j, 0] = col       # column index as float32 (or int32 if you like)
            used_items_arr[j, 1] = cmp       # comparator as float32 (or int32)
            used_items_arr[j, 2] = val       # value (float)

        used_items_dev = cuda.to_device(used_items_arr)


        return_vals_dev = cuda.device_array(3, dtype=cupy.float64)


        neg_dev = cuda.device_array(33000, dtype=np.int32)
        pos_dev = cuda.device_array(33000, dtype=np.int32)
        vals_dev = cuda.device_array(33000, dtype=np.int32)
        cats_dev = cuda.device_array(33000, dtype=np.int32)
        host_arr = np.zeros([3], dtype=cupy.float64)
        global global_kernel_call
        global_kernel_call+=1
        #print("KER CALL HOST ---------------------- ",global_kernel_call)
        host_arr[0]=global_kernel_call
        #print("used items", used_items_arr)
        device_arr = cuda.to_device(host_arr)
        host_arr_num = np.zeros([3], dtype=cupy.float32)
        host_arr_num[0]=global_kernel_call
        device_arr_num = cuda.to_device(host_arr_num)
        best_ig_gpu[blocks_grid,thread_per_block](embedded_data_original_dev,index_e_plus_dev, index_e_minus_dev, i,categorical_cols_dev,placeholder_nums_dev, max_range_cols_dev,return_vals_dev, pos_dev, neg_dev, vals_dev,cats_dev, device_arr,device_arr_num,used_items_dev )
        #print(best_ig_gpu.inspect_types())
        cuda.synchronize()
        host_arr = device_arr.copy_to_host()  # if ig is a device array
        #print("ig1 on host: ",host_arr[0], "ig2 on host: ", host_arr[1])
        return_vals_host = return_vals_dev.copy_to_host()  # returns a NumPy array
        ig, r, v = return_vals_host
        #print("kernel called ", str([ig,r,v]))
        
        v=int(v) if v != -1e20 else v
        r=int(r)
        #print("KER OVER ---------------------- ",global_kernel_call)
        #sync and get res
        if best < ig:
            best = ig
            ret = i, r, v
    #print("best item"+ str(ret))
    return ret

def fold_gpu(embedded_data_original,index_e_plus, index_e_minus, rev_index,categorical_cols,placeholder_nums, max_range_cols,used_items=[], ratio=0.5):
    ret = []
    while len(index_e_plus) > 0:
        #print("fold gpu")
        rule,_,_,_,_,_ = learn_rule_gpu(embedded_data_original,index_e_plus, index_e_minus, rev_index,categorical_cols, placeholder_nums,max_range_cols,used_items, ratio)
        data_fn = [i for i in index_e_plus if not cover_gpu_dev(rule, embedded_data_original,i, categorical_cols, placeholder_nums)]
        if len(index_e_plus) == len(data_fn):
            break
        index_e_plus = data_fn
        ret.append(rule)
    return ret


                #fixed size (can be >>) #non fixed size (int arrays) #int  #fixed size array #fixed size array #non fixed size array
#@cuda.jit           # copy once                 #   merge                          # copy once          copy once         copy once

@cuda.jit
def best_ig_gpu(embedded_data_original,index_e_plus, index_e_minus, i, categorical_cols, placeholder_nums, max_range_cols,return_vals_dev,pos, neg, unique_vals_present,unique_cats_present, device_arr,device_arr_num,used_items=[]):
    
    xp, xn, cp, cn = 0, 0, 0, 0

    #pos, neg = [0]*(max_range_cols[i]+1),[0]*(max_range_cols[i]+1)
    #unique_vals_present, unique_cats_present = [0]*(max_range_cols[i]+1),[0]*(max_range_cols[i]+1)

    #outer loop is on the columns
    #print("on GPUK ------------------------------", device_arr_num[0])
    for j in range(33000):
        pos[j] = 0
        neg[j] = 0
        unique_vals_present[j] = 0
        unique_cats_present[j] = 0
    '''   
    if(len(index_e_plus)==0):
        print("index_e plus is empty")
    else:
        print("index_e plus not empty ")
    '''
    for index in index_e_plus: #loop per example (row)
        d=embedded_data_original[index] #get 1 row

        #d[i] is the value of a cell in column i
        pos[d[i]] += 1 

        is_categorical = False
        for j in range(categorical_cols.size):
            if i == categorical_cols[j]:
                is_categorical = True
                break

        if is_categorical or d[i]==placeholder_nums[i]: #se is 
            unique_cats_present[(d[i])]=1 #add to the unique cat values found
            cp += 1
        else: 
            unique_vals_present[(d[i])]=1 #add to the unique num values found
            xp += 1

    '''
    if(len(index_e_minus)==0):
        print("index_e_minus is empty")
    else:
        print("index_e_minus not empty ")
    '''
    for index in index_e_minus:
        d=embedded_data_original[index]
        neg[d[i]] += 1

        is_categorical = False
        for j in range(categorical_cols.size):
            if i == categorical_cols[j]:
                is_categorical = True
                break

        if is_categorical or d[i]==placeholder_nums[i]:
            unique_cats_present[(d[i])]=1
            cn += 1
        else:
            unique_vals_present[(d[i])]=1
            xn += 1

    
    num_vals = 0
    for j in range(unique_vals_present.size):
        if unique_vals_present[j] == 1:
            unique_vals_present[num_vals] = j  # overwrite array with only present indices
            num_vals += 1
    # Now unique_vals_present[:num_vals] contains the indices of present values

    num_cats = 0

    #print("range",unique_cats_present.size)
    for j in range(unique_cats_present.size):
        if unique_cats_present[j] == 1:
            unique_cats_present[num_cats] = j
            num_cats += 1
    '''
    print("unique_vals_present len (loop size first)", num_vals)
    print("unique_cats_present len (loop size snd)", num_cats)
    '''
    #--------------------------
    for j in range(1, num_vals):
        pos[unique_vals_present[j]] += pos[unique_vals_present[j - 1]]
        neg[unique_vals_present[j]] += neg[unique_vals_present[j - 1]]

    best, v, r = cupy.float64(-1e20), cupy.float64(-1e20), 0
    
    #print("type:", type(best))
    num_used = len(used_items)  
    for x_i in range(0, num_vals):
        x=unique_vals_present[x_i]
        #print("considering val: ",x)
        skip = 0  # boolean flag
        for j in range(num_used):

            if int(used_items[j, 0]) == i and int(used_items[j, 1]) == 0 and int(used_items[j, 2]) == x:
                skip = 1
            if int(used_items[j, 0]) == i and int(used_items[j, 1]) == 1 and int(used_items[j, 2]) == x:
                skip = 1

        if skip:
            #print("skipping")
            continue
        ig = gain_device(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x]) #su gpu, tempi assurdi causa data transfer

        if best < ig:
            #print("in 1 (adding x and 0) ",x)
            best, v, r = ig, x, 0
        
        ig = gain_device(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x])

        #print("type:", type(ig))
        
        if best < ig:
            #print("in 2 (adding x and 1) ",x)
            best, v, r = ig, x, 1
    #print("num used of items: " ,num_used)
    for c_i in range(0, num_cats):
        c=unique_cats_present[c_i]
        #print("considering cat: ",c)
        skip = 0  # boolean flag
        for j in range(num_used):
            if int(used_items[j, 0]) == i and int(used_items[j, 1]) == 2 and int(used_items[j, 2]) == c:
                skip = 1
                break
            if int(used_items[j, 0]) == i and int(used_items[j, 1]) == 3 and int(used_items[j, 2]) == c:
                skip = 1
                break

        if skip:
            #print("skipping")
            continue
        ig = gain_device(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
        

        #print("type:", type(ig))
        device_arr[0]=ig
        if best < ig:
            #print("in 1 (adding c and 2) ",c)
            best, v, r = ig, c, 2
        ig = gain_device(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn)

        #print("type:", type(ig))
        device_arr[1]=ig
        if best < ig:
            #print("in 2 (adding c and 3) ",c)
            best, v, r = ig, c, 3

    #print("type:", type(best))
    return_vals_dev[0] = best
    return_vals_dev[1] = r
    return_vals_dev[2] = v
    #print("end of GPUK ------------------------------")

# GPU gain function
@cuda.jit(device=True)
def gain_device(tp, fn, tn, fp):
    # Force double precision
    tp = float(tp)
    fn = float(fn)
    tn = float(tn)
    fp = float(fp)

    if tp + tn < fp + fn:
        result=cp.float64(-1e20)
        #print("returning early ")
        return result

    tot_p = tp + fp
    tot_n = tn + fn
    tot = tot_p + tot_n
    ret = cp.float64(0.0)  # float64 by default

    if tp > 0.0:
        ret += tp / tot * math.log(tp / tot_p)
    if fp > 0.0:
        ret += fp / tot * math.log(fp / tot_p)
    if tn > 0.0:
        ret += tn / tot * math.log(tn / tot_n)
    if fn > 0.0:
        ret += fn / tot * math.log(fn / tot_n)
    
    return math.floor(ret / 1e-12) * 1e-12
    

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
