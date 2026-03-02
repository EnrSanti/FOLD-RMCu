import math
from numba import cuda
import numpy as np
from timeit import default_timer as timer
from algo import *
import cupy as cp
import cupy as cupy #yes to unify
import re
import ast

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

    # Detect categorical columns
    for col in range(num_cols):
        for row in data:
            val = row[col]
            if isinstance(val, str) and val != '?':
                categorical_cols.append(col)
                break
    #print("Categorical columns:", categorical_cols)

    # Collect all unique values across all categorical columns
    global_uniques = set()
    for row in data:
        for col in categorical_cols:
            val = row[col]
            global_uniques.add(val)

    # Create a global mapping: str -> unique int
    #print(global_uniques)
    #print(categorical_cols)
    mapping = {val: i for i, val in enumerate(sorted(global_uniques))}

    # Detect numeric columns
    numeric_cols = [col for col in range(num_cols) if col not in categorical_cols]

    # Find a safe placeholder for each numeric column
    placeholder_numeric = [0.0]*num_cols
    for col in numeric_cols:
        numeric_values = [v for row in data for v in [row[col]] if v != '?' and isinstance(v, (int, float))]
        if numeric_values:
            max_val = max(numeric_values)
            placeholder_numeric[col] = max_val + 1.0  # safe placeholder
        else:
            placeholder_numeric[col] = 1.0  # fallback if all missing

    # Encode the data
    encoded_data = encode_data_global_with_placeholder(data, mapping, categorical_cols, placeholder_numeric)

    return encoded_data, mapping, categorical_cols, placeholder_numeric


def encode_data_global_with_placeholder(data, mapping, categorical_cols, placeholder_numeric):
    encoded = []

    for row in data:
        new_row = list(row)
        for col in range(len(row)):
            val = new_row[col]
            if col in categorical_cols:
                new_row[col] = mapping[val]
            else:
                if val == '?':
                    new_row[col] = placeholder_numeric[col]
        encoded.append(new_row)

    #print("Encoded data:", encoded)
    #print("Placeholders for '?':", placeholder_numeric)
    return encoded

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

    embedded_data,mapping,categorical_cols,placeholder_nums=embed_data_global(data)

    #print(mapping) str -> int (0,1.. n)
    #i just need to keep track of the size of mapping (n) and a list of strings
    #reverse_map = {v: k for k, v in mapping.items()}

    size = len(mapping)
    reverse_map = [None] * size

    for key, value in mapping.items():
        reverse_map[value] = key #array of strings




    minus_1_col=len(data[0])-1
    if(minus_1_col in categorical_cols):
        categorical_cols.append(-1)
        
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
        rule,best_item, coversTime,foldTime,timeTotal,loops = learn_rule_gpu(embedded_data_original,index_e_plus, index_e_minus , reverse_index_T,categorical_cols, placeholder_nums, [], ratio)
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
        rule=remap_to_cat_rule((rule), categorical_cols, reverse_map, placeholder_nums)
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


def remap_to_cat_rule(obj, categorical_cols, reverse_map,placeholder_nums):
    
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
            val = reverse_map[val]
        elif (val == placeholder_nums[col]):
            val = '?'
        m_op=MAPPED_OPS[op]
        return (col, m_op, val)

    # Case 2: tuple (general)
    if isinstance(obj, tuple):
        #print("TUPLE calling it on \n", [x for x in obj])
        return tuple(
            remap_to_cat_rule(x, categorical_cols, reverse_map,placeholder_nums)
            for x in obj
        )

    # Case 3: list
    if isinstance(obj, list):
        
        #print("TUPLE calling it on \n", [x for x in obj])
        return [
            remap_to_cat_rule(x, categorical_cols, reverse_map,placeholder_nums)
            for x in obj
        ]

    # Case 4: anything else
    return obj
                    #fixed size            #can be fixed size   #can be fixed size  #can be fixed size   #fixed size       #fixed size      #can be fixed size                                       
def learn_rule_gpu(embedded_data_original,   index_e_plus,       index_e_minus,       rev_index,          categorical_cols, placeholder_nums, used_items=[], ratio=0.5):
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
        
        t = best_item_gpu(embedded_data_original,index_e_plus, index_e_minus,categorical_cols, placeholder_nums, used_items + items)
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
                ab = fold_gpu(embedded_data_original,index_e_minus,index_e_plus ,rev_index,categorical_cols,placeholder_nums, used_items + items, ratio)
                end_fold = timer()
                overall_fold += end_fold - start_fold
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break





    # Total time for profiling
    total_time = overall_best_item + overall_covers + overall_fold
    #print("returned rule: " +str(rule))
    return rule, overall_best_item, overall_covers,overall_fold,total_time,learn_rule_loops

def best_item_gpu(embedded_data_original,index_e_plus, index_e_minus,categorical_cols, placeholder_nums, used_items=[]):

    ret = -1, 0, 0
    if len(index_e_plus) == 0 and len(index_e_minus) == 0:
        return ret
    
    n = len(embedded_data_original[index_e_plus[0]]) if len(index_e_plus) > 0 else len(embedded_data_original[index_e_minus[0]]) #prende la lunghezza di una riga
    best = cp.float32(-1e20)
    
    #for each example check a literal providing the most IG
    for i in range(n - 1): # 0..n-1
        ig, r, v = best_ig_gpu(embedded_data_original,index_e_plus, index_e_minus, i,categorical_cols,placeholder_nums, used_items)
        if best < ig:
            best = ig
            ret = i, r, v
    #print("best item"+ str(ret))
    return ret

def fold_gpu(embedded_data_original,index_e_plus, index_e_minus, rev_index,categorical_cols,placeholder_nums, used_items=[], ratio=0.5):
    ret = []
    while len(index_e_plus) > 0:
        #print("fold gpu")
        rule,_,_,_,_,_ = learn_rule_gpu(embedded_data_original,index_e_plus, index_e_minus, rev_index,categorical_cols, placeholder_nums,used_items, ratio)
        data_fn = [i for i in index_e_plus if not cover_gpu_dev(rule, embedded_data_original,i, categorical_cols, placeholder_nums)]
        if len(index_e_plus) == len(data_fn):
            break
        index_e_plus = data_fn
        ret.append(rule)
    return ret

def best_ig_gpu(embedded_data_original,index_e_plus, index_e_minus, i, categorical_cols, placeholder_nums, used_items=[]):
    
    xp, xn, cp, cn = 0, 0, 0, 0

    pos, neg = dict(), dict()
    unique_vals_present, unique_cats_present = set(), set()

    #outer loop is on the columns

    for index in index_e_plus: #loop per example (row)
        d=embedded_data_original[index]
        if d[i] not in pos:
            pos[d[i]] = 0 
            neg[d[i]] = 0

        #d[i] is the value of a cell in column i
        pos[d[i]] += 1 

        if i in categorical_cols or d[i]==placeholder_nums[i]: #se is 
            unique_cats_present.add(d[i]) #add to the unique cat values found
            cp += 1
        else: 
            unique_vals_present.add(d[i]) #add to the unique num values found
            xp += 1

    for index in index_e_minus:
        d=embedded_data_original[index]
        if d[i] not in neg:
            pos[d[i]] = 0
            neg[d[i]] = 0
        neg[d[i]] += 1

        if i in categorical_cols or d[i]==placeholder_nums[i]:
            unique_cats_present.add(d[i])
            cn += 1
        else:
            unique_vals_present.add(d[i])
            xn += 1
    
    unique_cats_present, unique_vals_present = list(unique_cats_present), list(unique_vals_present)
    #print("uniquecats:")
    #print(unique_cats_present)
    #print("unique_vals_present_gpu:")
    #print(unique_vals_present)
    unique_vals_present.sort()
    unique_cats_present.sort()


    for j in range(1, len(unique_vals_present)):
        pos[unique_vals_present[j]] += pos[unique_vals_present[j - 1]]
        neg[unique_vals_present[j]] += neg[unique_vals_present[j - 1]]
    
    best, v, r = cupy.float32(-1e20), cupy.float32(-1e20), 0
    
    for x in unique_vals_present:
        if (i, 0, x) in used_items or (i, 1, x) in used_items:
            continue
        ig = gain(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x]) #su gpu, tempi assurdi causa data transfer
        if best < ig:
            best, v, r = ig, x, 0
        ig = gain(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x])
        if best < ig:
            best, v, r = ig, x, 1

    for c in unique_cats_present:
        if (i, 2, c) in used_items or (i, 3, c) in used_items:
            continue
        ig = gain(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
        if best < ig:
            best, v, r = ig, c, 2
        ig = gain(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn)
        if best < ig:
            best, v, r = ig, c, 3
    
    return best, r, v

def gain(tp, fn, tn, fp):
    if tp + tn < fp + fn:
        return cp.float32(-1e20)
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    ret += tp / tot * math.log(tp / tot_p) if tp > 0 else 0
    ret += fp / tot * math.log(fp / tot_p) if fp > 0 else 0
    ret += tn / tot * math.log(tn / tot_n) if tn > 0 else 0
    ret += fn / tot * math.log(fn / tot_n) if fn > 0 else 0
    return ret


@cuda.jit#(device=True)
def gain_device(tp, fn, tn, fp):

    if tp + tn < fp + fn:
        return -1e20

    tot_p = tp + fp
    tot_n = tn + fn
    tot   = tot_p + tot_n

    ret = 0.0

    if tp > 0:
        ret += (tp / tot) * math.log(tp / tot_p)
    if fp > 0:
        ret += (fp / tot) * math.log(fp / tot_p)
    if tn > 0:
        ret += (tn / tot) * math.log(tn / tot_n)
    if fn > 0:
        ret += (fn / tot) * math.log(fn / tot_n)

    return ret

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
