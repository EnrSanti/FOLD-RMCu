import math
from numba import cuda
import numpy as np
from timeit import default_timer as timer
from algo import *


def evaluate_gpu(item, x):
    def __eval_gpu(i, r, v):
        if isinstance(v, str):
            if r == '==':
                return x[i] == v
            elif r == '!=':
                return x[i] != v
            else:
                return False
        elif isinstance(x[i], str):
            return False
        elif r == '<=':
            return x[i] <= v
        elif r == '>':
            return x[i] > v
        else:
            return False

    def _eval_gpu(i):
        if len(i) == 3:
            return __eval_gpu(i[0], i[1], i[2])
        elif len(i) == 4:
            return evaluate(i, x)

    if len(item) == 0:
        return 0
    if len(item) == 3:
        return _eval_gpu(item[0], item[1], item[2])
    if item[3] == 0 and len(item[1]) > 0 and not all([_eval_gpu(i) for i in item[1]]):
        return 0
    if len(item[2]) > 0 and any([_eval_gpu(i) for i in item[2]]):
        return 0
    return 1


def cover_gpu(item, x):
    return evaluate_gpu(item, x)



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
    while len(data) > 0:
        total_loops += 1

        start_most = timer()
        l = most(data)
        end_most = timer()
        overall_most += end_most - start_most
        
        start_split = timer()
        e_plus, e_minus = split_data_by_item(data, l)
        end_split = timer()
        overall_split += end_split - start_split

        start_learn = timer()
        rule,best_item, coversTime,foldTime,timeTotal,loops = learn_rule_gpu(e_plus, e_minus, [], ratio)
        overall_best_item+=best_item
        overall_covers+=coversTime
        overall_fold+=foldTime
        total_time+=timeTotal
        learn_rule_loops+=loops
        end_learn = timer()
        overall_learn += end_learn - start_learn
        
        start_covers1 = timer()
        e_tp = [e_plus[i] for i in range(len(e_plus)) if not cover(rule, e_plus[i])]
        end_covers1 = timer()
        overall_covers1 += end_covers1 - start_covers1

        if len(e_tp) == len(e_plus):
            break

        start_setop = timer()
        data = e_tp + [e_minus[i] for i in range(len(e_minus)) if not cover(rule, e_minus[i])]
        end_setop = timer()

        overall_setop += end_setop - start_setop
        
        # Append rule with selected literal
        rule = l, rule[1], rule[2], rule[3]
        ret.append(rule)
    
    # Total time spent
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


def learn_rule_gpu(data_pos, data_neg, used_items=[], ratio=0.5):
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
        
        t = best_item_gpu(data_pos, data_neg, used_items + items)
        end_best_item = timer()
        overall_best_item += end_best_item - start_best_item 

        items.append(t)
        rule = -1, items, [], 0

        # ===== cover timing =====
        start_cover_pos_neg = timer()
        data_pos = [data_pos[i] for i in range(len(data_pos)) if cover_gpu(rule, data_pos[i])]
        data_neg = [data_neg[i] for i in range(len(data_neg)) if cover_gpu(rule, data_neg[i])]
        end_cover_pos_neg = timer()
        overall_covers += end_cover_pos_neg - start_cover_pos_neg

        # Check termination conditions
        if t[0] == -1 or len(data_neg) <= len(data_pos) * ratio:
            if t[0] == -1:
                rule = -1, items[:-1], [], 0

            if len(data_neg) > 0 and t[0] != -1:
                # ===== fold timing =====
                start_fold = timer()
                ab = fold_gpu(data_neg, data_pos, used_items + items, ratio)
                end_fold = timer()
                overall_fold += end_fold - start_fold

                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break

    # Total time for profiling
    total_time = overall_best_item + overall_covers + overall_fold

    return rule, overall_best_item, overall_covers,overall_fold,total_time,learn_rule_loops

def best_item_gpu(X_pos, X_neg, used_items=[]):

    ret = -1, '', ''
    if len(X_pos) == 0 and len(X_neg) == 0:
        return ret
    
    n = len(X_pos[0]) if len(X_pos) > 0 else len(X_neg[0])
    best = float('-inf')
    
    for i in range(n - 1):
        ig, r, v = best_ig(X_pos, X_neg, i, used_items)
        if best < ig:
            best = ig
            ret = i, r, v
    
    return ret

def fold_gpu(data_pos, data_neg, used_items=[], ratio=0.5):
    ret = []
    while len(data_pos) > 0:
        rule,_,_,_,_,_ = learn_rule_gpu(data_pos, data_neg, used_items, ratio)
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover_gpu(rule, data_pos[i])]
        if len(data_pos) == len(data_fn):
            break
        data_pos = data_fn
        ret.append(rule)
    return ret