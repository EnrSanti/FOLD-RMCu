from foldrm import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta
import re
import threading
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#si, molto alla buona, ma per ora va bene, con più test serve il refactor

def run_test1():
    model,data_train, data_test = MNIST()  #c.a. 6 min e 30 a 0.2 di ratio

    start = timer()
    model.fit(data_train, ratio=0.5)
    end = timer()

    h_cpu=model.get_asp(simple=True)

    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))



    del(model)
    del(data_train)
    del(data_test)

    model,data_train, data_test = MNIST()  #c.a. 6 min e 30 a 0.2 di ratio

    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.5)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")

    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel,"MNIST"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1,"MNIST"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "MNIST"

def run_test2():
    model,data = enterprise()  

    data_train, data_test = split_data_deterministically(data, ratio=0.9)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.05)
    end = timer()

    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)

    model,data = enterprise()  

    data_train, data_test = split_data_deterministically(data, ratio=0.9)

    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.05)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "MINITEST"
        else:
            print(f"{RED}test failed{RESET}")
            print("ACCURACIES ", accuracy_cpu, " vs ", accuracy_gpu)
            return 1,-1,-1, "MINITEST"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")
        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "MINITEST"

def run_test3():
    model, data = diabetes() #11 sec a 0.5 o 10 min & 30 a 0.2 di ratio

    data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.25)
    end = timer()
    h_cpu=model.get_asp(simple=True)

    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))



    del(model)
    del(data)
    del(data_train)
    del(data_test)


    model, data = diabetes() #11 sec a 0.5 o 10 min & 30 a 0.2 di ratio

    data_train, data_test = split_data_deterministically(data, ratio=0.8)

    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.25)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel,"diabetes"
        else:
            print(f"{RED}test failed{RESET}")
            print("ACCURACIES ", accuracy_cpu, " vs ", accuracy_gpu)
            return 1,-1,-1,"diabetes"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "diabetes"

def run_test4():
    model, data = weather() 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.25)
    end = timer()
    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))


    del(model)
    del(data)
    del(data_train)
    del(data_test)

    model, data = weather() # 6 min a 0.1 di ratio 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.8)
    
    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.25)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "weather"
        else:
            print(f"{RED}test failed{RESET}")
            print("ACCURACIES ", accuracy_cpu, " vs ", accuracy_gpu)
            return 1,-1,-1, "weather"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "weather"

def run_test6():
    model, data = smoke_drink() # 6 min a 0.1 di ratio 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.7)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.4)
    end = timer()
    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))


    del(model)
    del(data)
    del(data_train)
    del(data_test)

    model, data = smoke_drink() # 6 min a 0.1 di ratio 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.7)
    
    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.4)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "smoke_drink"
        else:
            print(f"{RED}test failed{RESET}")
            print("ACCURACIES ", accuracy_cpu, " vs ", accuracy_gpu)
            return 1,-1,-1, "smoke_drink"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "smoke_drink"

def run_test5():

    model, data = coverType() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.2)
    end = timer()


    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)
    
    model, data = coverType() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.8)



    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.2)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "coverType"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1, "coverType"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "coverType"

def run_test7():

    model, data = sloan() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.3)
    end = timer()


    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)
    
    model, data = sloan() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)



    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.3)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "sloan"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1, "sloan"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "sloan"

def run_test8():

    model, data = human_activity() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.3)
    end = timer()


    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)
    
    model, data = human_activity() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)



    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.3)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "human"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1, "human"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "human"

def run_test9():

    model, data = swat() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.05)
    end = timer()


    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)
    
    model, data = swat() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.9)



    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.05)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "swat"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1, "swat"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "swat"

def run_test10():

    model, data = lifestyle() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.2)
    end = timer()


    h_cpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_cpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))




    del(model)
    del(data)
    del(data_train)
    del(data_test)
    
    model, data = lifestyle() #16 mine 20 a 0.2 ratio 
    data_train, data_test = split_data_deterministically(data, ratio=0.8)



    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.2)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    accuracy_gpu = get_scores(Y_test_hat, data_test)
    print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

    print("----------------------------------------------------------------")
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    if h_cpu != h_gpu:
        if(accuracy_cpu == accuracy_gpu):
            print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            time_serial=end - start
            time_parallel=end_gpu - start_gpu
            return 0,time_serial,time_parallel, "lifestyle"
        else:
            print(f"{RED}test1 failed{RESET}")
            return 1,-1,-1, "lifestyle"
    else:
        print(f"{GREEN}test passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

        time_serial=end - start
        time_parallel=end_gpu - start_gpu
        return 0,time_serial,time_parallel, "lifestyle"

def compare_times():
    tests = [
        #run_test2,
        #run_test9,
        run_test8,
        run_test10,
        run_test7,
        run_test3,
        run_test4,
        run_test6,
        run_test5,
        run_test1,
    ]

    errors = 0
    lock = threading.Lock()
    serial_times=[-1]*len(tests)
    parallel_times=[-1]*len(tests)
    names=[""]*len(tests)
    # Sequential execution
    for idx, test in enumerate(tests):
        try:
            print(f"[Test {idx}] starting")
            result = test()
            result,time_serial,time_parallel, name = result
            if result:  # test reported failure
                errors += 1
                serial_times[idx]=None
                parallel_times[idx]=None
            else:
                serial_times[idx]=time_serial
                parallel_times[idx]=time_parallel
            names[idx]=name
        except Exception as e:
            import traceback
            print(f"[Test {idx}] crashed:", e)
            print(traceback.format_exc()) 
            print("-" * 30)
            errors += 1

    # Final summary
    if errors == 0:
        print("\033[92mALL TESTS PASSED\033[0m")
    else:
        print(f"\033[91m{errors} TESTS FAILED\033[0m")
    plot_test_results(names,serial_times,parallel_times)

def plot_test_results(names, serial_times, parallel_times):
    # 1. Filter out tests that failed
    plot_data = [
        (n, s, p) for n, s, p in zip(names, serial_times, parallel_times) 
        if s is not None and s != -1
    ]
    
    if not plot_data:
        print("No valid test data to plot.")
        return

    valid_names = [d[0] for d in plot_data]
    s_times = [d[1] for d in plot_data]
    p_times = [d[2] for d in plot_data]

    x = np.arange(len(valid_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7)) # Slightly wider for labels
    
    rects1 = ax.bar(x - width/2, s_times, width, label='Serial', color='#5dade2')
    rects2 = ax.bar(x + width/2, p_times, width, label='Parallel', color='#58d68d')

    # --- Add Speedup Labels ---
    for i in range(len(valid_names)):
        s = s_times[i]
        p = p_times[i]
        
        # Calculate speedup (avoid division by zero)
        speedup = s / p if p > 0 else 0
        
        # Determine height for the label (top of the tallest bar in the pair)
        max_height = max(s, p)
        
        # Add text: "x2.5" etc.
        ax.text(x[i], max_height + (max_height * 0.02), f'x{speedup:.1f}', 
                ha='center', va='bottom', fontweight='bold', color='#2e4053')

    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Test Execution Comparison with Speedup Labels ({datetime.now().strftime("%Y-%m-%d")})')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.png"
    
    plt.savefig(filename)
    plt.close(fig)
    print(f"Results saved to: {filename}")

# Call this at the very end of your script
def fast_check():
    test_failed=0
    loaders = [acute,adult,breastw,autism, credit,heart,kidney, krkp, mushroom]


    for i in range(len(loaders)):
        model, data = loaders[i]()   # call function
        data_train, data_test = split_data_deterministically(data, ratio=0.8)
        
        start = timer()
        model.fit(data_train, ratio=0.4)
        end = timer()
        h_cpu=model.get_asp(simple=True)

        Y = [d[-1] for d in data_test]
        Y_test_hat = model.predict(data_test)
        accuracy_cpu = get_scores(Y_test_hat, data_test)
        print('% acc', round(accuracy_cpu, 4), '# rules', len(model.crs))
        acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
        print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

        del(model)
        del(data)
        del(data_train)
        del(data_test)

        model, data = loaders[i]()   # call function
        data_train, data_test = split_data_deterministically(data, ratio=0.8)

        start_gpu = timer()
        model.fitGPU(data_train, ratio=0.4)
        end_gpu = timer()

        h_gpu=model.get_asp(simple=True)
        Y = [d[-1] for d in data_test]
        Y_test_hat = model.predict(data_test)
        accuracy_gpu = get_scores(Y_test_hat, data_test)
        print('% acc', round(accuracy_gpu, 4), '# rules', len(model.crs))
        acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
        print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))

        print("----------------------------------------------------------------")
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        print("----------------------------------------------------------------")
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        YELLOW = "\033[33m"
        if h_cpu != h_gpu:
            if(accuracy_cpu == accuracy_gpu):
                print(f"{YELLOW}OK WORKS, != hyp = accuracy{RESET}")
            else:
                print(f"{RED}test1 failed{RESET}")
                print(h_cpu+"\n-----------------------------------\n"+h_gpu)
                test_failed+=1
        elif(accuracy_cpu != accuracy_gpu):
            print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
            print(str(accuracy_cpu)+"\n-----------------------------------\n"+str(accuracy_gpu))
            test_failed+=1
        else:
            print(f"{GREEN}test1 passed{RESET}")
            print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

            print(h_cpu+"\n-----------------------------------\n"+h_gpu)
        
        
    
    if(test_failed==0):
        print(f"{GREEN}-------------------\nALL passed\n-------------------{RESET}")
    else:        
        print(f"{RED}{test_failed}-------------------\nTEST FAILED\n-------------------{RESET}")

def main():
    
    #compare_times()
    fast_check()

if __name__ == '__main__':
    main()