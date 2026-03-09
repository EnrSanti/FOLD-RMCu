from foldrm import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta
import re
import threading


def run_test1():
    model,data_train, data_test = MNIST()  #c.a. 6 min e 30 a 0.2 di ratio

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
    del(data_train)
    del(data_test)

    model,data_train, data_test = MNIST()  #c.a. 6 min e 30 a 0.2 di ratio

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

    if h_cpu != h_gpu:
        print(f"{RED}test1 failed{RESET}")
        print(h_cpu+"\n-----------------------------------\n"+h_gpu)
    else:
        print(f"{GREEN}test1 passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")


    if(accuracy_cpu != accuracy_gpu):
        print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
        print(accuracy_cpu+"\n-----------------------------------\n"+accuracy_gpu)

def run_test2():
    model,data = MINITEST()  

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

    model,data = MINITEST()  

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

    if h_cpu != h_gpu:
        print(f"{RED}test2 failed{RESET}")
        print(h_cpu+"\n-----------------------------------\n"+h_gpu)
    else:
        print(f"{GREEN}test2 passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")

    if(accuracy_cpu != accuracy_gpu):
        print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
        print(accuracy_cpu+"\n-----------------------------------\n"+accuracy_gpu)

def run_test3():
    model, data = diabetes() #11 sec a 0.5 o 10 min & 30 a 0.2 di ratio

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


    model, data = diabetes() #11 sec a 0.5 o 10 min & 30 a 0.2 di ratio

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

    if h_cpu != h_gpu:
        print(f"{RED}test3 failed{RESET}")
        print(h_cpu+"\n-----------------------------------\n"+h_gpu)
    else:
        print(f"{GREEN}test3 passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")


    if(accuracy_cpu != accuracy_gpu):
        print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
        print(accuracy_cpu+"\n-----------------------------------\n"+accuracy_gpu)

def run_test4():
    model, data = australia() # 6 min a 0.1 di ratio 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    start = timer()
    model.fit(data_train, ratio=0.1)
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

    model, data = australia() # 6 min a 0.1 di ratio 
    
    data_train, data_test = split_data_deterministically(data, ratio=0.8)
    
    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.1)
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

    if h_cpu != h_gpu:
        print(f"{RED}test4 failed{RESET}")
        print(h_cpu+"\n-----------------------------------\n"+h_gpu)
    else:
        print(f"{GREEN}test4 passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")


    if(accuracy_cpu != accuracy_gpu):
        print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
        print(accuracy_cpu+"\n-----------------------------------\n"+accuracy_gpu)

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

    if h_cpu != h_gpu:
        print(f"{RED}test5 failed{RESET}")
        print(h_cpu+"\n-----------------------------------\n"+h_gpu)
    else:
        print(f"{GREEN}test5 passed{RESET}")
        print(f"Serial: {timedelta(seconds=end - start)} Parallel: {timedelta(seconds=end_gpu - start_gpu)}")


    if(accuracy_cpu != accuracy_gpu):
        print(f"{RED}ACCURACY DIFFERENCE(?){RESET}")
        print(accuracy_cpu+"\n-----------------------------------\n"+accuracy_gpu)


#si, molto alla buona, ma per ora va bene

def compare_times(parallel=False):
    tests = [
        run_test1,
        run_test2,
        run_test3,
        run_test4,
        run_test5
    ]

    errors = 0
    lock = threading.Lock()

    if parallel:
        barrier = threading.Barrier(len(tests) + 1)  # +1 for main thread

        def worker(test_func, idx):
            nonlocal errors
            try:
                print(f"[Test {idx}] starting")
                result = test_func()  # assume returns 0 if ok, 1 if failed

                if result:   # test reported failure
                    with lock:
                        errors += 1

            except Exception as e:
                print(f"[Test {idx}] crashed:", e)
                with lock:
                    errors += 1

            finally:
                barrier.wait()  # only in parallel

        threads = []
        for idx, test in enumerate(tests):
            t = threading.Thread(target=worker, args=(test, idx))
            t.start()
            threads.append(t)

        barrier.wait()  # main thread waits for all workers
        for t in threads:
            t.join()

    else:
        # Sequential execution
        for idx, test in enumerate(tests):
            try:
                print(f"[Test {idx}] starting")
                result = test()
                if result:  # test reported failure
                    errors += 1
            except Exception as e:
                print(f"[Test {idx}] crashed:", e)
                errors += 1

    # Final summary
    if errors == 0:
        print("\033[92mALL TESTS PASSED\033[0m")
    else:
        print(f"\033[91m{errors} TESTS FAILED\033[0m")
def fast_check():
    test_failed=0
    loaders = [breastw]


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


def minitest_for_debugging():

    model,data = MINITEST_check()

    data_train, data_test = split_data_deterministically(data, ratio=1)


    start_gpu = timer()
    model.fitGPU(data_train, ratio=0.2)
    end_gpu = timer()

    h_gpu=model.get_asp(simple=True)
    print(h_gpu)


    del(model)
    del(data)
    del(data_train)
    del(data_test)

    model,data = MINITEST_check()

    data_train, data_test = split_data_deterministically(data, ratio=1)


    model.fit(data_train, ratio=0.2)

    h_gpu=model.get_asp(simple=True)
    print(h_gpu)


def main():
    
    #compare_times()
    fast_check()
    #minitest_for_debugging()

if __name__ == '__main__':
    main()