from foldrm import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta

def main():
    
    #model,data = osmer_coverage()
    model,data_train, data_test = MNIST()  #c.a. 6 min e 30 a 0.2 di ratio
    #model,data = MINITEST()  
    #model, data = diabetes() #11 sec a 0.5 o 10 min & 30 a 0.2 di ratio
    #model, data = australia() # 6 min a 0.1 di ratio 
    
    #model, data = coverType() #16 mine 20 a 0.2 ratio 
    #data_train, data_test = split_data_deterministically(data, ratio=0.8)


    #sposta dati su gpu
    print("train: "+str(data_train)+"\n")
    print("test: "+str(data_test)+"\n")
    start = timer()
    model.fit(data_train, ratio=0.2)
    end = timer()

    model.print_asp(simple=True)
    Y = [d[-1] for d in data_test]
    Y_test_hat = model.predict(data_test)
    acc = get_scores(Y_test_hat, data_test)
    print('% acc', round(acc, 4), '# rules', len(model.crs))
    acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
    print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))
    print('% foldrm costs: ', timedelta(seconds=end - start), '\n')

    # k = 1
    # for i in range(len(data_test)):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(data_test[i]))
    #     print('Proof Tree for example number', k, ':')
    #     print(model.proof(data_test[i]))
    #     k += 1



if __name__ == '__main__':
    main()
