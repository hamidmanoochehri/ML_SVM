# Machine Learning - Dr. Shandian Zhe, University of Utah, Fall 2022
# Hamid Manoochehri (u1303306)
# Homework 4
# Part 2 - Question 2(a)
import pandas as pd
import numpy as np

# SVM define
def sgd_svm(df , epoch , w_init , gamma0 , a , C):
    for t in range(epoch):
        gamma_t = ( gamma0/(1 + (gamma0/a)*t) )
        w = w_init
        shuffled_df = df.sample(frac = 1)
        shuffled_df = shuffled_df.reset_index(drop = True)
        N = shuffled_df.shape[0]
        for r in range(N):
            xi = shuffled_df.iloc[r , :-1].to_numpy()
            yi = shuffled_df.iloc[r ,  -1]
            b  = w[0, 0]
            w0 = np.delete(w , 0 , axis=1)
            z  = (2*yi - 1)*np.dot(w , xi)
            if z <= 1:
                w_aug = np.insert(w0 , 0 , [[0]] , axis=1)
                w     = w - gamma_t*(w_aug) + gamma_t*C*N*(2*yi - 1)*(xi.transpose())
            else:
                w0 = (1-gamma_t)*w0
                w  = np.insert(w0 , 0 , [[b]] , axis=1)
    return w
            
# Predictor define
def pred(df,w):
    rows = df.shape[0]
    e = 0
    for r in range(rows):
        p = 0 # prediction
        xi= df.iloc[r,:-1]
        yi= df.iloc[r,-1]    
        p = np.sign( np.dot(w , xi) )
        if p == 0:
            p = 1
        elif p != 2*(yi) - 1:
            e += 1
    return e/rows


csv_cols = ['variance','skewness','curtosis','entropy','label']
# data read
tr_data = pd.read_csv('train.csv' , names = csv_cols , dtype = np.float64())
te_data = pd.read_csv('test.csv'  , names = csv_cols , dtype = np.float64())
# integrating b
tr_data.insert(0,'b',1)
te_data.insert(0,'b',1)

# hyperparams
tr_cols   = tr_data.shape[1]
w_init    = np.zeros( (1 , tr_cols - 1) )
Cset      = [100/873 , 500/873 , 700/873]
aset      = [100 , 10 ,1 , 0.1 , 0.01 , 0.001 , 0.0001]
gamma0set = [0.1,0.01,0.001,0.0001,0.00001]
for gamma0 in gamma0set:
    for a in aset:
        for C in Cset:
            epoch = 100
            w = sgd_svm(tr_data , epoch , w_init , gamma0 , a , C)
            tr_err = pred(tr_data , w)
            te_err = pred(te_data , w)
            
            # print results
            print ('C: ', C, ', gamma0: ', gamma0 , ', a: ' ,a )
            print ('w=', w)
            print ('error(tr)', tr_err, ', error(te)', te_err)