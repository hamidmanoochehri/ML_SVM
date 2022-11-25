# Machine Learning - Dr. Shandian Zhe, University of Utah, Fall 2022
# Hamid Manoochehri (u1303306)
# Homework 4
# Part 2 - Question 2(b)
import numpy as np
import pandas as pd

def sgdSVM(df , epoch , w_init , gamma_0 , C):
    for ep in range(epoch):
        N = df.shape[0]
        gamma_t = (gamma_0/(ep+1))
        w       = w_init
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        for r in range(N):
            xi = shuffled_df.iloc[r ,:-1].to_numpy()
            yi = shuffled_df.iloc[r , -1]
            b      = w[0,0]
            w0     = np.delete(w,0 , axis=1) # removing bias values to keep weights only 
            check  = (2*yi - 1)*np.dot(w , xi)
            if check <= 1:
                w_aug = np.insert(w0 , 0 , list2(0) , axis=1)
                w     -= gamma_t*(w_aug) + gamma_t*C*N*(2*yi - 1)*(xi.transpose())
            elif check > 1:
                w0 = (1 - gamma_t)*w0
                w  = np.insert(w0 , 0 , list2(b) , axis=1)
    return w

def pred(df,w):
    e = 0 # init error
    rows = df.shape[0]
    for r in range(rows):
        p = 0 # init prediction
        xi, yi = df.iloc[r ,:-1] ,  df.iloc[r , -1]
        p  = np.sign(np.dot(w , xi))
        if p == 0:
            p = 1
        if p != 2*yi - 1:
            e += 1
    return e/rows

def list2(c):
    return [[c]]

csv_cols = ['variance' , 'skewness' , 'curtosis' , 'entropy' , 'label']

# reading data
tr_data = pd.read_csv('train.csv' , names = csv_cols , dtype = np.float64())
tr_data.insert(0 , 'b' , 1)
te_data = pd.read_csv('test.csv'  , names = csv_cols , dtype = np.float64())
te_data.insert(0 , 'b' , 1)

# hyperparam settings
w_init = np.zeros( (1 , tr_data.shape[1]-1) )
Cset=[100/873 , 500/873 , 700/873]
gamma_0set= [.1 , .01 , .001 , .0001 , .00001]


for g0 in gamma_0set:
    for C in Cset:
        epoch = 100
        w = sgdSVM(tr_data , epoch , w_init , g0 , C)
        tr_err = pred(tr_data , w)
        te_err = pred(te_data , w)
        # print results
        print ('Penalty coeff: ', C, 'gamma_0: ', g0)
        print ('error(tr): ', tr_err, 'error(te): ', te_err)