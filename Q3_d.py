# Machine Learning - Dr. Shandian Zhe, University of Utah, Fall 2022
# Hamid Manoochehri (u1303306)
# Homework 4
# Part 2 - Question 3(d)
import numpy as np
import pandas as pd

def pred(tr_data , te_data , kernel , C):
    e = 0 # init error
    tr_rows = tr_data.shape[0]
    te_rows = te_data.shape[0]
    ytr = tr_data['label'].to_numpy().reshape((tr_rows,1))
    for r in range(te_rows):
        cyk = np.sum(np.multiply(np.multiply(ytr , C) , kern[:,r].reshape((kern.shape[0] , 1)))) #np.reshape(kernel[:,row],(kernel.shape[0],1))))
        p   = np.sign(cyk) # prediction
        if p == 0:
            p = 1
        yte = te_data.iloc[r , -1]   
        if p != yte:
            e += 1
    return e/te_rows

#reading data
csv_cols = ['variance' , 'skewness' , 'curtosis' , 'entropy' , 'label']
tr_data=pd.read_csv('train.csv' , names = csv_cols , dtype = np.float64()).replace({'label':0} , -1)
tr_data.insert(0,'b',1)
tr_rows = tr_data.shape[0]
te_data=pd.read_csv('test.csv'  , names = csv_cols , dtype = np.float64()).replace({'label':0} , -1)
te_data.insert(0,'b',1)
xi = tr_data.iloc[: , :-1].to_numpy()
yi = tr_data['label'].to_numpy().reshape((tr_rows,1))

# hyperparam settings
gammaset = [ 0.1,.05,1,5,100]
epoch    = 100
for gamma in gammaset:
    def kernel(u , v , gamma):
        k = np.zeros( (u.shape[0] , v.shape[0]) )
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                d = u[i,:] - v[j,:]
                dist = np.linalg.norm(d)
                k[i,j] = np.exp(-(dist**2)/gamma)
        return k
    kern = kernel(xi , xi , gamma)
    C = np.zeros( (tr_rows , 1) )
    for ep in range(epoch):
        for j in range(tr_rows):
            yC     = np.multiply(yi , C)
            kernt  = kern[:,j].reshape( (kern.shape[0] , 1) )
            YCK    = np.multiply(kernt , yC)
            yck    = np.sum(YCK)
            y_pred = np.sign(yck)
            if y_pred == 0:
                y_pred = 1
            if y_pred != yi[j]:
                C[j] = C[j] + 1
    
    C_final = C.reshape((tr_rows,1))
    tr_err  = pred(tr_data , tr_data , kern , C_final)
    xi_te   = te_data.iloc[: , :-1].to_numpy()
    kern_te = kernel(xi , xi_te , gamma)
    te_err  = pred(tr_data , te_data , kern_te , C_final)
    
    print('gamma=', gamma)
    print ('error(tr): ', tr_err, ', error(te): ', te_err)
    
