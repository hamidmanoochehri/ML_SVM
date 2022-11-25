# Machine Learning - Dr. Shandian Zhe, University of Utah, Fall 2022
# Hamid Manoochehri (u1303306)
# Homework 4
# Part 2 - Question 3(b)
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def pred(tr_data , te_data , b , kernel , alpha):
    e    = 0
    ytr = tr_data['label'].to_numpy()
    ytr = np.reshape(ytr , (tr_data.shape[0] , 1))
    for r in range(te_data.shape[0]):
        ayk = 0
        for i in range(kernel.shape[0]):
            ayk += alpha_final[i]*ytr[i]*kernel[i,r]
        p = np.sign(ayk + b)
        if p == 0:
            p = 1
        yte = te_data.iloc[r , -1]
        if p != yte:
            e += 1
    return e/te_data.shape[0]

# data read
csv_cols  = ['variance','skewness','curtosis','entropy','label']
tr_data = pd.read_csv('train.csv' , names = csv_cols , dtype = np.float64() ).replace({'label':0} , -1)
tr_rows = tr_data.shape[0]
te_data = pd.read_csv('test.csv'  , names = csv_cols , dtype = np.float64() ).replace({'label':0} , -1)
tr_rows = tr_data.shape[0]
xi      = tr_data.iloc[:,:-1].to_numpy()
yi      = tr_data['label'].to_numpy().reshape((tr_data.shape[0] , 1))

# hyperparam settings
Cset     = [100/873 , 500/873 , 700/873]
gammaset = [.1 , .5 , 1 , 5 , 100]

for C in Cset:
    for gamma in gammaset:
        def kernel(u , v , gamma):
            k = np.zeros( (u.shape[0] , v.shape[0]) )
            for i in range(u.shape[0]):
                for j in range(v.shape[0]):
                    d       = u[i,:] - v[j,:]
                    dist    = np.linalg.norm(d)
                    k[i,j]  = np.exp( -(dist**2)/gamma )
            return k
        
        kern = kernel(xi , xi , gamma)
        yij  = np.matmul(yi , yi.transpose())
        of   = np.multiply(kern , yij)
        
        def dual(alpha):
            alpha    = np.reshape(alpha , (tr_data.shape[0] , 1))
            alpha_ij = np.matmul(alpha , alpha.transpose())
            of_b     = np.multiply(of , alpha_ij)
            d        = (1/2)*(of_b.sum()) - alpha.sum()
            return d
        
        def const(alpha):
            alpha = np.reshape(alpha , (tr_data.shape[0] , 1) )
            return np.multiply(yi , alpha).sum()
        
        constraints = ( {'type':'eq' , 'fun':const } )
        
        bounds      = np.ones( (tr_rows,1) )*(0,C)
        x0 = np.zeros((tr_data.shape[0] , 1))
        #bounds = [( 0 , C) for _ in x0] 
        
        solver = minimize(fun = dual , x0 = x0 , method = 'SLSQP' , bounds = bounds , constraints=constraints)
        alpha_final = solver.x
        alpha_final = np.reshape(alpha_final , (tr_data.shape[0],1))
        
        bset = []
        for j in range( alpha_final.shape[0] ):
            if alpha_final[j] > 0 and alpha_final[j] < C:
                ayk = 0
                for i in range( alpha_final.shape[0] ):
                     ayk += alpha_final[i]*yi[i]*kern[i , j]
                b = yi[j] - ayk
                bset.append(b)
        b = np.mean(bset)
        
        tr_err = pred(tr_data , tr_data , b , kern , alpha_final)
        xi_te  = df_te.iloc[: , :-1].to_numpy()
        kern_te = kernel(xi , xi_te , gamma)
        te_err  = pred(tr_data , df_te , b , kern_te , alpha_final)
        
        # output
        print ('penalty coeff: ', C, ', gamma: ', gamma, ', bias: ' , b)
        print ('error(tr)', tr_err, ', error(te)', te_err)
