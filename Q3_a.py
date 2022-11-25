# Machine Learning - Dr. Shandian Zhe, University of Utah, Fall 2022
# Hamid Manoochehri (u1303306)
# Homework 4
# Part 2 - Question 3(a)
import numpy as np
import pandas as pd
from scipy.optimize import minimize

csv_cols = ['variance','skewness','curtosis','entropy','label']

# read training (tr) data
tr_data   = pd.read_csv('train.csv' , names = csv_cols , dtype = np.float64()).replace({'label':0} , -1)
tr_rows = tr_data.shape[0]
xi      = tr_data.iloc[: , :-1].to_numpy()
yi      = tr_data['label'].to_numpy().reshape( (tr_rows , 1))
# read testing (te) data
te_data   = pd.read_csv('test.csv'  , names = csv_cols , dtype = np.float64()).replace({'label':0},-1)

print('ENTER \'C\' VALUE')
A = float(input('Nominator: '  ))
B = float(input('Denominator: '))
C = A/B

# define predictor fn
def pred(df , w):
    e = 0 # error
    rows = df.shape[0]
    for r in range(rows):
        xi, yi = df.iloc[r ,:-1], df.iloc[r , -1]    
        p = np.sign(np.dot(w , xi)) # prediction
        if p == 0:
            p = 1
        if p != yi:
            e += 1
    return e/rows

# define dual L function
def dual(alpha):
    alpha = np.reshape(alpha , (tr_rows , 1))
    d     = (1/2)*( np.matmul(np.multiply(np.multiply(xi,yi) , alpha) , np.multiply(np.multiply(xi,yi) , alpha).transpose()).sum() ) - np.sum(alpha)
    return d

# define constraints function
def const(alpha):
    alpha  = np.reshape(alpha , (tr_rows , 1))
    yalpha = np.multiply(yi , alpha).sum()
    return yalpha

constraints = ({'type':'eq' , 'fun':const})
bounds      = np.ones( (tr_rows,1) )*(0,C)
x0          = np.zeros( (tr_rows , 1) )

solver      = minimize(fun = dual , x0 = x0 , method = 'SLSQP' , bounds = bounds , constraints = constraints)
alpha_final = solver.x

# final weights
alpha_final = np.reshape(alpha_final , (tr_rows , 1))
xyo         = np.multiply(xi , yi) # output x,y
wo          = np.multiply(xyo , alpha_final).sum(axis = 0) # output weights

wt   = wo.reshape(1 , wo.shape[0])
bset = []
for j in range (alpha_final.shape[0]):
    if alpha_final[j] > 0 and alpha_final[j] < C:
        xj = xi[j,:]
        b  = yi[j] - np.dot(wt,xj)
        bset.append(b)
b = np.mean(bset)

# augmentation
w = np.insert(wt , 0 , [b] , axis=1)
tr_data.insert(0 , 'b' , 1)
te_data.insert(0 , 'b' , 1)

tr_err = pred(tr_data , w)
te_err = pred(te_data , w)

print('weights: ' , wo, ', bias: ' , b, ', penalty coeff: ' , C)
print('error(tr):', tr_err, ', error(te):', te_err)


