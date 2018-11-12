import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#student dataset
data = pd.read_csv('student.csv')
alpha = 0.0001


math = data['Math'].values
read = data['Reading'].values
writing = data['Writing'].values
m = len(math)
X0 = np.ones(m)
X = np.array([X0,math,read]).T
B = np.array([0,0,0])
Y = np.array(writing)

def cost_function(x,y,b):
    m = len(y)
    j = np.sum((x.dot(b) - y)**2)/(2*m)
    return j

def gradient_descent(x,y,b,alpha,itera):
    cost_history = [0]*itera
    m = len(x)
    for i in range(itera):
        h = x.dot(b)
        loss = h - y
        gradient = x.T.dot(loss)
        b = b - (alpha/m) * gradient
        #y_pred = x.dot(b)
        #print(y_pred)
        cost = cost_function(x,y,b)
        cost_history[i] = cost
    return b,cost_history

#model evaluation-Rmse
def rmse(y_predict,y):
    rmse = np.sqrt(sum((y - y_predict)**2)/len(y))
    return rmse


#model evaluation-R2 score
def r2_score(y,y_predict):
    mean_y = np.mean(y)
    ss_tot = sum((y - mean_y)**2)
    ss_res = sum((y - y_predict)**2)
    r2 = 1 - (ss_res/ss_tot)
    return r2

iters = 100000

newb,cost_history = gradient_descent(X,Y,B,alpha,iters)
y_pred = X.dot(newb)


