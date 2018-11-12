from matplotlib import pyplot as plt
import numpy as np
from csv import reader




def load_csv(filename):
    dataset  = list()
    with open(filename,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset,column):
    for row in dataset:
        row[column]  = float(row[column])

def normalize_dataset(dataset,minmax):
    for row in dataset:
        for  i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])


def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min,value_max])
    return minmax


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))



def accurancy_metric(actual,predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        return correct/float(len(actual)) * 100.0

    
def logistic_regression(dataset,learning_rate,iters):
    costs = []
    w1 = 0
    w2 = 0
    w3 = 0
    w4 = 0
    w5 = 0
    w6 = 0
    w7 = 0
    w8 = 0
    b  = 0

    for i in range(iters):
        index = np.random.randint(len(dataset))
        point = dataset[index]

        z  = point[0] * w1 + point[1] * w2 + point[2] * w3 + point[3] * w4 + point[4] * w5 + point[5] * w6 + point[6] * w7 + point[7] * w8 + b
        pred = sigmoid(z)

        target = point[8]
        
        cost = np.square(pred - target)
        
        derivative_cost_pred =  2 * (pred - target)
        derivative_pred_z = derivative_sigmoid(z)
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_dw3 = point[2]
        dz_dw4 = point[3]
        dz_dw5 = point[4]
        dz_dw6 = point[5]
        dz_dw7 = point[6]
        dz_dw8 = point[7]
        dz_db = 1
        dcost_dw1  = derivative_cost_pred * derivative_pred_z * dz_dw1
        dcost_dw2 =  derivative_cost_pred * derivative_pred_z * dz_dw2
        dcost_dw3  = derivative_cost_pred * derivative_pred_z * dz_dw3
        dcost_dw4 =  derivative_cost_pred * derivative_pred_z * dz_dw4
        dcost_dw5  = derivative_cost_pred * derivative_pred_z * dz_dw5
        dcost_dw6 =  derivative_cost_pred * derivative_pred_z * dz_dw6
        dcost_dw7  = derivative_cost_pred * derivative_pred_z * dz_dw7
        dcost_dw8 =  derivative_cost_pred * derivative_pred_z * dz_dw8
        dcost_db = derivative_cost_pred * derivative_pred_z * dz_db
        
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        w3 = w3 - learning_rate * dcost_dw3
        w4 = w4 - learning_rate * dcost_dw4
        w5 = w5 - learning_rate * dcost_dw5
        w6 = w6 - learning_rate * dcost_dw6
        w7 = w7 - learning_rate * dcost_dw7
        w8 = w8 - learning_rate * dcost_dw8
        b = b - learning_rate * dcost_db
        if i % 100 == 0:
            cost_sum = 0
            for j in range(len(data)):
                p = data[j]
                z  = point[0] * w1 + point[1] * w2 + point[2] * w3 + point[3] * w4 + point[4] * w5 + point[5] * w6 + point[6] * w7 + point[7] * w8 + b
                pred = sigmoid(z)
                cost_sum += np.square(pred)
            costs.append(cost_sum/len(dataset))  
    parameter = [b,w1,w2,w3,w4,w5,w6,w7,w8]
    return parameter,costs
        

    
#plt.plot(costs)
#plt.show()
#for i in costs:
    #print(i)

data = 'diabetes.csv'
dataset = load_csv(data)

dataset.remove(dataset[0])
for row in dataset:
    for i in range(len(row)):
        str_column_to_float(dataset,i)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset,minmax)

learning_rate = 0.01
iters = 100000
parameters,cost = logistic_regression(dataset,learning_rate,iters)

test = dataset[25]
print(test[-1])
for i in range( len(test) - 1):
    parameters[0] += test[i] * parameters[i + 1]
print(sigmoid(parameters[0]))
    

