from random import seed
from random import randrange
from csv import reader
from math import exp


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


def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min,value_max])
    return minmax
                      

def normalize_dataset(dataset,minmax):
    for row in dataset:
        for  i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])
                                              
                                                                                            
                    
                      
def cross_validation_split(dataset,n_fold):
    dataset_split = list()
    dataset_copy = list(dataset)
    for i in range(n_fold):
        index = randrange(len(dataset_copy))
        dataset_split.append(dataset_copy.pop(index))
    return dataset_split
            

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) * 100.0


def evaluation_algorithm(dataset,algorithm,n_fold,*args):
    train_set = cross_validation_split(dataset,n_fold)
    scores = list()
    test_set = list(dataset)
    for fold in train_set:
        test_set.remove(fold)
    predicted = algorithm(train_set,test_set,*args)
    actual = [row[-1] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy


def predict(row,coefficient):
    yhat = coefficient[0]
    for i in range(len(row) - 1):
        yhat += coefficient[i + 1] * row[i]
    return 1.0/(1.0 + exp(-yhat))



def coefficients_sgd(train, lrate, iters):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(iters):
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + lrate * error * yhat * (1.0 - yhat)
            for i in range(len(row)- 1):
                coef[i + 1] = coef[i + 1] + lrate * error * yhat * (1.0 - yhat) * row[i]
    return coef


def logistic_regression(train,lrate,iters):
     predictions = list()
     coef = coefficients_sgd(train,lrate,iters)
     for row in train:
         yhat = predict(row,coef)
         #yhat = round(yhat)
         predictions.append(yhat)
     return predictions


data = 'diabetes.csv'

dataset = load_csv(data)
dataset.remove(dataset[0])
for row in dataset:
        for i in range(len(row)):
            str_column_to_float(dataset,i)


minmax = dataset_minmax(dataset)
normalize_dataset(dataset,minmax)
n_fold = 1000
lrate = 0.1
iters =  100


Y_pred = logistic_regression(dataset,lrate,iters)
print(Y_pred[25])
