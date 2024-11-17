
from utils import load_data, split_data
from LogisticRegressionClass import LogisticRegression

X,y = load_data("dataset/hw1Data.txt")
data = split_data(X,y)

logistic_regression = LogisticRegression(data, learning_rate = 0.1, epochs = 150)

print("******************************************")
logistic_regression.fit(data_type = "TRAIN")
print(logistic_regression.logs)
print("******************************************")

""" 
logistic_regression.fit(data_type = "VALIDATION")
print(logistic_regression.logs)
print("******************************************")
logistic_regression.fit(data_type = "TEST")
print(logistic_regression.logs)
"""