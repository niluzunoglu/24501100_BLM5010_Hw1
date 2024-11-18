from scipy.special import expit
from common import * 
import keras
from keras import backend as k
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_data,split_data

# One Logistic Regression object to call the functions from the class

X,y = load_data("dataset/examResultsAndLabels.txt")
data = split_data(X,y)

def test_sigmoid():

    test_array = list(range(-3,10))

    true_count = 0

    for i in test_array:

        if(expit(i) != calculate_sigmoid(i)):
            print("[*] Yanlış var -",i)
        else:
            true_count += 1

    if true_count == len(test_array):
    # Hepsini doğru bilmiş demektir.
        print("[*] Algorithm works good. Congratulations :) ")
    else:
        print("Your score is : ", (true_count/len(test_array)))
            
def test_cross_entropy_loss():
    
    len_of_test_array = 100
    
    y_true = np.random.randint(0, 2, size=len_of_test_array)  
    y_pred = np.random.rand(len_of_test_array) 

    true_count = 0
    loss_real = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss_found = calculate_average_cross_entropy_loss(y_true, y_pred)

    
    epsilon = 1e-6  # küçük bir tolerans
    are_equal = abs(loss_real - loss_found) < epsilon
    print(are_equal)  

    if are_equal:
        # Hepsini doğru bilmiş demektir.
        print("[*] Algorithm works good. Congratulations :) ")
    else:
        print("Your score is : ", (true_count/len_of_test_array))
        
def test_metrics():
    
    np.random.seed(0)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    values = calculate_results(y_true, y_pred)

    def are_equal(real, found):
        epsilon = 1e-6 
        are_equal = abs(real - found) < epsilon
        return are_equal

    all_true = True

    if not(are_equal(accuracy_score(y_true, y_pred), calculate_accuracy(values))):
        print("[*] Accuracy ölçümü yanlış")
        all_true = False

    if not(are_equal(precision_score(y_true, y_pred), calculate_precision(values))):
        print("[*] Precision ölçümü yanlış")
        all_true = False

    if not(are_equal(recall_score(y_true, y_pred), calculate_recall(values))):
        print("[*] Recall ölçümü yanlış")
        all_true = False

    if not(are_equal(f1_score(y_true, y_pred), calculate_fscore(values))):
        print("[*] F1Score ölçümü yanlış")
        all_true = False

    if all_true:
        print("[*] Tests passed.")
        
        
if __name__ == "__main__":
    test_cross_entropy_loss()
    test_metrics()
    test_sigmoid()