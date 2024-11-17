from utils import load_data, split_data, write_logs_to_txt
import numpy as np
from common import *
from time import time
from datetime import datetime

class LogisticRegression_Eval:
    
    def __init__(self, data):
    
        self.weights = None
        self.bias = None
        self.logs = {}
        self.data_type = "TEST"

        # Alınan veriden test verilerinin çekilmesi.
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        
    def load_model(self):
        
        """
        Bu fonksiyon, train sınıfı tarafından dosyaya kaydedilen weight değerlerini yükler.
        Ve bu değerleri sınıf değişkeninde tutar.
        """
        
        self.weights = np.load("saved_models/savedWeights.npy", allow_pickle=True)
        self.bias = np.load("saved_models/savedBias.npy", allow_pickle=True)
        
        print("[*] Model başarıyla yüklendi.")

    def evaluate_model(self):
        
        """
        Bu fonksiyon, sınıf değişkenlerinde bulunan weights ve bias değerleri ile 
        bir model kurarak prediction yapar. Bu predictionların ne kadar doğru olduğunu
        ölçmek amacıyla metrik hesaplamalarını yapar. Ve logları doldurur.
        """
        y_predicted = calculate_prediction(self.X_test, self.weights, self.bias)
        
        values = calculate_results(self.y_test, y_predicted)
        print("Accuracy : ", calculate_accuracy(values))
        print("Precision : ", calculate_precision(values))
        print("Recall :", calculate_recall(values))
        print("Fscore:", calculate_fscore(values))
        
        # Logların doldurulması
        self.logs["data_type"] = "TEST"
        self.logs["accuracy"] = calculate_accuracy(values)
        self.logs["precision"] = calculate_precision(values)
        self.logs["recall"] = calculate_recall(values)
        self.logs["fscore"] = calculate_fscore(values)
        self.logs["datetime"] = datetime.fromtimestamp(time()).strftime("%d-%m-%Y %H:%M:%S")

        
    def save_results(self):
        
        """
        Bu fonksiyon, modelin çalıştırılması sonucu elde edilen metrikleri ilgili dosyaya kaydeder.
        """
        return write_logs_to_txt(self.logs, f"results/{self.data_type}_metrics.txt")
        
if __name__ == "__main__":
    
    print("[*] Now Running --- Logistic Regression - Eval.py ")
    
    dataset_X, dataset_y = load_data("dataset/examResultsAndLabels.txt")
    data = split_data(dataset_X, dataset_y)
    
    eval_logistic_regression = LogisticRegression_Eval(data=data)
    
    eval_logistic_regression.load_model()
    eval_logistic_regression.evaluate_model()
    eval_logistic_regression.save_results()