from utils import load_data, split_data, write_logs_to_txt, write_costs_to_txt
from common import *
from time import time
from datetime import datetime

class LogisticReggression_Train:
    
    def __init__(self, data, learning_rate=0.01, epochs=1000,  data_type="TRAIN"):
        
        """
        Lojistik regresyon algoritmasında eğitim için kullanılan sınıfın constructor fonksiyonu
        
        Argümanlar:
            data(dict): Verisetindeki tüm veriler
            learning_rate (float): Öğrenme oranı. Default 0.01
            epochs (int): Epoch sayısı. Default 1000
            data_type(String): Hangi veri türüyle öğrenme yapılacağını belirtir. 
                2 değer alabilir - Train ve Validation. Default Train.
        
        """
    
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.logs = {}
        self.data_type = data_type
        
        # Alınan veriden train ve validation verilerinin çekilmesi.
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]

        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        
    
    def train(self):

        """
        Bu fonksiyon, girilen veri tipine göre ("TRAIN","VALIDATION" olabilir)
        verilerin üzerinde Lojistik Regresyon algoritmasını çalıştırır.
        Tek girdisi tiptir. Verinin kendisini, learning_rate, epoch sayısı gibi bilgileri
        sınıf değişkenlerinden alır. 

        Girdi:
            Sınıf değişkenlerinin çoğu burada kullanılır.
        
        Çıktı:
          Belirli bir çıktısı yok. 
          Algoritma çalışma bilgilerinin bulunduğu logs değişkenine atama yapar.
          Eğer istenirse cost/epoch grafiğini çizdirir. 
        """

        X, y = 0,0 

        if self.data_type == "TRAIN":
          X = self.X_train
          y = self.y_train

        elif self.data_type == "VALIDATION":
          X = self.X_val
          y = self.y_val
      
        # Ağırlıkları ilklendirme
        self.weights = np.zeros(X.shape[1])
        #self.bias = np.random.rand()
        
        self.bias = 7 # Random seçilmiş bir sayı
        
        # Ağırlıklar ile ilk tahminin yapılması
        y_predicted = calculate_prediction(X, self.weights, self.bias)

        # İlk tahmin için cross entropy loss hesabının yapılması
        average_cost = calculate_average_cross_entropy_loss(y, y_predicted)
        self.weights = calculate_stochastic_gradient_descent(self.weights, X, y, y_predicted, self.learning_rate, y.size)

        cost_list = []

        for i in range(self.epochs):
            
            y_predicted = calculate_prediction(X, self.weights, self.bias)
            
            losses_for_each_sample = np.array([calculate_cross_entropy_loss(y_true, y_pred_i) for y_true, y_pred_i in zip(y, y_predicted)])

            cost = calculate_average_cross_entropy_loss(y, y_predicted)
            cost_list.append({"epoch":i,"cost":float(cost)})

            # SGD algoritması ile yeni ağırlıkların hesaplanması
            self.weights = calculate_stochastic_gradient_descent(self.weights, X, y, y_predicted, self.learning_rate, y.size)

            # Belirli epochlarda cost değerlerini görebilmek için yazdırılmıştır.
            #if i % 10 == 0:
            #    print('Cost: {}'.format(cost))

        # En son epoch sonrası ağırlık değerleri
        print('Final weights: {}'.format(self.weights))

        values = calculate_results(y, y_predicted)
        print("Accuracy : ", calculate_accuracy(values))
        print("Precision : ", calculate_precision(values))
        print("Recall :", calculate_recall(values))
        print("Fscore:", calculate_fscore(values))

        # Logların doldurulması
        self.logs["data_type"] = self.data_type
        self.logs["epoch"] = self.epochs
        self.logs["learning_rate"] = self.learning_rate
        self.logs["accuracy"] = calculate_accuracy(values)
        self.logs["precision"] = calculate_precision(values)
        self.logs["recall"] = calculate_recall(values)
        self.logs["fscore"] = calculate_fscore(values)
        self.logs["cost_list"] = cost_list
        self.logs["datetime"] = datetime.fromtimestamp(time()).strftime("%d-%m-%Y %H:%M:%S")
        
    
    def save_model(self):
        
        """
        Modelin weight ve bias değerlerini npy uzantılı dosyalara kaydeden fonksiyon.
        """
        print("[*] Weights to save : ",self.weights)
        np.save("saved_models/savedWeights.npy", self.weights)
    
        print("[*] Bias to save : ",self.bias)
        np.save("saved_models/savedBias.npy", self.bias)
        
    
    def save_logs(self):
        
        """
        Modelin çalışma bilgilerini results/ dizini altında ilgili txt dosyasına ekleyen fonksiyon.
        """
        return write_logs_to_txt(self.logs, f"results/{self.data_type}_metrics.txt")

    def save_costs_per_epoch(self):
        """
        Detay bilgiye ihtiyaç duyulursa, her bir epoch için cost değerlerini dosyaya yazan fonksiyon.
        """
        return write_costs_to_txt(self.logs, f"results/costs-per-epoch/{self.data_type}_lr_{self.learning_rate}_epoch_{self.epochs}_costs_per_epoch.txt")


if __name__ == "__main__":
    
    print("[*] Now Running --- Logistic Regression - Train.py ")
    
    dataset_X, dataset_y = load_data("dataset/examResultsAndLabels.txt")
    data = split_data(dataset_X, dataset_y)

    try:
        learning_rate = float(input("Learning rate giriniz: (Girilmezse 0.1 alınacaktır) "))
    except ValueError as e:
        learning_rate = 0.1
    
    try:
        epochs = int(input("Epoch sayısı giriniz: (Girilmezse 1000 alınacaktır) "))
    except ValueError as e:
        epochs = 1000
        
    data_type = str(input("Eğitim verisi türü seçiniz: (Girilmezse Train alınacaktır) (T for Train, V for Validation) : "))
    
    if data_type.capitalize() == "T":
        data_type = "TRAIN"
        
    elif data_type.capitalize() == "V":
        data_type = "VALIDATION"
        
    else:
        data_type = "TRAIN"
   
    if(epochs and learning_rate and data_type and learning_rate>0 and epochs>0):
        train_Logistic_Regression = LogisticReggression_Train(data=data, learning_rate=learning_rate, epochs=epochs, data_type=data_type)
    
    else:
        train_Logistic_Regression = LogisticReggression_Train(data=data, learning_rate=0.1, epochs=1000, data_type=data_type)
    
    train_Logistic_Regression.train()
    train_Logistic_Regression.save_model()
    train_Logistic_Regression.save_logs()    
    train_Logistic_Regression.save_costs_per_epoch()