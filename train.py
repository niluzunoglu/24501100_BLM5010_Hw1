from utils import load_data, split_data, write_logs_to_txt, write_costs_to_txt
from common import *
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

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
        
        self.X = None
        self.y = None
    
        # Alınan veriden train ve validation verilerinin çekilmesi.
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]

        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        
        
        print("[*] Train ve validasyon verileri çekildi.")
        
    
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


        if self.data_type == "TRAIN":
            self.X = self.X_train
            self.y = self.y_train

        elif self.data_type == "VALIDATION":
            self.X = self.X_val
            self.y = self.y_val
      
        # Ağırlıkları ilklendirme
        self.weights = np.zeros(self.X.shape[1])
        #self.bias = np.random.rand()
        self.bias = 0 # Random seçilmiş bir sayı
        
        """ 
        # Ağırlıklar ile ilk tahminin yapılması
        y_predicted = calculate_prediction(X, self.weights, self.bias)

        # İlk tahmin için cross entropy loss hesabının yapılması
        average_cost = calculate_average_cross_entropy_loss(y, y_predicted)
        self.weights = calculate_stochastic_gradient_descent(self.weights, X, y, y_predicted, self.learning_rate)
        """
        self.cost_list = []
        
        for i in range(self.epochs):
            
            # Bu döngü her bir epoch için 1 defa döner.
            
            print("[*] Epoch - ",i)
            total_loss = 0 
            self.y_predicted = np.zeros(len(self.y)) 
            
            for j in range(0, len(self.y)):
                
                # Bu döngü her bir örnek için bir defa döner.
                # Bu döngü içerisinde tahmin yapılır, loss ve yeni weightler hesaplanır.
                
                x_sample = self.X.iloc[j]
                y_sample = self.y.iloc[j]
                
                # Prediction yapılır.
                y_predicted_sample = calculate_prediction(x_sample, self.weights, self.bias)
                                
                # Örnek başına kayıp hesaplanır.
                loss = calculate_cross_entropy_loss_for_one_sample(y_sample, y_predicted_sample)
                
                total_loss += loss 
                
                # Ağırlıkları stochastic gradient descent ile güncelle
                self.weights = calculate_stochastic_gradient_descent(self.weights, x_sample, y_sample, y_predicted_sample, self.learning_rate)

                self.y_predicted[j] = y_predicted_sample

            # Ortalama epoch kaybı
            cost = total_loss / len(self.y)
            print("[*] Cost -",cost)
            
            self.cost_list.append({"epoch": i, "cost": float(cost)})

            # Belirli epochlarda cost değerlerini yazdır (isteğe bağlı)
            # if i % 10 == 0:
            #    print(f"Epoch {i}, Cost: {cost}")

        # En son epoch sonrası ağırlık değerleri
        print('Final weights: {}'.format(self.weights))

        values = calculate_results(self.y, self.y_predicted)
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
        self.logs["cost_list"] = self.cost_list
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

    def plot_cost_from_cost_list(self):
        """
        Her epoch için kaydedilen cost verisinden bir grafiği çizer.
        
        Args:
            cost_list (list): Her eleman {"epoch": i, "cost": value} formatında bir liste.
        """
        # Epoch ve cost değerlerini ayır
        epochs = [entry["epoch"] for entry in self.cost_list]
        costs = [entry["cost"] for entry in self.cost_list]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, costs, label='Ortalama Cross-Entropy Loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Ortalama Cross-Entropy Loss')
        plt.title('Eğitim Süresince Loss Değişimi')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"graphs/cost_sample_Graphs/{self.epochs}_{self.learning_rate}.png")

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
    train_Logistic_Regression.plot_cost_from_cost_list()