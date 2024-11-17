import numpy as np
from utils import * 

class LogisticRegression:

    """
    Bu class, içerisinde Lojistik Regresyon ile ilgili fonksiyonları barındırır.
    """

    def __init__(self, data, learning_rate=0.01, epochs=1000):

        """
        
        """
      
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.logs = {}

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]

        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

    def calculate_sigmoid(self, z):
      """
      Bu fonksiyon, verilen Z değerinin sigmoid fonksiyonundaki karşılığını hesaplar ve döndürür.
      Sigmoid fonksiyonu, modelin verdiği cevaptan ne kadar emin olduğunu belirtir.
      Çıktısı her zaman 0 ve 1 arasında olacaktır.
      Formülü, 1 / (1 + e^(-z)) şeklindedir. Z değeri, modelden alınan tahmindir (prediction).

      Argümanlar: 
        Z ():

      Çıktı: Z'nin sigmoid fonksiyonundaki karşılığı. 
      """

      return (1/(1+np.exp(-z)))

    def calculate_results(self,y_target, y_pred):

        """
          Bu fonksiyon, tahmin edilen ve gerçek etiket(y) değerlerini alır ve bunlar üzerinden çıkarımlar yapar.
          True Negative, True Positive, False Negative, False Positive değerlerini hesaplar.

          Girdi:
            y_target(): Doğru etiket değerleri.
            y_pred(): Algoritma tarafından tahmin edilen etiket değerleri.

          Çıktı:
            values (dictionary): Algoritmaların True Negative, True Positive, 
                              False Negative, False Positive ve Toplam değerini içeren sözlük.
        """

        y_target = np.array(y_target)
        y_pred = np.array(y_pred)

        tp = np.sum((y_target == 1) & (y_pred == 1))
        tn = np.sum((y_target == 0) & (y_pred == 0))
        fp = np.sum((y_target == 0) & (y_pred == 1))
        fn = np.sum((y_target == 1) & (y_pred == 0))

        values = {"true_positive":tp, "true_negative":tn, "false_positive":fp, "false_negative":fn, "total": tp+tn+fp+fn}
        return values

    def calculate_accuracy(self, values):
      """
        Bu fonksiyon, accuracy değerini döndürür.
      """
      return ( values["true_positive"] + values["true_negative"] ) / values["total"]

    def calculate_precision(self, values):
      """
        Bu fonksiyon, precision değerini döndürür.
      """

      return ( values["true_positive"] / (values["true_positive"] + values["false_positive"]))

    def calculate_recall(self,values):
      """
        Bu fonksiyon, recall değerini döndürür.
      """

      return ( values["true_positive"] / (values["true_positive"] + values["false_negative"]))

    def calculate_fscore(self,values):
      """
        Bu fonksiyon, f-score değerini döndürür.
      """
      return (2*((self.calculate_recall(values)*self.calculate_precision(values)) / (self.calculate_recall(values)+self.calculate_precision(values))))
      
    def calculate_prediction(self, X, w):

        """
        Bu fonksiyon, özelliklerin değerlerini (X) ve weightleri alır. 
        Sonucunda bir tahmin oluşturur. Bu tahmini sigmoid fonksiyonuna verir.
        Sigmoid fonksiyonunun çıktısı 0.5'den büyük ise 1, değilse 0 döndürür.

        Argümanlar:
          X () : Özelliklerden oluşan matris.
          w () : Ağırlıklar matrisi. 

        Çıktı: 
          Tahmin edilen Y değeri. (0 veya 1)

        """
        z = np.dot(X, w)
        y_predicted = self.calculate_sigmoid(z)
        return np.where(y_predicted > 0.5, 1, 0)

    def calculate_cross_entropy_loss(self, y_target, y_predicted):

        """
        Bu fonksiyon, kurulan modelin çıktısı olan y_predicted matrisini ve y_target matrisini alır.
        İkisi arasındaki farka bağlı olan loss metriğini döndürür.

        epsilon ayarlaması, np.log() fonksiyonlarının içi 0 olduğunda uyarı mesajı verdiği için uygulanmıştır.
        Bu ayarlama sayesinde y_predicted değerinin 0 ve 1 olması kontrol altında tutulur.

        Argümanlar:
          y_predicted(): Modelin çıktısı olan sonuçlar
          y_target(): Gerçek sonuçlar.

        Çıktı:
          Cross Entropy Loss değeri.
        """

        epsilon = 1e-15  
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon) 

        part1 = y_target * np.log(y_predicted)
        part2 = (1-y_target) * np.log(1-y_predicted)

        return -(part1+part2)

    def calculate_average_cross_entropy_loss(self, y_target, y_predicted):

        """
        Bu fonksiyon her bir örnek için ortalama cross entropy loss değerini hesaplar.

        Argümanlar:
          y_predicted(): Modelin çıktısı olan sonuçlar
          y_target(): Gerçek sonuçlar.

        Çıktı:
          Cross Entropy Loss ortalama değeri.
        """

        return self.calculate_cross_entropy_loss(y_target, y_predicted).mean();

    def calculate_stochastic_gradient_descent(self, weights, x_train, y_train, y_predicted, learning_rate, number_of_samples):

        """
        Bu fonksiyon, bir ağırlıklar dizisini alır (weights), ve bir önceki epochta tahmin edilen değere, 
        kullanılan verinin sayısına, ve diğer çeşitli argümanlara göre bir gradient 
        vektörü hesaplar. Bu vektöre göre learning rate'le birlikte yeni weigthlere karar verirler.
        """

        gradient = np.dot(x_train.T, (y_predicted - y_train)) / number_of_samples
        weights -= learning_rate * gradient

        return weights 

    def fit(self, data_type):

        """
        Bu fonksiyon, girilen veri tipine göre ("TRAIN","TEST","VALIDATION" olabilir)
        verilerin üzerinde Lojistik Regresyon algoritmasını çalıştırır. 
        Tek girdisi tiptir. Verinin kendisini, learning_rate, epoch sayısı gibi bilgileri
        sınıf değişkenlerinden alır. 

        Girdi:
          data_type (String) :  ["TRAIN","TEST","VALIDATION"] değerlerinden biri.
        
        Çıktı:
          Belirli bir çıktısı yok. 
          Algoritma çalışma bilgilerinin bulunduğu logs değişkenine atama yapar.
          Eğer istenirse cost/epoch grafiğini çizdirir. 
        """

        X, y = 0,0 

        if data_type == "TRAIN":
          X = self.X_train
          y = self.y_train

        elif data_type == "VALIDATION":
          X = self.X_val
          y = self.y_val

        elif data_type == "TEST":
          X = self.X_test
          y = self.y_test
      
        # Ağırlıkları ilklendirme
        weights = np.zeros(X.shape[1])

        # Ağırlıklar ile ilk tahminin yapılması
        y_predicted = self.calculate_prediction(X, weights)

        # İlk tahmin için cross entropy loss hesabının yapılması
        average_cost = self.calculate_average_cross_entropy_loss(y, y_predicted)
        weights = self.calculate_stochastic_gradient_descent(weights, X, y, y_predicted, self.learning_rate, y.size)

        cost_list = []

        for i in range(self.epochs):
          y_predicted = self.calculate_prediction(X, weights)
          cost = self.calculate_average_cross_entropy_loss(y, y_predicted)
          cost_list.append({"epoch":i,"cost":cost})
      
          # SGD algoritması ile yeni ağırlıkların hesaplanması
          weights = self.calculate_stochastic_gradient_descent(weights, X, y, y_predicted, self.learning_rate, y.size)

          # Belirli epochlarda cost değerlerini görebilmek için yazdırılmıştır.
          #if i % 10 == 0:
          #    print('Cost: {}'.format(cost))

        # En son epoch sonrası ağırlık değerleri
        print('Final weights: {}'.format(weights))

        values = self.calculate_results(y, y_predicted)
        print("Accuracy : ", self.calculate_accuracy(values))
        print("Precision : ", self.calculate_precision(values))
        print("Recall :", self.calculate_recall(values));
        print("Fscore:", self.calculate_fscore(values))

        # Logların doldurulması
        self.logs["data_type"] = data_type
        self.logs["epoch"] = self.epochs
        self.logs["learning_rate"] = self.learning_rate
        self.logs["accuracy"] = self.calculate_accuracy(values)
        self.logs["precision"] = self.calculate_precision(values)
        self.logs["recall"] = self.calculate_recall(values)
        self.logs["fscore"] = self.calculate_fscore(values)
        self.logs["cost_list"] = cost_list

    def draw_graphs(self):
      plot_data(self.logs["cost_list"])
