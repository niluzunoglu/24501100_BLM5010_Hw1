
# Bu dosya, Lojistik Regresyon algoritması için hem train hem de eval
# dosyalarında kullanılacak olan ortak fonksiyonların tanımlarını içerir.

import numpy as np

def calculate_sigmoid(z):
    """
    Bu fonksiyon, verilen Z değerinin sigmoid fonksiyonundaki karşılığını hesaplar ve döndürür.
    Sigmoid fonksiyonu, modelin verdiği cevaptan ne kadar emin olduğunu belirtir.
    Çıktısı her zaman 0 ve 1 arasında olacaktır.
    Formülü, 1 / (1 + e^(-z)) şeklindedir. Z değeri, modelden alınan tahmindir (prediction).

    Argümanlar: 
        Z (ndarray): Modelin çıktısı, y_predicted'i temsil eder.

    Çıktı: 
        Z'nin sigmoid fonksiyonundaki karşılığı. 0 ile 1 arasındadır.
    """

    return (1/(1+np.exp(-z)))

def calculate_results(y_target, y_pred):

    """
        Bu fonksiyon, tahmin edilen ve gerçek etiket(y) değerlerini alır ve bunlar üzerinden çıkarımlar yapar.
        True Negative, True Positive, False Negative, False Positive değerlerini hesaplar.

        Girdi:
        y_target (pandas.core.series.Series): Doğru etiket değerleri.
        y_pred (numpy.ndarray): Algoritma tarafından tahmin edilen etiket değerleri.

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

def calculate_accuracy(values):
    """
    Bu fonksiyon, accuracy değerini döndürür.
    """
    return ( values["true_positive"] + values["true_negative"] ) / values["total"]

def calculate_precision(values):
    """
    Bu fonksiyon, precision değerini döndürür.
    """

    return ( values["true_positive"] / (values["true_positive"] + values["false_positive"]))

def calculate_recall(values):
    """
    Bu fonksiyon, recall değerini döndürür.
    """

    return ( values["true_positive"] / (values["true_positive"] + values["false_negative"]))

def calculate_fscore(values):
    """
    Bu fonksiyon, f-score değerini döndürür.
    """
    return (2*((calculate_recall(values)*calculate_precision(values)) / (calculate_recall(values)+calculate_precision(values))))
    
def calculate_prediction(X, w, b):

    """
    Bu fonksiyon, özelliklerin değerlerini (X) ve weightleri alır. 
    Sonucunda bir tahmin oluşturur. Bu tahmini sigmoid fonksiyonuna verir.
    Sigmoid fonksiyonunun çıktısı 0.5'den büyük ise 1, değilse 0 döndürür.

    Argümanlar:
        X (pandas.core.frame.DataFrame) : Özelliklerden oluşan matris.
        w (numpy.ndarray) : Ağırlıklar matrisi. 
        b (int): Bias değeri (w0)

    Çıktı: 
        Tahmin edilen Y değeri. (0 veya 1)

    """

    z = np.dot(X, w) + b
    y_predicted = calculate_sigmoid(z)
    return np.where(y_predicted > 0.5, 1, 0)

def calculate_cross_entropy_loss(y_target, y_predicted):

    """
    Bu fonksiyon, kurulan modelin çıktısı olan y_predicted matrisini ve y_target matrisini alır.
    İkisi arasındaki farka bağlı olan loss metriğini döndürür.

    epsilon ayarlaması, np.log() fonksiyonlarının içi 0 olduğunda uyarı mesajı verdiği için uygulanmıştır.
    Bu ayarlama sayesinde y_predicted değerinin 0 ve 1 olması kontrol altında tutulur.

    Argümanlar:
        y_predicted(pandas.core.series.Series): Modelin çıktısı olan sonuçlar
        y_target(numpy.ndarray): Gerçek sonuçlar.

    Çıktı:
        Cross Entropy Loss değeri.
    """

    epsilon = 1e-15  
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon) 

    part1 = y_target * np.log(y_predicted)
    part2 = (1-y_target) * np.log(1-y_predicted)

    return -(part1+part2)

def calculate_average_cross_entropy_loss( y_target, y_predicted):

    """
    Bu fonksiyon her bir örnek için ortalama cross entropy loss değerini hesaplar.

    Argümanlar:
        y_predicted(pandas.core.series.Series): Modelin çıktısı olan sonuçlar
        y_target(numpy.ndarray): Gerçek sonuçlar.

    Çıktı:
        Cross Entropy Loss ortalama değeri.
    """
    
    return calculate_cross_entropy_loss(y_target, y_predicted).mean()

def calculate_stochastic_gradient_descent(weights, x_train, y_train, y_predicted, learning_rate, number_of_samples):

    """
    Bu fonksiyon, bir ağırlıklar dizisini alır (weights), ve bir önceki epochta tahmin edilen değere, 
    kullanılan verinin sayısına, ve diğer çeşitli argümanlara göre bir gradient 
    vektörü hesaplar. Bu vektöre göre learning rate'le birlikte yeni weigthlere karar verirler.
    """

    gradient = np.dot(x_train.T, (y_predicted - y_train)) / number_of_samples
    weights -= learning_rate * gradient

    return weights 
