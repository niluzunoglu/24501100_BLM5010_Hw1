# Utils 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_data(filepath):

    """
    TXT dosyasından veriyi yükler, X (özellikler) ve y(etiketler) olarak ayırır.

    Argümanlar:
        filepath (str): Verinin bulunduğu dosyanın yolu

    Çıktı:
        İki ayrı veri yapısı olacak şekilde Özellikler (X) ve etiketler (y).
    """

    data = pd.read_csv(filepath, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def split_data(X, y):

    """
    Veriyi, ödev dokümanında istendiği gibi eğitim (%60), doğrulama (%20) ve test (%20) olarak böler.

    Argümanlar:
        X (DataFrame): Özelliklerden oluşan veri
        y (Series): Etiketlerden oluşan veri

    Çıktı:
    
        data (dict) : Aşağıda bulunan değerleri içeren sözlük yapısı
            X_train (pandas.core.frame.DataFrame): Eğitim verisinin input değerleri
            X_val (pandas.core.frame.DataFrame): Doğrulama verisinin input değerleri
            X_test (pandas.core.frame.DataFrame): Test verisinin input değerleri

            y_train (pandas.core.series.Series): Eğitim verisinin etiket (label) değerleri
            y_val (pandas.core.series.Series) : Doğrulama verisinin etiket (label) değerleri
            y_test (pandas.core.series.Series) : Test verisinin etiket (label) değerleri
    """
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    data = {
        "X_train": X_train, "X_val":X_val, "X_test":X_test, 
        "y_train": y_train, "y_val": y_val, "y_test": y_test
    }
    return data

def plot_data(X, y, title, x_label, y_label):
    """
    Verinin verilen değerlere göre grafiğini oluşturur ve kaydeder.

    Argümanlar:
        X (DataFrame): Özellikler.
        y (Series): Etiketler.
        Title (str): Grafiğin başlığı
        x_label (String) : X ekseninin başlığı
        y_label (String) : Y ekseninin başlığı

    Çıktı:
      Grafik.
    """

    class_0 = X[y == 0]
    class_1 = X[y == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(class_0.iloc[:, 0], class_0.iloc[:, 1], color='orange', label='İşe Alınmamış')
    plt.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], color='purple', label='İşe Alınmış')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()

    plt.savefig(title+".jpg") 

def write_logs_to_txt(logs, file_name="output.txt"):

    """
    Verilen log verilerini hizalı bir şekilde bir .txt dosyasına yazar.

    Argümanlar:
        logs (dict): Log verilerini içeren sözlük.
        file_name (str): Dosya adı (varsayılan: "logs.txt").
    """
    
    # Dosya yolunun bulunduğu dizini al
    directory = os.path.dirname(file_name)

    # Dizin yoksa oluştur
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    try:
        with open(file_name, "a", encoding="utf-8") as file:
            
            file.write(f"{'Log Key':<20}{'Log Value':<20}\n")
            file.write("=" * 40 + "\n")
            
            for key, value in logs.items():
                
                if(key=="cost_list"):
                    pass
                else:
                    file.write(f"{key:<20}{str(value):<20}\n")
                
            file.write("=" * 40 + "\n")
        
        print(f"Loglar başarıyla '{file_name}' dosyasına yazıldı.")
    except Exception as e:
        print(f"Hata: {e}")

def write_costs_to_txt(logs, file_name="costs_per_epoch.txt"):

    """
    Verilen cost verilerini hizalı bir şekilde bir .txt dosyasına yazar.

    Argümanlar:
        logs (dict): Log verilerini içeren sözlük.
        file_name (str): Dosya adı (varsayılan: "costs_per_epoch.txt").
    """
    
    directory = os.path.dirname(file_name)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    try:
        with open(file_name, "a", encoding="utf-8") as file:
            
            file.write(f"{'Epoch':<20}{'Cost':<20}\n")
            file.write("=" * 40 + "\n")
            
            for log in logs["cost_list"]:
                file.write(f"{log['epoch']:<20}{log['cost']:<20}\n")
                
            file.write("=" * 40 + "\n")
        
        print(f"Loglar başarıyla '{file_name}' dosyasına yazıldı.")
    except Exception as e:
        print(f"Hata: {e}")
    
def plot_train_val_graph(train_loss, validation_loss):

  # X ekseni için örneklerin sırası
  x_axis = np.arange(1, len(train_loss) + 1)

  # Grafik çizimi
  plt.figure(figsize=(8, 6))

  plt.plot(x_axis, train_loss, marker="o", label="Train Data Loss", color="blue")
  plt.plot(x_axis, validation_loss, marker="x", label="Validation Data Loss", color="orange")

  # Grafik düzenlemeleri
  plt.xlabel("Örnek No")
  plt.ylabel("Cross-Entropy Loss")
  plt.title("Loss Değişim Grafiği (Train ve Validation Verileri)")
  plt.legend()  
  plt.grid()  
  plt.show()
  plt.savefig("Loss Değişim Grafiği (Train ve Validation Verileri)"+".jpg") 