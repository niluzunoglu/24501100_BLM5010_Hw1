# Utils 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

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
            X_train (): Eğitim verisinin input değerleri
            X_val (): Doğrulama verisinin input değerleri
            X_test (): Test verisinin input değerleri

            y_train (): Eğitim verisinin etiket (label) değerleri
            y_val () : Doğrulama verisinin etiket (label) değerleri
            y_test () : Test verisinin etiket (label) değerleri
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

def write_to_txt(output, file_name="output.txt"):
  """
  Verilen bir çıktıyı bir .txt dosyasına yazar.

  Argümanlar:
      output (str): Yazılacak içerik.
      file_name (str): Dosya adı (varsayılan: "output.txt").
      
    Çıktı:
        Belirli bir çıktısı yok. Dosyaya yazma işlemini gerçekleştirir.
  """
  try:
      with open(file_name, "w", encoding="utf-8") as file:
          file.write(str(output))
      print(f"Çıktı başarıyla '{file_name}' dosyasına yazıldı.")
      
  except Exception as e:
      print(f"Hata: {e}")
      
def write_to_excel(data, file_name="output.xlsx"):
    """
    Verilen bir veri yapısını (sözlük veya DataFrame) bir Excel dosyasına yazar.

    Argümanlar:
        data (dict veya pandas.DataFrame): Yazılacak veri.
        file_name (str): Dosya adı (varsayılan: "output.xlsx").
        
    Çıktı:
        Belirli bir çıktısı yok. Excel dosyasına yazma işlemini gerçekleştirir.
    """
    try:
        # Eğer veri bir sözlükse, DataFrame'e dönüştür
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # DataFrame'i Excel'e yaz
        data.to_excel(file_name, index=False, engine="openpyxl")
        print(f"Çıktı başarıyla '{file_name}' dosyasına yazıldı.")
        
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