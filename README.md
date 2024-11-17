

## Açıklama
Bu kod reposunda Makine Öğrenmesi (BLM5110) dersi kapsamında Python kullanılarak geliştirilmiş bir sınıflandırma modeli bulunmaktadır. Bu model, Makine Öğrenmesi dersinin 1. ödevi için hazırlanmıştır.

___

## Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler ve versiyonları requirements.txt dosyasında bulunmaktadır. Bu kütüphaneler 

``` pip install -r requirements.txt ```

komutu ile yüklenebilir.
___

## Çalıştırma
Logistic Regression sınıfından oluşturulacak bir obje ile,
eğitim ve değerlendirme main.py dosyasındaki main fonksiyonunda yapılmaktadır. Bu fonksiyon

``` python main.py ```

 komutu kullanılarak çalıştırılabilir.
___

## Klasör Yapısı

    proje_klasoru/

        main.py           # Model eğitimi için ana dosya

        eval.py            # Model değerlendirme dosyası

        test.py            # Yazılan fonksiyonların testlerinin bulunduğu dosya

        LogisticRegressionClass.py # Ana objenin bulunduğu dosya

        requirements.txt   # Gerekli kütüphaneler

        dataset/           # Veri setlerini içerir

            hw1Data.txt      # Eğitim, Doğrulama ve Test verileri

        results/           # Sonuçların saklandığı klasör

        model_output/  # Eğitilen model çıktıları

        logs/          # Eğitim sırasında üretilen log  dosyaları
___

### Öğrenci Bilgileri

24501100 - Aleyna Nil Uzunoğlu

YTÜ Bilgisayar Mühendisliği Tezli YL Öğrencisi
___
