

## Açıklama
Bu kod reposunda Makine Öğrenmesi (BLM5110) dersi kapsamında Python kullanılarak geliştirilmiş bir sınıflandırma modeli bulunmaktadır. Bu model, Makine Öğrenmesi dersinin 1. ödevi için hazırlanmıştır.

## Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler ve versiyonları requirements.txt dosyasında bulunmaktadır. Bu kütüphaneler 

> pip install -r requirements.txt

komutu ile yüklenebilir.

## Çalıştırma
Logistic Regression sınıfından oluşturulacak bir obje ile,
eğitim ve değerlendirme main.py dosyasındaki main fonksiyonunda yapılmaktadır. Bu fonksiyon

> python main.py

 komutu kullanılarak çalıştırılabilir.

## Klasör Yapısı
proje_klasoru/
├── train.py           # Model eğitimi için ana dosya
├── eval.py            # Model değerlendirme dosyası
├── requirements.txt   # Gerekli kütüphaneler
├── dataset/           # Veri setlerini içerir
│   ├── train.csv      # Eğitim verisi
│   ├── test.csv       # Test verisi
├── results/           # Sonuçların saklandığı klasör
│   ├── model_output/  # Eğitilen model çıktıları
│   ├── logs/          # Eğitim sırasında üretilen log dosyaları


### Öğrenci Bilgileri

24501100 - Aleyna Nil Uzunoğlu
YTÜ Bilgisayar Mühendisliği Tezli YL Öğrencisi