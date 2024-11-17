

## Açıklama
Bu kod reposunda Makine Öğrenmesi (BLM5110) dersi kapsamında Python kullanılarak geliştirilmiş bir sınıflandırma modeli bulunmaktadır. Bu model, Makine Öğrenmesi dersinin 1. ödevi için hazırlanmıştır.

___

## Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler ve versiyonları requirements.txt dosyasında bulunmaktadır. Bu kütüphaneler 

``` pip install -r requirements.txt ```

komutu ile yüklenebilir.
___

## Çalıştırma

Projenin eğitim kısmı train.py dosyasının içerisindeki fonksiyonlar ile yapılmaktadır. Bu dosya

``` python train.py ```

 komutu kullanılarak çalıştırılabilir.

Kaydedilen ağırlıklar ile değerlendirmeler test.py dosyasından yapılmaktadır. Bu dosya 

``` python test.py ```

 komutu kullanılarak çalıştırılabilir.
___

## Klasör Yapısı

    proje_klasoru/

        eval.py            # Model değerlendirme dosyası

        test.py            # Yazılan fonksiyonların testlerinin bulunduğu dosya

        train.py            # Eğitimlerin yapıldığı dosya

        utils.py          # Yardımcı fonksiyonların bulunduğu dosya

        common.py          # test.py ve train.py dosyalarının ortak fonksiyonlarının bulunduğu dosya

        requirements.txt   # Gerekli kütüphaneler

        dataset/           # Veri setlerini içerir

            examResultsAndLabels.txt      # Eğitim, Doğrulama ve Test verileri

        results/           # Sonuçların saklandığı klasör

            costs-per-epoch/ # Ayrıntılı logların (her bir epoch için kayıp miktarı) saklandığı dizin

            TEST_metrics.txt    # Test metriklerini barındıran dosya.

            TRAIN_metrics.txt   # Eğitim metriklerini barındıran dosya.

            VALIDATION_metrics.txt  # Validasyon metriklerini barındıran dosya

        saved_models/       # Kaydedilen model ağırlıkları burada bulunur.

        graphs/          # Üretilen grafikler burada bulunur.
___

### Öğrenci Bilgileri

24501100 - Aleyna Nil Uzunoğlu

YTÜ Bilgisayar Mühendisliği Tezli YL Öğrencisi
___
