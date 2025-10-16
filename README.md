# Türkiye İkinci El Araç Fiyat Tahmini için Açıklanabilir Makine Öğrenmesi

Bu proje, Türkiye'deki ikinci el araç piyasası için makine öğrenmesi modelleri kullanarak fiyat tahmini yapmayı ve bu tahminlerin hangi faktörlere dayandığını Açıklanabilir Yapay Zeka (Explainable AI - XAI) yöntemleriyle anlaşılır kılmayı amaçlamaktadır.

##  Projenin Amacı

* **Doğru Fiyat Tahmini:** Geniş bir veri setini kullanarak ikinci el araçlar için yüksek doğrulukta fiyat tahminleri üreten bir makine öğrenmesi modeli geliştirmek.
* **Modelin Yorumlanabilirliği:** Fiyat tahminlerinin hangi kriterlere (örneğin; aracın yılı, markası, modeli, kilometresi, motor gücü vb.) dayandığını şeffaf bir şekilde ortaya koymak.
* **Şeffaflık:** Modelin karar mekanizmasını SHAP, LIME gibi XAI teknikleri kullanarak görselleştirmek ve kullanıcılar için anlaşılır hale getirmek.

## Kullanılan Teknolojiler

* **Programlama Dili:** Python
* **Veri Analizi ve İşleme:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-learn, XGBoost, LightGBM [Kullandığınız diğer kütüphaneler]
* **Açıklanabilir Yapay Zeka (XAI):** SHAP, LIME
* **Veri Görselleştirme:** Matplotlib, Seaborn
* **Veritabanı:** PostgreSQL

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1.  **Depoyu klonlayın:**
    ```sh
    git clone [https://github.com/SEYFULLAH01/arac_fiyat_tahmin.git](https://github.com/SEYFULLAH01/arac_fiyat_tahmin.git)
    ```
2.  **Proje dizinine gidin:**
    ```sh
    cd arac_fiyat_tahmin
    ```
3.  **Gerekli kütüphaneleri yükleyin:**
    (Projenize bir `requirements.txt` dosyası eklemeniz tavsiye edilir.)
    ```sh
    pip install -r requirements.txt
    ```

## Kullanım

1.  **Veri Seti:** Projede kullanılan veri seti, [Veri setini nereden aldığınızı buraya yazın, örn: arabam.com üzerinden toplanmıştır]. Veri setini `data/` klasörü altına yerleştirin.
2.  **Model Eğitimi:** Modeli eğitmek için aşağıdaki komutu çalıştırın:
    ```sh
    python train_model.py
    ```
3.  **Tahmin Yapma:** Eğitilmiş modeli kullanarak tahmin yapmak için:
    ```sh
    python predict.py
    ```

## Geliştiriciler

* **Seyfullah Adıgüzel**
## Danışman

* **Sinan Keskin**
