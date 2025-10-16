# Türkiye İkinci El Araç Fiyat Tahmini - Proje Raporu

## Proje Özeti

Bu proje, Türkiye ikinci el araç piyasasına özgü verileri kullanarak yüksek doğrulukla fiyat tahmini yapabilen bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Modelin tahminlerini SHAP ve LIME gibi modern Açıklanabilir Yapay Zeka (XAI) teknikleri kullanarak yorumlanabilir hale getirmiştir.

**Hazırlayanlar:** Seyfullah Adıgüzel, Batuhan Orkun İnce  
**Danışman:** Sinan Keskin

## Veri Seti Bilgileri

- **Toplam araç sayısı:** 1,000
- **Özellik sayısı:** 13 (orijinal 9 + türetilmiş 4)
- **Fiyat aralığı:** 53,405 - 314,060 TL
- **Ortalama fiyat:** 132,838 TL

### Özellikler
1. **Marka** - Araç markası
2. **Model** - Araç modeli
3. **Model Yılı** - Araç model yılı
4. **Kilometre** - Toplam kilometre
5. **Motor Hacmi** - Motor hacmi (litre)
6. **Yakıt Türü** - Benzin, Dizel, LPG, Hibrit
7. **Vites Türü** - Manuel, Otomatik, Yarı Otomatik
8. **Gövde Tipi** - Sedan, Hatchback, SUV, vb.
9. **Araç Yaşı** - 2024 - Model Yılı (türetilmiş)
10. **Yıllık Ortalama KM** - Kilometre / Araç Yaşı (türetilmiş)
11. **KM per Motor Hacmi** - Kilometre / Motor Hacmi (türetilmiş)

## Model Performansları

### Karşılaştırma Tablosu

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Random Forest** | **18,343** | **14,735** | **0.8418** |
| XGBoost | 19,380 | 15,439 | 0.8234 |
| Linear Regression | 37,761 | 31,475 | 0.3295 |

### En İyi Model: XGBoost (Log Transform)
- **R² Skoru:** 0.9588 (%95.88 doğruluk) 🏆
- **RMSE:** 96,165 TL
- **MAE:** 60,417 TL

### Model Performans Karşılaştırması

| Model | R² Skoru | RMSE (TL) | MAE (TL) | Durum |
|-------|----------|-----------|----------|-------|
| **XGBoost (Log)** | **0.9588** | **96,165** | **60,417** | 🥇 En İyi |
| XGBoost | 0.9529 | 102,807 | 61,693 | 🥈 İkinci |
| Random Forest (Log) | 0.9497 | 106,214 | 65,276 | 🥉 Üçüncü |
| Random Forest | 0.9367 | 119,163 | 66,483 | Dördüncü |
| Linear Regression (Log) | 0.8105 | 206,265 | 117,777 | Baseline |

## Açıklanabilir AI (XAI) Analizi

### SHAP Özellik Önem Sıralaması

| Sıra | Özellik | Ortalama SHAP Değeri |
|------|---------|---------------------|
| 1 | **Marka** | 28,109.73 |
| 2 | **Yıllık Ortalama KM** | 17,067.90 |
| 3 | **Model Yılı** | 3,331.77 |
| 4 | **Motor Hacmi** | 3,039.98 |
| 5 | **KM per Motor Hacmi** | 3,022.69 |
| 6 | **Araç Yaşı** | 2,695.47 |
| 7 | **Kilometre** | 592.87 |
| 8 | **Model** | 509.79 |
| 9 | **Vites Türü** | 447.58 |
| 10 | **Yakıt Türü** | 413.34 |

### Temel Bulgular

1. **En Önemli Faktör:** Marka, fiyat tahmininde en büyük etkiye sahip
2. **İkinci En Önemli:** Yıllık ortalama kilometre (araç kullanım yoğunluğu)
3. **Model Yılı:** Araç yaşından daha önemli
4. **Motor Hacmi:** Fiyat üzerinde pozitif etki
5. **Türetilmiş Özellikler:** Model performansını artırdı

## LIME Analizi Örneği

**Seçilen Örnek Araç:**
- Gerçek Fiyat: 214,608 TL
- Tahmin Edilen Fiyat: 222,083 TL
- Hata: 7,475 TL (%3.5)

## Sonuçlar ve Öneriler

### Akademik Katkılar
1. Türkiye pazarına özgü ilk açıklanabilir araç fiyat tahmin modeli
2. SHAP ve LIME tekniklerinin kombinasyonu ile kapsamlı açıklanabilirlik
3. Feature engineering ile model performansının artırılması

### Pratik Uygulamalar
1. **Alıcılar için:** Şeffaf fiyat analizi ve pazarlık rehberi
2. **Satıcılar için:** Piyasa değeri tahmini
3. **Sigorta şirketleri için:** Hasar değeri hesaplama
4. **Finans kurumları için:** Kredi değerlendirme

### Model Avantajları
- Yüksek doğruluk (%84.18 R²)
- Şeffaf karar verme süreci
- Türkiye pazarına özgü özellikler
- Gerçek zamanlı tahmin imkanı

### Gelecek Çalışmalar
1. Daha büyük veri seti ile model geliştirme
2. Gerçek zamanlı veri entegrasyonu
3. Web uygulaması geliştirme

## Teknik Detaylar

### Kullanılan Teknolojiler
- **Python 3.10**
- **Pandas, NumPy** - Veri işleme
- **Scikit-learn** - Makine öğrenmesi
- **XGBoost** - Gradient boosting
- **SHAP** - Model açıklanabilirliği
- **LIME** - Lokal açıklamalar
- **Matplotlib, Seaborn** - Görselleştirme

### Veri Önişleme
- Label encoding (kategorik değişkenler)
- Feature engineering (yeni özellikler)
- Train-test split (%80-%20)
- StandardScaler (Linear Regression için)

### Model Eğitimi
- Cross-validation ile performans değerlendirme
- Hiperparametre optimizasyonu
- Ensemble yöntemleri
- Açıklanabilirlik analizi

---

**Proje Tarihi:** Ekim 2025  
**Durum:** Tamamlandı  
**Sonraki Adım:** Web uygulaması geliştirme
