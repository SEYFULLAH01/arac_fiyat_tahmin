# Gerçek vs Örnek Veri Karşılaştırması

## Özet

Bu dokümanda, örnek veri seti ile Kaggle'dan indirilen gerçek veri seti arasındaki farkları ve model performanslarını karşılaştırıyoruz.

## Veri Seti Karşılaştırması

### Örnek Veri Seti
- **Kaynak:** Programatik olarak oluşturulan
- **Boyut:** 1,000 kayıt
- **Veri Kalitesi:** Temiz, eksik değer yok
- **Gerçeklik:** Simüle edilmiş

### Kaggle Veri Seti (İşlenmiş)
- **Kaynak:** Kaggle - Türkiye araç piyasası
- **Boyut:** 986 kayıt (aykırı değerler temizlendikten sonra)
- **Veri Kalitesi:** Gerçek piyasa verisi
- **Gerçeklik:** %100 gerçek

## Model Performans Karşılaştırması

### Örnek Veri ile Sonuçlar

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 18,343 | 14,735 | **0.8418** |
| XGBoost | 19,380 | 15,439 | 0.8234 |
| Linear Regression | 37,761 | 31,475 | 0.3295 |

### Kaggle Veri ile Sonuçlar

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 18,681 | 14,942 | **0.8149** |
| XGBoost | 19,552 | 15,469 | 0.7973 |
| Linear Regression | 35,658 | 28,250 | 0.3257 |

## Detaylı Analiz

### Performans Karşılaştırması

#### Random Forest Model
- **R² Farkı:** -0.0269 (örnek veri daha yüksek)
- **RMSE Farkı:** +338 TL (örnek veri daha düşük)
- **MAE Farkı:** +207 TL (örnek veri daha düşük)

#### XGBoost Model
- **R² Farkı:** -0.0261 (örnek veri daha yüksek)
- **RMSE Farkı:** +172 TL (örnek veri daha düşük)
- **MAE Farkı:** +30 TL (örnek veri daha düşük)

#### Linear Regression Model
- **R² Farkı:** -0.0038 (örnek veri daha yüksek)
- **RMSE Farkı:** -2,103 TL (Kaggle veri daha düşük)
- **MAE Farkı:** -3,225 TL (Kaggle veri daha düşük)

### SHAP Özellik Önem Karşılaştırması

#### Örnek Veri - En Önemli 5 Özellik
1. **Marka:** 28,109.73
2. **Yıllık Ortalama KM:** 17,067.90
3. **Model Yılı:** 3,331.77
4. **Motor Hacmi:** 3,039.98
5. **KM per Motor Hacmi:** 3,022.69

#### Kaggle Veri - En Önemli 5 Özellik
1. **Marka:** 25,574.80
2. **Yıllık Ortalama KM:** 15,428.13
3. **Motor Hacmi:** 3,243.47
4. **Araç Yaşı:** 2,812.72
5. **Model Yılı:** 2,626.40

## Önemli Bulgular

### 1. Model Performansı
- **Örnek veri** model eğitimi için daha "kolay" görünüyor
- **Gerçek veri** daha karmaşık ilişkiler içeriyor
- Performans farkları %2-3 arasında, kabul edilebilir seviyede

### 2. Özellik Önemleri
- **Marka** her iki veri setinde de en önemli faktör
- **Yıllık ortalama kilometre** ikinci sırada
- Gerçek veride **motor hacmi** daha önemli
- Örnek veride **türetilmiş özellikler** daha etkili

### 3. Veri Kalitesi
- **Kaggle verisi** daha gerçekçi fiyat dağılımı
- **Aykırı değerler** gerçek veride daha fazla
- **Eksik değerler** gerçek veride mevcut

## Sonuçlar ve Öneriler

### Akademik Değer
1. **Gerçek veri** mezuniyet projesi için çok daha değerli
2. **Türkiye pazarına özgü** bulgular
3. **Gerçek dünya uygulamaları** için uygun

### Pratik Uygulamalar
1. **Kaggle verisi** gerçek piyasa analizi için ideal
2. **Örnek veri** prototip geliştirme için uygun
3. **Karşılaştırma** model güvenilirliğini artırıyor

### Model Geliştirme
1. **Feature engineering** her iki veri setinde de etkili
2. **Random Forest** en tutarlı performans
3. **XGBoost** gerçek veride biraz daha düşük performans

## Sonraki Adımlar

### Kısa Vadeli
1. Kaggle veri setini tam olarak indirin
2. Daha büyük veri seti ile çalışın
3. Hiperparametre optimizasyonu yapın

### Uzun Vadeli
1. Gerçek zamanlı veri entegrasyonu
2. Web scraping ile veri toplama
3. Model güncelleme mekanizması

---

**Sonuç:** Kaggle veri seti, mezuniyet projeniz için örnek veriden çok daha değerli ve akademik açıdan daha güvenilir sonuçlar sunmaktadır. Gerçek piyasa verileri ile çalışmak, projenizin kalitesini önemli ölçüde artıracaktır.
