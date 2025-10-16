# Kaggle Veri Seti İndirme Talimatları

Bu dokümanda, Kaggle'dan Türkiye araç fiyat veri setini nasıl indireceğinizi açıklayacağız.

## Veri Seti Bilgileri

- **Kaggle Linki:** https://www.kaggle.com/datasets/smailburakarkan/car-price-dataset-turkish-lira
- **Veri Seti Adı:** Car Price Dataset Turkish Lira
- **Boyut:** Yaklaşık 1,000+ araç verisi
- **Dil:** Türkçe (Türk Lirası)

## Yöntem 1: Manuel İndirme (Önerilen)

### Adım 1: Kaggle'a Giriş
1. https://www.kaggle.com adresine gidin
2. Hesabınızla giriş yapın (ücretsiz hesap oluşturabilirsiniz)

### Adım 2: Veri Setini Bulun
1. Arama çubuğuna "car price dataset turkish lira" yazın
2. İlk sonuca tıklayın: "Car Price Dataset Turkish Lira"

### Adım 3: Veri Setini İndirin
1. Veri seti sayfasında "Download" butonuna tıklayın
2. ZIP dosyası indirilecek
3. ZIP dosyasını açın

### Adım 4: Veriyi Projeye Yerleştirin
1. ZIP dosyasından çıkan CSV dosyasını kopyalayın
2. Projenizin `data/` klasörüne yerleştirin
3. Dosya adını `car-price-dataset-turkish-lira.csv` olarak değiştirin

### Adım 5: Veriyi İşleyin
```bash
cd src/data_collection
python manuel_veri_indirme.py
```

## Yöntem 2: Kaggle API (Gelişmiş)

### Adım 1: Kaggle API Token
1. Kaggle hesabınızda "Account" sekmesine gidin
2. "API" bölümünde "Create New API Token" tıklayın
3. `kaggle.json` dosyası indirilecek

### Adım 2: Token'ı Yerleştirin
1. Windows'ta: `C:\Users\[KULLANICI_ADI]\.kaggle\` klasörü oluşturun
2. `kaggle.json` dosyasını bu klasöre kopyalayın

### Adım 3: API ile İndirin
```bash
cd src/data_collection
python kaggle_veri_indirme.py
```

## Veri Seti İçeriği

Kaggle veri seti şu sütunları içerir:

| Sütun | Açıklama | Örnek |
|-------|----------|-------|
| Brand | Araç markası | Toyota, BMW, Mercedes |
| Model | Araç modeli | Corolla, 3 Series, C-Class |
| Year | Model yılı | 2020, 2018, 2015 |
| KM | Kilometre | 50000, 120000, 200000 |
| Engine | Motor hacmi | 1.6, 2.0, 3.0 |
| Fuel | Yakıt türü | Benzin, Dizel, LPG |
| Transmission | Vites türü | Manuel, Otomatik |
| Body Type | Gövde tipi | Sedan, Hatchback, SUV |
| Price | Fiyat (TL) | 150000, 250000, 400000 |

## Beklenen Sonuçlar

Gerçek veri seti ile çalıştığınızda:

- **Daha fazla veri:** 1,000+ gerçek araç verisi
- **Daha yüksek doğruluk:** Gerçek piyasa verileri
- **Daha güvenilir sonuçlar:** Türkiye pazarına özgü
- **Daha detaylı analiz:** Gerçek marka ve model çeşitliliği

## Sorun Giderme

### Veri Yüklenmiyor
- CSV dosyasının `data/` klasöründe olduğundan emin olun
- Dosya adının doğru olduğunu kontrol edin
- Dosyanın bozuk olmadığını kontrol edin

### Encoding Hatası
- CSV dosyasını UTF-8 encoding ile kaydedin
- Excel ile açıp "UTF-8" olarak kaydedin

### Sütun İsimleri Uyuşmuyor
- `manuel_veri_indirme.py` scripti otomatik olarak sütun isimlerini düzenler
- Script çalıştırıldıktan sonra `kaggle_arac_verisi_processed.csv` oluşur

## Sonraki Adımlar

Veri seti başarıyla indirildikten sonra:

1. `python main.py` ile projeyi çalıştırın
2. Gerçek veri ile model performanslarını karşılaştırın
3. SHAP ve LIME analizlerini inceleyin
4. Sonuçları raporunuza ekleyin

---

**Not:** Bu veri seti Türkiye ikinci el araç piyasasına özgü gerçek verilerdir ve projenizin akademik değerini önemli ölçüde artıracaktır.
