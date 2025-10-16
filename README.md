# Türkiye İkinci El Araç Fiyat Tahmini - Açıklanabilir Makine Öğrenmesi

## Proje Hakkında
Bu proje, Türkiye ikinci el araç piyasasına özgü verileri kullanarak yüksek doğrulukla fiyat tahmini yapabilen bir makine öğrenmesi modeli geliştirmeyi amaçlar. Modelin tahminlerini SHAP ve LIME gibi modern Açıklanabilir Yapay Zeka (XAI) teknikleri kullanarak yorumlanabilir hale getirir.

## Hazırlayanlar
- Seyfullah Adıgüzel
- Batuhan Orkun İnce

## Danışman
- Sinan Keskin

## Kurulum
```bash
pip install -r requirements.txt
```

## Proje Yapısı
```
├── data/                   # Veri dosyaları
├── notebooks/              # Jupyter notebook'lar
├── src/                    # Kaynak kodlar
│   ├── data_collection/    # Veri toplama
│   ├── preprocessing/      # Veri önişleme
│   ├── modeling/          # Model geliştirme
│   ├── xai/              # Açıklanabilirlik analizi
│   └── visualization/     # Görselleştirme
├── results/               # Sonuçlar ve raporlar
└── README.md
```

## Kullanım
1. Veri setini `data/` klasörüne yerleştirin
2. `notebooks/` klasöründeki notebook'ları sırayla çalıştırın
3. Sonuçları `results/` klasöründe inceleyin

