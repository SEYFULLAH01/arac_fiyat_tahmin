"""
Çoklu Test - Farklı Araçlarla Tahmin
"""

import pandas as pd
import numpy as np
import joblib

# Modeli yükle
model_data = joblib.load('models/hizli_arac_modeli.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']

print("Model yuklendi!")

# Test araçları
test_araclar = [
    {
        'isim': 'BMW 3 Series',
        'marka': 'BMW',
        'seri': '3 Series',
        'model': '320i',
        'model_yili': 2020,
        'kilometre': 50000,
        'motor_hacmi': 2.0,
        'yakit_turu': 'Benzin',
        'vites_turu': 'Otomatik',
        'kasatipi': 'Sedan',
        'motorgucu': 150
    },
    {
        'isim': 'Toyota Corolla',
        'marka': 'Toyota',
        'seri': 'Corolla',
        'model': '1.6 Comfort',
        'model_yili': 2018,
        'kilometre': 80000,
        'motor_hacmi': 1.6,
        'yakit_turu': 'Benzin',
        'vites_turu': 'Manuel',
        'kasatipi': 'Sedan',
        'motorgucu': 110
    },
    {
        'isim': 'Volkswagen Golf',
        'marka': 'Volkswagen',
        'seri': 'Golf',
        'model': '1.6 TDI',
        'model_yili': 2019,
        'kilometre': 60000,
        'motor_hacmi': 1.6,
        'yakit_turu': 'Dizel',
        'vites_turu': 'Manuel',
        'kasatipi': 'Hatchback',
        'motorgucu': 105
    },
    {
        'isim': 'Mercedes C-Class',
        'marka': 'Mercedes',
        'seri': 'C-Class',
        'model': 'C200',
        'model_yili': 2021,
        'kilometre': 30000,
        'motor_hacmi': 1.5,
        'yakit_turu': 'Benzin',
        'vites_turu': 'Otomatik',
        'kasatipi': 'Sedan',
        'motorgucu': 184
    },
    {
        'isim': 'Ford Focus',
        'marka': 'Ford',
        'seri': 'Focus',
        'model': '1.5 TDCi',
        'model_yili': 2017,
        'kilometre': 120000,
        'motor_hacmi': 1.5,
        'yakit_turu': 'Dizel',
        'vites_turu': 'Manuel',
        'kasatipi': 'Hatchback',
        'motorgucu': 120
    }
]

def tahmin_yap(test_arac):
    """Tek araç için tahmin yap"""
    
    # DataFrame oluştur
    df = pd.DataFrame([test_arac])
    
    # Feature engineering
    df['arac_yasi'] = 2025 - df['model_yili']
    df['yillik_ortalama_km'] = df['kilometre'] / (df['arac_yasi'] + 1)
    df['km_per_motor_hacmi'] = df['kilometre'] / (df['motor_hacmi'] + 0.1)
    df['guc_per_hacim'] = df['motorgucu'] / (df['motor_hacmi'] + 0.1)
    
    # Kategorik değişkenleri encode et
    categorical_features = ['marka', 'seri', 'model', 'vites_turu', 'yakit_turu', 'kasatipi']
    
    for feature in categorical_features:
        if feature in label_encoders:
            try:
                df[feature + '_encoded'] = label_encoders[feature].transform(df[feature].astype(str))
            except:
                df[feature + '_encoded'] = 0
    
    # Özellik seçimi
    feature_names = model_data['feature_names']
    available_features = [col for col in feature_names if col in df.columns]
    
    # Tahmin yap
    X = df[available_features]
    tahmin = model.predict(X)[0]
    
    return tahmin

print("\n" + "="*80)
print("ÇOKLU ARAÇ FİYAT TAHMİNİ")
print("="*80)

for i, arac in enumerate(test_araclar, 1):
    tahmin = tahmin_yap(arac)
    
    print(f"\n{i}. {arac['isim']}")
    print(f"   Model Yılı: {arac['model_yili']} ({2025-arac['model_yili']} yaş)")
    print(f"   Kilometre: {arac['kilometre']:,} km")
    print(f"   Motor: {arac['motor_hacmi']}L {arac['motorgucu']}HP")
    print(f"   Yakıt: {arac['yakit_turu']} | Vites: {arac['vites_turu']}")
    print(f"   TAHMİN: {tahmin:,.0f} TL")
    
    # Fiyat kategorisi
    if tahmin < 300000:
        kategori = "Ekonomik"
    elif tahmin < 600000:
        kategori = "Orta Segment"
    elif tahmin < 1000000:
        kategori = "Üst Segment"
    else:
        kategori = "Lüks"
    
    print(f"   Kategori: {kategori}")

print(f"\n" + "="*80)
print("TÜM TAHMİNLER TAMAMLANDI!")
print("="*80)
