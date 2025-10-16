"""
Test Tahmin - Hızlı Test
"""

import pandas as pd
import numpy as np
import joblib

# Modeli yükle
model_data = joblib.load('models/hizli_arac_modeli.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']

print("Model yuklendi!")

# Test verisi oluştur
test_data = {
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
}

print("\nTest verisi:")
print(f"Marka: {test_data['marka']}")
print(f"Model Yili: {test_data['model_yili']}")
print(f"Kilometre: {test_data['kilometre']:,} km")
print(f"Motor Hacmi: {test_data['motor_hacmi']}L")
print(f"Motor Gucu: {test_data['motorgucu']} HP")

# DataFrame oluştur
df = pd.DataFrame([test_data])

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
            print(f"{feature}: {df[feature].iloc[0]} -> {df[feature + '_encoded'].iloc[0]}")
        except:
            df[feature + '_encoded'] = 0
            print(f"{feature}: Bilinmiyor -> 0")

# Özellik seçimi
feature_names = model_data['feature_names']
available_features = [col for col in feature_names if col in df.columns]

print(f"\nKullanilan ozellikler ({len(available_features)} adet):")
for feature in available_features:
    print(f"  - {feature}")

# Tahmin yap
X = df[available_features]
tahmin = model.predict(X)[0]

print(f"\n=== SONUÇ ===")
print(f"TAHMİN EDİLEN FİYAT: {tahmin:,.0f} TL")

# Fiyat aralığı
alt_fiyat = tahmin * 0.9
ust_fiyat = tahmin * 1.1
print(f"FİYAT ARALIĞI: {alt_fiyat:,.0f} - {ust_fiyat:,.0f} TL")

print("\nTest tamamlandi!")
