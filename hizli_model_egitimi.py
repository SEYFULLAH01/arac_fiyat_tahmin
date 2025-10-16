"""
Hızlı Model Eğitimi - Gerçek Tahmin İçin
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def hizli_model_egit():
    """Hızlı model eğitimi"""
    print("Hizli model egitimi baslatiliyor...")
    
    # Veriyi yükle
    df = pd.read_csv('data/kaggle_arac_verisi_processed.csv')
    print(f"Veri yüklendi: {df.shape}")
    
    # Veri temizleme
    df = df[(df['kilometre'] >= 0) & (df['kilometre'] <= 1_000_000)]
    df = df[(df['model_yili'] >= 1980) & (df['model_yili'] <= 2025)]
    df = df[(df['motor_hacmi'] >= 500) & (df['motor_hacmi'] <= 8000)]
    
    # Fiyat filtreleri
    Q1_price = df['fiyat'].quantile(0.01)
    Q99_price = df['fiyat'].quantile(0.99)
    df = df[(df['fiyat'] >= Q1_price) & (df['fiyat'] <= Q99_price)]
    
    print(f"Temizlenmiş veri: {df.shape}")
    
    # Feature engineering
    df['arac_yasi'] = 2025 - df['model_yili']
    df['yillik_ortalama_km'] = df['kilometre'] / (df['arac_yasi'] + 1)
    df['km_per_motor_hacmi'] = df['kilometre'] / (df['motor_hacmi'] + 0.1)
    
    if 'motorgucu' in df.columns:
        df['guc_per_hacim'] = df['motorgucu'] / (df['motor_hacmi'] + 0.1)
    
    # Kategorik değişkenleri encode et
    categorical_features = ['marka', 'seri', 'model', 'vites_turu', 'yakit_turu', 'kasatipi']
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le
    
    # Özellik seçimi
    feature_cols = ['model_yili', 'kilometre', 'motor_hacmi', 'arac_yasi', 
                   'yillik_ortalama_km', 'km_per_motor_hacmi']
    
    if 'motorgucu' in df.columns:
        feature_cols.extend(['motorgucu', 'guc_per_hacim'])
    
    # Kategorik özellikler
    categorical_features_encoded = [col for col in df.columns if col.endswith('_encoded')]
    feature_cols.extend(categorical_features_encoded)
    
    # Sadece mevcut özellikleri al
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['fiyat']
    
    print(f"Özellik sayısı: {len(available_features)}")
    print(f"Örnek sayısı: {len(X)}")
    
    # Model eğitimi
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Modeli kaydet
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_names': available_features,
        'scaler': None
    }
    
    joblib.dump(model_data, 'models/hizli_arac_modeli.pkl')
    
    # Test tahmini
    test_pred = model.predict(X.head(5))
    print(f"\nModel egitildi ve kaydedildi!")
    print(f"Test tahminleri: {test_pred}")
    print(f"Model dosyasi: models/hizli_arac_modeli.pkl")
    
    return model, label_encoders, available_features

if __name__ == "__main__":
    hizli_model_egit()
