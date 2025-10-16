"""
Manuel olarak Kaggle veri setini indirme scripti
Bu script, kullanıcının manuel olarak indirdiği veri dosyasını işler
"""

import pandas as pd
import os
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def veri_dosyasi_ara():
    """Data klasöründe veri dosyasını arar"""
    
    data_dir = '../../data/'
    
    # Olası dosya isimleri
    possible_files = [
        'car-price-dataset-turkish-lira.csv',
        'car_price_dataset.csv',
        'turkish_car_prices.csv',
        'arac_fiyat_verisi.csv'
    ]
    
    for filename in possible_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            logger.info(f"Veri dosyası bulundu: {filename}")
            return file_path
    
    # Tüm CSV dosyalarını listele
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if csv_files:
        logger.info(f"Bulunan CSV dosyaları: {csv_files}")
        return os.path.join(data_dir, csv_files[0])
    
    return None

def veri_yukle_ve_analiz_et():
    """Veri dosyasını yükler ve analiz eder"""
    
    file_path = veri_dosyasi_ara()
    
    if not file_path:
        logger.error("Veri dosyası bulunamadı!")
        logger.info("Lütfen veri dosyasını data/ klasörüne yerleştirin")
        return None
    
    try:
        # Veriyi yükle
        df = pd.read_csv(file_path)
        logger.info(f"Veri seti yüklendi: {file_path}")
        logger.info(f"Boyut: {df.shape}")
        
        # Sütun isimlerini göster
        logger.info(f"Sütunlar: {list(df.columns)}")
        
        # İlk birkaç satırı göster
        logger.info("\nİlk 5 satır:")
        print(df.head())
        
        # Veri analizi
        veri_analiz_et(df)
        
        # Veriyi standart formata çevir
        df_processed = veri_isle(df)
        
        # İşlenmiş veriyi kaydet
        output_path = '../../data/kaggle_arac_verisi_processed.csv'
        df_processed.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"İşlenmiş veri kaydedildi: {output_path}")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None

def veri_analiz_et(df):
    """Veri setini detaylı analiz eder"""
    
    logger.info("\n=== VERİ SETİ DETAYLI ANALİZİ ===")
    
    # Temel bilgiler
    logger.info(f"Toplam kayıt sayısı: {len(df):,}")
    logger.info(f"Toplam sütun sayısı: {df.shape[1]}")
    
    # Eksik değerler
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        logger.info("\nEksik değerler:")
        missing_percent = (missing_data / len(df)) * 100
        for col, count in missing_data[missing_data > 0].items():
            logger.info(f"  {col}: {count} ({missing_percent[col]:.1f}%)")
    else:
        logger.info("Eksik değer bulunamadı!")
    
    # Sayısal değişkenler
    numeric_cols = df.select_dtypes(include=['number']).columns
    logger.info(f"\nSayısal değişkenler ({len(numeric_cols)} adet):")
    for col in numeric_cols:
        logger.info(f"  {col}: {df[col].dtype}")
    
    # Kategorik değişkenler
    categorical_cols = df.select_dtypes(include=['object']).columns
    logger.info(f"\nKategorik değişkenler ({len(categorical_cols)} adet):")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        logger.info(f"  {col}: {unique_count} benzersiz değer")
        if unique_count <= 10:
            logger.info(f"    Değerler: {list(df[col].unique())}")
    
    # Hedef değişken analizi
    price_columns = [col for col in df.columns if 'price' in col.lower() or 'fiyat' in col.lower()]
    if price_columns:
        price_col = price_columns[0]
        logger.info(f"\nFiyat değişkeni ({price_col}) analizi:")
        logger.info(f"  Min: {df[price_col].min():,.0f}")
        logger.info(f"  Max: {df[price_col].max():,.0f}")
        logger.info(f"  Ortalama: {df[price_col].mean():,.0f}")
        logger.info(f"  Medyan: {df[price_col].median():,.0f}")
        logger.info(f"  Standart sapma: {df[price_col].std():,.0f}")

def veri_isle(df):
    """Veriyi standart formata çevirir"""
    
    logger.info("\nVeri işleme başlatılıyor...")
    
    df_processed = df.copy()
    
    # Sütun isimlerini küçük harfe çevir ve Türkçe karakterleri düzelt
    column_mapping = {}
    for col in df_processed.columns:
        new_col = col.lower()
        new_col = new_col.replace('ı', 'i').replace('ş', 's').replace('ğ', 'g')
        new_col = new_col.replace('ü', 'u').replace('ö', 'o').replace('ç', 'c')
        column_mapping[col] = new_col
    
    df_processed = df_processed.rename(columns=column_mapping)
    
    # Fiyat sütununu bul ve 'fiyat' olarak yeniden adlandır
    price_columns = [col for col in df_processed.columns if 'price' in col.lower() or 'fiyat' in col.lower()]
    if price_columns:
        df_processed = df_processed.rename(columns={price_columns[0]: 'fiyat'})
    
    # Marka sütununu bul
    brand_columns = [col for col in df_processed.columns if 'brand' in col.lower() or 'marka' in col.lower() or 'make' in col.lower()]
    if brand_columns:
        df_processed = df_processed.rename(columns={brand_columns[0]: 'marka'})
    
    # Model sütununu bul
    model_columns = [col for col in df_processed.columns if 'model' in col.lower()]
    if model_columns:
        df_processed = df_processed.rename(columns={model_columns[0]: 'model'})
    
    # Yıl sütununu bul
    year_columns = [col for col in df_processed.columns if 'year' in col.lower() or 'yil' in col.lower()]
    if year_columns:
        df_processed = df_processed.rename(columns={year_columns[0]: 'model_yili'})
    
    # Kilometre sütununu bul
    km_columns = [col for col in df_processed.columns if 'km' in col.lower() or 'mileage' in col.lower() or 'kilometre' in col.lower()]
    if km_columns:
        df_processed = df_processed.rename(columns={km_columns[0]: 'kilometre'})
    
    # Motor hacmi sütununu bul
    engine_columns = [col for col in df_processed.columns if 'engine' in col.lower() or 'motor' in col.lower() or 'cc' in col.lower()]
    if engine_columns:
        df_processed = df_processed.rename(columns={engine_columns[0]: 'motor_hacmi'})
    
    # Yakıt türü sütununu bul
    fuel_columns = [col for col in df_processed.columns if 'fuel' in col.lower() or 'yakit' in col.lower()]
    if fuel_columns:
        df_processed = df_processed.rename(columns={fuel_columns[0]: 'yakit_turu'})
    
    # Vites türü sütununu bul
    transmission_columns = [col for col in df_processed.columns if 'transmission' in col.lower() or 'vites' in col.lower() or 'gear' in col.lower()]
    if transmission_columns:
        df_processed = df_processed.rename(columns={transmission_columns[0]: 'vites_turu'})
    
    logger.info(f"İşlenmiş sütunlar: {list(df_processed.columns)}")
    
    return df_processed

def main():
    """Ana fonksiyon"""
    
    print("Manuel Kaggle Veri İşleme")
    print("=" * 40)
    print("Bu script, data/ klasörüne yerleştirdiğiniz veri dosyasını işler.")
    print("Lütfen önce Kaggle'dan veri setini indirip data/ klasörüne yerleştirin.")
    print()
    
    # Veri dosyasını işle
    df = veri_yukle_ve_analiz_et()
    
    if df is not None:
        print(f"\nVeri seti basariyla islendi!")
        print(f"Toplam {len(df):,} kayit, {df.shape[1]} sutun")
        print(f"Islenmis veri: data/kaggle_arac_verisi_processed.csv")
        print("\nArtik main.py ile gercek veri seti uzerinde calisabilirsiniz!")
    else:
        print("\nVeri isleme basarisiz!")
        print("Lutfen veri dosyasini kontrol edin.")

if __name__ == "__main__":
    main()
