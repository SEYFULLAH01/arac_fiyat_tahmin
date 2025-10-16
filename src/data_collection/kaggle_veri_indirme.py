"""
Kaggle'dan Türkiye Araç Fiyat Veri Setini İndirme Scripti
"""

import os
import subprocess
import pandas as pd
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kaggle_veri_indir():
    """Kaggle'dan veri setini indirir"""
    
    # Kaggle API token kontrolü
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
        logger.error("Kaggle API token bulunamadı!")
        logger.info("Lütfen https://www.kaggle.com/account adresinden API token'ınızı indirin")
        logger.info("Token'ı ~/.kaggle/kaggle.json dosyasına yerleştirin")
        return False
    
    try:
        # Kaggle'dan veri setini indir
        logger.info("Kaggle'dan veri seti indiriliyor...")
        
        # Veri setini indir
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'smailburakarkan/car-price-dataset-turkish-lira',
            '-p', '../../data/',
            '--unzip'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Veri seti başarıyla indirildi!")
            return True
        else:
            logger.error(f"Veri indirme hatası: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Kaggle API hatası: {e}")
        return False

def veri_kontrol_et():
    """İndirilen veriyi kontrol eder"""
    
    data_dir = '../../data/'
    
    # CSV dosyalarını bul
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning("CSV dosyası bulunamadı!")
        return None
    
    # İlk CSV dosyasını yükle
    csv_file = csv_files[0]
    file_path = os.path.join(data_dir, csv_file)
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Veri seti yüklendi: {csv_file}")
        logger.info(f"Boyut: {df.shape}")
        logger.info(f"Sütunlar: {list(df.columns)}")
        
        # İlk birkaç satırı göster
        logger.info("İlk 5 satır:")
        print(df.head())
        
        return df, file_path
        
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None

def veri_analiz_et(df):
    """Veri setini analiz eder"""
    
    logger.info("\n=== VERİ SETİ ANALİZİ ===")
    
    # Temel bilgiler
    logger.info(f"Toplam kayıt sayısı: {len(df):,}")
    logger.info(f"Toplam sütun sayısı: {df.shape[1]}")
    
    # Eksik değerler
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        logger.info("\nEksik değerler:")
        print(missing_data[missing_data > 0])
    else:
        logger.info("Eksik değer bulunamadı!")
    
    # Sayısal değişkenler
    numeric_cols = df.select_dtypes(include=['number']).columns
    logger.info(f"\nSayısal değişkenler: {list(numeric_cols)}")
    
    # Kategorik değişkenler
    categorical_cols = df.select_dtypes(include=['object']).columns
    logger.info(f"Kategorik değişkenler: {list(categorical_cols)}")
    
    # Hedef değişken analizi (price sütunu varsa)
    if 'price' in df.columns:
        logger.info(f"\nFiyat istatistikleri:")
        logger.info(f"Min: {df['price'].min():,.0f} TL")
        logger.info(f"Max: {df['price'].max():,.0f} TL")
        logger.info(f"Ortalama: {df['price'].mean():,.0f} TL")
        logger.info(f"Medyan: {df['price'].median():,.0f} TL")

def main():
    """Ana fonksiyon"""
    
    print("Kaggle Türkiye Araç Fiyat Veri Seti İndirme")
    print("=" * 50)
    
    # 1. Kaggle'dan veri indir
    if not kaggle_veri_indir():
        print("\nKaggle'dan veri indirilemedi!")
        print("Manuel olarak veri setini indirip data/ klasörüne yerleştirebilirsiniz.")
        return
    
    # 2. Veriyi kontrol et
    result = veri_kontrol_et()
    if result is None:
        print("Veri kontrol edilemedi!")
        return
    
    df, file_path = result
    
    # 3. Veri analizi
    veri_analiz_et(df)
    
    print(f"\nVeri seti başarıyla hazırlandı: {file_path}")
    print("Artık main.py ile gerçek veri seti üzerinde çalışabilirsiniz!")

if __name__ == "__main__":
    main()
