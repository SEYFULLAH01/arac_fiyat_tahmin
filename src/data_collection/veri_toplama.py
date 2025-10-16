"""
Türkiye İkinci El Araç Veri Toplama Scripti
Bu script, çeşitli kaynaklardan ikinci el araç verilerini toplar.
"""

import pandas as pd
import numpy as np
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArabaVeriToplayici:
    def __init__(self):
        """Veri toplayıcı sınıfını başlatır."""
        self.veri_listesi = []
    
    
    def ornek_veri_olustur(self, adet=1000):
        """Örnek veri seti oluşturur."""
        logger.info(f"{adet} adet örnek veri oluşturuluyor...")
        
        np.random.seed(42)
        
        markalar = ['Toyota', 'Volkswagen', 'Ford', 'Renault', 'Opel', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Fiat']
        modeller = ['Corolla', 'Golf', 'Focus', 'Megane', 'Astra', '3 Series', 'C-Class', 'A4', 'i20', 'Punto']
        yakit_turleri = ['Benzin', 'Dizel', 'LPG', 'Hibrit']
        vites_turleri = ['Manuel', 'Otomatik', 'Yarı Otomatik']
        govde_tipleri = ['Sedan', 'Hatchback', 'SUV', 'Station Wagon', 'Coupe']
        
        # Daha gerçekçi fiyat dağılımı için
        def fiyat_hesapla(marka, model_yili, kilometre, motor_hacmi):
            base_fiyat = 100000
            
            # Marka etkisi
            marka_etki = {
                'BMW': 1.5, 'Mercedes': 1.5, 'Audi': 1.4,
                'Toyota': 1.2, 'Volkswagen': 1.1, 'Ford': 1.0,
                'Renault': 0.9, 'Opel': 0.9, 'Hyundai': 0.8, 'Fiat': 0.7
            }
            
            # Yaş etkisi (yeni araçlar daha pahalı)
            yas_etki = 1 + (2024 - model_yili) * 0.05
            
            # Kilometre etkisi (az kilometre daha pahalı)
            km_etki = 1 - (kilometre / 300000) * 0.3
            
            # Motor hacmi etkisi
            motor_etki = 1 + (motor_hacmi - 1.6) * 0.1
            
            fiyat = base_fiyat * marka_etki.get(marka, 1.0) * yas_etki * km_etki * motor_etki
            
            # Rastgelelik ekle
            fiyat *= np.random.uniform(0.8, 1.2)
            
            return max(50000, min(800000, int(fiyat)))
        
        for i in range(adet):
            marka = np.random.choice(markalar)
            model = np.random.choice(modeller)
            model_yili = np.random.randint(2010, 2024)
            kilometre = np.random.randint(5000, 300000)
            motor_hacmi = np.random.choice([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0])
            
            arac_verisi = {
                'marka': marka,
                'model': model,
                'model_yili': model_yili,
                'kilometre': kilometre,
                'motor_hacmi': motor_hacmi,
                'yakit_turu': np.random.choice(yakit_turleri),
                'vites_turu': np.random.choice(vites_turleri),
                'govde_tipi': np.random.choice(govde_tipleri),
                'fiyat': fiyat_hesapla(marka, model_yili, kilometre, motor_hacmi),
                'kaynak': 'ornek_veri'
            }
            
            self.veri_listesi.append(arac_verisi)
        
        logger.info(f"{len(self.veri_listesi)} adet örnek veri oluşturuldu")
    
    def veriyi_kaydet(self, dosya_adi='ikinci_el_arac_verisi.csv'):
        """Toplanan veriyi CSV dosyasına kaydeder."""
        if not self.veri_listesi:
            logger.warning("Kaydedilecek veri bulunamadı!")
            return
        
        df = pd.DataFrame(self.veri_listesi)
        df.to_csv(f'../../data/{dosya_adi}', index=False, encoding='utf-8')
        logger.info(f"Veri {dosya_adi} dosyasına kaydedildi. Toplam {len(df)} kayıt.")
        
        return df
    
    def temizle(self):
        """Temizlik işlemleri."""
        logger.info("Temizlik işlemleri tamamlandı")

def main():
    """Ana fonksiyon"""
    veri_toplayici = ArabaVeriToplayici()
    
    try:
        # Örnek veri oluştur (gerçek veri toplama yerine)
        veri_toplayici.ornek_veri_olustur(adet=1000)
        
        # Veriyi kaydet
        df = veri_toplayici.veriyi_kaydet('ornek_ikinci_el_arac_verisi.csv')
        
        print("\n=== TOPLANAN VERİ ÖZETİ ===")
        print(f"Toplam kayıt sayısı: {len(df)}")
        print(f"Sütunlar: {list(df.columns)}")
        print("\nİlk 5 kayıt:")
        print(df.head())
        
    except Exception as e:
        logger.error(f"Ana fonksiyon hatası: {e}")
    
    finally:
        veri_toplayici.temizle()

if __name__ == "__main__":
    main()

