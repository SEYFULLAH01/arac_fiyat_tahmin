"""
Gerçek Araç Fiyat Tahmini - Kullanıcı Girişli
Eğitilmiş model ile gerçek tahmin yapma
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

class ArabaFiyatTahminci:
    """Gerçek araç fiyat tahmincisi"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.model_yuklendi = False
        
    def modeli_yukle(self, model_dosya='models/hizli_arac_modeli.pkl'):
        """Eğitilmiş modeli yükle"""
        try:
            if os.path.exists(model_dosya):
                model_data = joblib.load(model_dosya)
                self.model = model_data['model']
                self.label_encoders = model_data['label_encoders']
                self.feature_names = model_data['feature_names']
                self.model_yuklendi = True
                print("Model basariyla yuklendi!")
                return True
            else:
                print("Model dosyasi bulunamadi!")
                return False
        except Exception as e:
            print(f"Model yukleme hatasi: {e}")
            return False
    
    def kullanici_girdisi_al(self):
        """Kullanıcıdan araç bilgilerini al"""
        print("\n🚗 ARAÇ FİYAT TAHMİNİ")
        print("=" * 40)
        
        # Marka seçimi
        print("\n📋 Marka seçin:")
        markalar = ['BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Toyota', 'Ford', 'Renault', 
                   'Opel', 'Hyundai', 'Fiat', 'Peugeot', 'Nissan', 'Honda', 'Mazda', 
                   'Kia', 'Skoda', 'Seat', 'Citroen', 'Dacia', 'Chevrolet']
        
        for i, marka in enumerate(markalar, 1):
            print(f"{i:2d}. {marka}")
        
        while True:
            try:
                marka_secim = int(input("\nMarka numarasını girin (1-20): ")) - 1
                if 0 <= marka_secim < len(markalar):
                    marka = markalar[marka_secim]
                    break
                else:
                    print("❌ Geçersiz seçim! 1-20 arası bir sayı girin.")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Model yılı
        while True:
            try:
                model_yili = int(input(f"\n📅 {marka} aracınızın model yılı (1990-2025): "))
                if 1990 <= model_yili <= 2025:
                    break
                else:
                    print("❌ Geçersiz yıl! 1990-2025 arası girin.")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Kilometre
        while True:
            try:
                kilometre = int(input("🛣️  Kilometre: "))
                if kilometre >= 0:
                    break
                else:
                    print("❌ Kilometre negatif olamaz!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Motor hacmi
        print("\n🔧 Motor hacmi seçin:")
        motor_hacimleri = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0]
        for i, hacim in enumerate(motor_hacimleri, 1):
            print(f"{i:2d}. {hacim}L")
        
        while True:
            try:
                motor_secim = int(input("Motor hacmi numarasını girin: ")) - 1
                if 0 <= motor_secim < len(motor_hacimleri):
                    motor_hacmi = motor_hacimleri[motor_secim]
                    break
                else:
                    print("❌ Geçersiz seçim!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Yakıt türü
        print("\n⛽ Yakıt türü seçin:")
        yakit_turleri = ['Benzin', 'Dizel', 'LPG', 'Hibrit', 'Elektrik']
        for i, yakit in enumerate(yakit_turleri, 1):
            print(f"{i}. {yakit}")
        
        while True:
            try:
                yakit_secim = int(input("Yakıt türü numarasını girin: ")) - 1
                if 0 <= yakit_secim < len(yakit_turleri):
                    yakit_turu = yakit_turleri[yakit_secim]
                    break
                else:
                    print("❌ Geçersiz seçim!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Vites türü
        print("\n⚙️  Vites türü seçin:")
        vites_turleri = ['Manuel', 'Otomatik', 'Yarı Otomatik']
        for i, vites in enumerate(vites_turleri, 1):
            print(f"{i}. {vites}")
        
        while True:
            try:
                vites_secim = int(input("Vites türü numarasını girin: ")) - 1
                if 0 <= vites_secim < len(vites_turleri):
                    vites_turu = vites_turleri[vites_secim]
                    break
                else:
                    print("❌ Geçersiz seçim!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Gövde tipi
        print("\n🚙 Gövde tipi seçin:")
        govde_tipleri = ['Sedan', 'Hatchback', 'SUV', 'Station Wagon', 'Coupe', 'Cabrio']
        for i, govde in enumerate(govde_tipleri, 1):
            print(f"{i}. {govde}")
        
        while True:
            try:
                govde_secim = int(input("Gövde tipi numarasını girin: ")) - 1
                if 0 <= govde_secim < len(govde_tipleri):
                    govde_tipi = govde_tipleri[govde_secim]
                    break
                else:
                    print("❌ Geçersiz seçim!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        # Motor gücü
        while True:
            try:
                motor_gucu = int(input("🔋 Motor gücü (HP): "))
                if motor_gucu > 0:
                    break
                else:
                    print("❌ Motor gücü pozitif olmalı!")
            except ValueError:
                print("❌ Lütfen bir sayı girin!")
        
        return {
            'marka': marka,
            'model_yili': model_yili,
            'kilometre': kilometre,
            'motor_hacmi': motor_hacmi,
            'yakit_turu': yakit_turu,
            'vites_turu': vites_turu,
            'kasatipi': govde_tipi,
            'motorgucu': motor_gucu
        }
    
    def veriyi_hazirla(self, kullanici_girdisi):
        """Kullanıcı girdisini model için hazırla"""
        
        # DataFrame oluştur
        df = pd.DataFrame([kullanici_girdisi])
        
        # Feature engineering
        df['arac_yasi'] = 2025 - df['model_yili']
        df['yillik_ortalama_km'] = df['kilometre'] / (df['arac_yasi'] + 1)
        df['km_per_motor_hacmi'] = df['kilometre'] / (df['motor_hacmi'] + 0.1)
        df['guc_per_hacim'] = df['motorgucu'] / (df['motor_hacmi'] + 0.1)
        
        # Kategorik gruplar
        df['yas_grubu'] = pd.cut(df['arac_yasi'], 
                                bins=[0, 3, 7, 15, 100], 
                                labels=['Yeni', 'Az_Kullanilmis', 'Orta', 'Eski'])
        
        df['km_grubu'] = pd.cut(df['kilometre'], 
                               bins=[0, 50000, 100000, 200000, 1000000], 
                               labels=['Dusuk', 'Orta', 'Yuksek', 'Cok_Yuksek'])
        
        df['motor_grubu'] = pd.cut(df['motor_hacmi'], 
                                  bins=[0, 1200, 1600, 2000, 8000], 
                                  labels=['Kucuk', 'Orta', 'Buyuk', 'Cok_Buyuk'])
        
        # Kategorik değişkenleri encode et
        categorical_features = ['marka', 'vites_turu', 'yakit_turu', 'kasatipi', 
                              'yas_grubu', 'km_grubu', 'motor_grubu']
        
        for feature in categorical_features:
            if feature in df.columns and feature in self.label_encoders:
                try:
                    df[feature + '_encoded'] = self.label_encoders[feature].transform(df[feature].astype(str))
                except:
                    # Bilinmeyen kategori için varsayılan değer
                    df[feature + '_encoded'] = 0
        
        # Özellik seçimi (model eğitiminde kullanılan özellikler)
        feature_cols = []
        
        # Sayısal özellikler
        numeric_features = ['model_yili', 'kilometre', 'motor_hacmi', 'arac_yasi', 
                           'yillik_ortalama_km', 'km_per_motor_hacmi', 'motorgucu', 'guc_per_hacim']
        
        # Kategorik özellikler (encoded)
        categorical_features_encoded = [col for col in df.columns if col.endswith('_encoded')]
        
        feature_cols = numeric_features + categorical_features_encoded
        
        # Sadece mevcut özellikleri al
        available_features = [col for col in feature_cols if col in df.columns]
        
        return df[available_features]
    
    def tahmin_yap(self, kullanici_girdisi):
        """Fiyat tahmini yap"""
        if not self.model_yuklendi:
            print("❌ Model yüklenmedi! Önce modeli yükleyin.")
            return None
        
        try:
            # Veriyi hazırla
            X = self.veriyi_hazirla(kullanici_girdisi)
            
            # Tahmin yap
            tahmin = self.model.predict(X)[0]
            
            # Log transform'dan geri dönüştür (eğer log model kullanıldıysa)
            if hasattr(self.model, 'predict') and 'Log' in str(type(self.model)):
                tahmin = np.expm1(tahmin)
            
            return tahmin
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return None
    
    def sonuclari_goster(self, kullanici_girdisi, tahmin):
        """Sonuçları göster"""
        print("\n" + "="*50)
        print("🎯 TAHMİN SONUÇLARI")
        print("="*50)
        
        print(f"\n🚗 Araç Bilgileri:")
        print(f"   Marka: {kullanici_girdisi['marka']}")
        print(f"   Model Yılı: {kullanici_girdisi['model_yili']}")
        print(f"   Araç Yaşı: {2025 - kullanici_girdisi['model_yili']} yıl")
        print(f"   Kilometre: {kullanici_girdisi['kilometre']:,} km")
        print(f"   Motor Hacmi: {kullanici_girdisi['motor_hacmi']}L")
        print(f"   Motor Gücü: {kullanici_girdisi['motorgucu']} HP")
        print(f"   Yakıt Türü: {kullanici_girdisi['yakit_turu']}")
        print(f"   Vites: {kullanici_girdisi['vites_turu']}")
        print(f"   Gövde: {kullanici_girdisi['kasatipi']}")
        
        if tahmin:
            print(f"\n💰 TAHMİN EDİLEN FİYAT:")
            print(f"   {tahmin:,.0f} TL")
            
            # Fiyat aralığı
            alt_fiyat = tahmin * 0.9
            ust_fiyat = tahmin * 1.1
            print(f"\n📊 FİYAT ARALIĞI:")
            print(f"   {alt_fiyat:,.0f} - {ust_fiyat:,.0f} TL")
            
            # Öneriler
            print(f"\n💡 ÖNERİLER:")
            arac_yasi = 2025 - kullanici_girdisi['model_yili']
            
            if arac_yasi > 10:
                print("   ⚠️  10 yaş üzeri araç - detaylı muayene önerilir")
            
            if kullanici_girdisi['kilometre'] > 200000:
                print("   ⚠️  Yüksek kilometre - bakım geçmişi önemli")
            
            if kullanici_girdisi['motor_hacmi'] < 1.4:
                print("   💡 Düşük motor hacmi - yakıt tasarrufu avantajı")
            
            if kullanici_girdisi['yakit_turu'] == "Elektrik":
                print("   🌱 Elektrikli araç - çevre dostu seçim")
            
            if tahmin > 500000:
                print("   💎 Yüksek değerli araç - sigorta ve güvenlik önemli")
            
        else:
            print("❌ Tahmin yapılamadı!")

def main():
    """Ana fonksiyon"""
    print("🚗 TÜRKIYE ARAÇ FİYAT TAHMİNİ")
    print("Açıklanabilir Makine Öğrenmesi ile")
    print("="*40)
    
    # Tahminci oluştur
    tahminci = ArabaFiyatTahminci()
    
    # Modeli yükle
    if not tahminci.modeli_yukle():
        print("\n❌ Model yüklenemedi!")
        print("Önce main_advanced.py ile modeli eğitin.")
        return
    
    while True:
        try:
            # Kullanıcı girdisi al
            kullanici_girdisi = tahminci.kullanici_girdisi_al()
            
            # Tahmin yap
            print("\n🔄 Tahmin yapılıyor...")
            tahmin = tahminci.tahmin_yap(kullanici_girdisi)
            
            # Sonuçları göster
            tahminci.sonuclari_goster(kullanici_girdisi, tahmin)
            
            # Devam etmek isteyip istemediğini sor
            print(f"\n" + "="*50)
            devam = input("🔄 Başka bir araç için tahmin yapmak ister misiniz? (e/h): ").lower()
            
            if devam not in ['e', 'evet', 'y', 'yes']:
                print("\n👋 Teşekkürler! İyi günler!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı!")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            print("Tekrar deneyin...")

if __name__ == "__main__":
    main()
