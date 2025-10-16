"""
Basit Araç Fiyat Tahmini
"""

import pandas as pd
import numpy as np
import joblib
import os

def modeli_yukle():
    """Modeli yükle"""
    try:
        model_data = joblib.load('models/hizli_arac_modeli.pkl')
        print("Model yuklendi!")
        return model_data
    except:
        print("Model yuklenemedi!")
        return None

def araç_bilgileri_al():
    """Araç bilgilerini al"""
    print("\n=== ARAÇ FİYAT TAHMİNİ ===")
    
    # Marka
    markalar = ['BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Toyota', 'Ford', 'Renault', 
               'Opel', 'Hyundai', 'Fiat', 'Peugeot', 'Nissan', 'Honda', 'Mazda', 'Kia']
    
    print("\nMarka secin:")
    for i, marka in enumerate(markalar, 1):
        print(f"{i:2d}. {marka}")
    
    while True:
        try:
            secim = int(input("\nMarka numarasi (1-15): ")) - 1
            if 0 <= secim < len(markalar):
                marka = markalar[secim]
                break
            else:
                print("Gecersiz secim!")
        except:
            print("Sayi girin!")
    
    # Model yılı
    while True:
        try:
            model_yili = int(input("Model yili (1990-2025): "))
            if 1990 <= model_yili <= 2025:
                break
            else:
                print("Gecersiz yil!")
        except:
            print("Sayi girin!")
    
    # Kilometre
    while True:
        try:
            kilometre = int(input("Kilometre: "))
            if kilometre >= 0:
                break
            else:
                print("Pozitif sayi girin!")
        except:
            print("Sayi girin!")
    
    # Motor hacmi
    motor_hacimleri = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
    print("\nMotor hacmi secin:")
    for i, hacim in enumerate(motor_hacimleri, 1):
        print(f"{i}. {hacim}L")
    
    while True:
        try:
            secim = int(input("Motor hacmi (1-9): ")) - 1
            if 0 <= secim < len(motor_hacimleri):
                motor_hacmi = motor_hacimleri[secim]
                break
            else:
                print("Gecersiz secim!")
        except:
            print("Sayi girin!")
    
    # Yakıt türü
    yakit_turleri = ['Benzin', 'Dizel', 'LPG', 'Hibrit']
    print("\nYakit turu secin:")
    for i, yakit in enumerate(yakit_turleri, 1):
        print(f"{i}. {yakit}")
    
    while True:
        try:
            secim = int(input("Yakit turu (1-4): ")) - 1
            if 0 <= secim < len(yakit_turleri):
                yakit_turu = yakit_turleri[secim]
                break
            else:
                print("Gecersiz secim!")
        except:
            print("Sayi girin!")
    
    # Vites türü
    vites_turleri = ['Manuel', 'Otomatik', 'Yarı Otomatik']
    print("\nVites turu secin:")
    for i, vites in enumerate(vites_turleri, 1):
        print(f"{i}. {vites}")
    
    while True:
        try:
            secim = int(input("Vites turu (1-3): ")) - 1
            if 0 <= secim < len(vites_turleri):
                vites_turu = vites_turleri[secim]
                break
            else:
                print("Gecersiz secim!")
        except:
            print("Sayi girin!")
    
    # Gövde tipi
    govde_tipleri = ['Sedan', 'Hatchback', 'SUV', 'Station Wagon', 'Coupe']
    print("\nGovde tipi secin:")
    for i, govde in enumerate(govde_tipleri, 1):
        print(f"{i}. {govde}")
    
    while True:
        try:
            secim = int(input("Govde tipi (1-5): ")) - 1
            if 0 <= secim < len(govde_tipleri):
                govde_tipi = govde_tipleri[secim]
                break
            else:
                print("Gecersiz secim!")
        except:
            print("Sayi girin!")
    
    # Motor gücü
    while True:
        try:
            motor_gucu = int(input("Motor gucu (HP): "))
            if motor_gucu > 0:
                break
            else:
                print("Pozitif sayi girin!")
        except:
            print("Sayi girin!")
    
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

def veriyi_hazirla(model_data, kullanici_girdisi):
    """Veriyi model için hazırla"""
    
    # DataFrame oluştur
    df = pd.DataFrame([kullanici_girdisi])
    
    # Feature engineering
    df['arac_yasi'] = 2025 - df['model_yili']
    df['yillik_ortalama_km'] = df['kilometre'] / (df['arac_yasi'] + 1)
    df['km_per_motor_hacmi'] = df['kilometre'] / (df['motor_hacmi'] + 0.1)
    df['guc_per_hacim'] = df['motorgucu'] / (df['motor_hacmi'] + 0.1)
    
    # Kategorik değişkenleri encode et
    label_encoders = model_data['label_encoders']
    categorical_features = ['marka', 'vites_turu', 'yakit_turu', 'kasatipi']
    
    for feature in categorical_features:
        if feature in label_encoders:
            try:
                df[feature + '_encoded'] = label_encoders[feature].transform(df[feature].astype(str))
            except:
                # Bilinmeyen kategori için varsayılan değer
                df[feature + '_encoded'] = 0
    
    # Özellik seçimi
    feature_names = model_data['feature_names']
    available_features = [col for col in feature_names if col in df.columns]
    
    return df[available_features]

def tahmin_yap(model_data, kullanici_girdisi):
    """Fiyat tahmini yap"""
    try:
        # Veriyi hazırla
        X = veriyi_hazirla(model_data, kullanici_girdisi)
        
        # Tahmin yap
        model = model_data['model']
        tahmin = model.predict(X)[0]
        
        return tahmin
        
    except Exception as e:
        print(f"Tahmin hatasi: {e}")
        return None

def sonuclari_goster(kullanici_girdisi, tahmin):
    """Sonuçları göster"""
    print("\n" + "="*50)
    print("TAHMİN SONUÇLARI")
    print("="*50)
    
    print(f"\nArac Bilgileri:")
    print(f"   Marka: {kullanici_girdisi['marka']}")
    print(f"   Model Yili: {kullanici_girdisi['model_yili']}")
    print(f"   Arac Yasi: {2025 - kullanici_girdisi['model_yili']} yil")
    print(f"   Kilometre: {kullanici_girdisi['kilometre']:,} km")
    print(f"   Motor Hacmi: {kullanici_girdisi['motor_hacmi']}L")
    print(f"   Motor Gucu: {kullanici_girdisi['motorgucu']} HP")
    print(f"   Yakit Turu: {kullanici_girdisi['yakit_turu']}")
    print(f"   Vites: {kullanici_girdisi['vites_turu']}")
    print(f"   Govde: {kullanici_girdisi['kasatipi']}")
    
    if tahmin:
        print(f"\nTAHMİN EDİLEN FİYAT:")
        print(f"   {tahmin:,.0f} TL")
        
        # Fiyat aralığı
        alt_fiyat = tahmin * 0.9
        ust_fiyat = tahmin * 1.1
        print(f"\nFİYAT ARALIĞI:")
        print(f"   {alt_fiyat:,.0f} - {ust_fiyat:,.0f} TL")
        
        # Öneriler
        print(f"\nONERİLER:")
        arac_yasi = 2025 - kullanici_girdisi['model_yili']
        
        if arac_yasi > 10:
            print("   - 10 yas uzeri arac - detayli muayene onerilir")
        
        if kullanici_girdisi['kilometre'] > 200000:
            print("   - Yuksek kilometre - bakim gecmisi onemli")
        
        if kullanici_girdisi['motor_hacmi'] < 1.4:
            print("   - Dusuk motor hacmi - yakit tasarrufu avantaji")
        
        if tahmin > 500000:
            print("   - Yuksek degerli arac - sigorta ve guvenlik onemli")
        
    else:
        print("Tahmin yapilamadi!")

def main():
    """Ana fonksiyon"""
    print("TURKIYE ARAÇ FİYAT TAHMİNİ")
    print("Açıklanabilir Makine Öğrenmesi ile")
    print("="*40)
    
    # Modeli yükle
    model_data = modeli_yukle()
    if not model_data:
        print("Model yuklenemedi!")
        return
    
    while True:
        try:
            # Kullanıcı girdisi al
            kullanici_girdisi = araç_bilgileri_al()
            
            # Tahmin yap
            print("\nTahmin yapiliyor...")
            tahmin = tahmin_yap(model_data, kullanici_girdisi)
            
            # Sonuçları göster
            sonuclari_goster(kullanici_girdisi, tahmin)
            
            # Devam etmek isteyip istemediğini sor
            print(f"\n" + "="*50)
            devam = input("Baska bir arac icin tahmin yapmak ister misiniz? (e/h): ").lower()
            
            if devam not in ['e', 'evet', 'y', 'yes']:
                print("\nTesekkurler! İyi gunler!")
                break
                
        except KeyboardInterrupt:
            print("\n\nProgram sonlandirildi!")
            break
        except Exception as e:
            print(f"\nHata: {e}")
            print("Tekrar deneyin...")

if __name__ == "__main__":
    main()
