"""
Türkiye İkinci El Araç Fiyat Tahmini - Ana Proje Dosyası
Açıklanabilir Makine Öğrenmesi ile Araç Fiyat Tahmini

Hazırlayanlar: Seyfullah Adıgüzel, Batuhan Orkun İnce
Danışman: Sinan Keskin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import os
import warnings
warnings.filterwarnings('ignore')

class ArabaFiyatTahmini:
    """Araç fiyat tahmini için ana sınıf"""
    
    def __init__(self):
        """Sınıfı başlatır"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def veri_yukle(self, dosya_yolu=None):
        """Veri setini yükler - önce Kaggle verisini dener, sonra örnek veriyi"""
        
        # Önce Kaggle veri setini dene
        if dosya_yolu is None:
            kaggle_dosya = 'data/kaggle_arac_verisi_processed.csv'
            ornek_dosya = 'data/ornek_ikinci_el_arac_verisi.csv'
            
            if os.path.exists(kaggle_dosya):
                dosya_yolu = kaggle_dosya
                print("Kaggle veri seti bulundu!")
            elif os.path.exists(ornek_dosya):
                dosya_yolu = ornek_dosya
                print("Örnek veri seti kullanılıyor.")
            else:
                print("Veri dosyası bulunamadı! Önce veri toplama scriptini çalıştırın.")
                return False
        
        try:
            self.df = pd.read_csv(dosya_yolu)
            print(f"Veri seti yüklendi! Boyut: {self.df.shape}")
            print(f"Dosya: {dosya_yolu}")
            return True
        except FileNotFoundError:
            print(f"Veri dosyası bulunamadı: {dosya_yolu}")
            return False
    
    def veri_onisle(self):
        """Veri önişleme işlemlerini yapar"""
        if self.df is None:
            print("Önce veri setini yükleyin!")
            return
        
        print("Veri önişleme başlatılıyor...")
        
        # Eksik değerleri kontrol et
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"Eksik değerler bulundu: {missing_data[missing_data > 0].to_dict()}")
            # Eksik değerleri doldur
            for col in missing_data[missing_data > 0].index:
                if self.df[col].dtype in ['object']:
                    self.df[col] = self.df[col].fillna('Bilinmiyor')
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Kategorik değişkenleri encode et
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['kaynak', 'source']:  # kaynak sütunlarını atla
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        # Yeni özellikler türet (Feature Engineering)
        if 'model_yili' in self.df.columns:
            self.df['arac_yasi'] = 2024 - self.df['model_yili']
            if 'kilometre' in self.df.columns:
                self.df['yillik_ortalama_km'] = self.df['kilometre'] / (self.df['arac_yasi'] + 1)
        
        if 'kilometre' in self.df.columns and 'motor_hacmi' in self.df.columns:
            self.df['km_per_motor_hacmi'] = self.df['kilometre'] / (self.df['motor_hacmi'] + 0.1)
        
        # Aykırı değerleri temizle (fiyat için)
        if 'fiyat' in self.df.columns:
            Q1 = self.df['fiyat'].quantile(0.25)
            Q3 = self.df['fiyat'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(self.df)
            self.df = self.df[(self.df['fiyat'] >= lower_bound) & (self.df['fiyat'] <= upper_bound)]
            outliers_after = len(self.df)
            
            if outliers_before != outliers_after:
                print(f"Aykırı değerler temizlendi: {outliers_before - outliers_after} kayıt çıkarıldı")
        
        print("Veri önişleme tamamlandı!")
        print(f"İşlenmiş veri boyutu: {self.df.shape}")
        print(f"Toplam sütun sayısı: {self.df.shape[1]}")
    
    def veri_bol(self):
        """Veriyi eğitim ve test setlerine böler"""
        if self.df is None:
            print("Önce veri setini yükleyin!")
            return
        
        # Hedef değişken ve özellikler
        y = self.df['fiyat']
        X = self.df.drop(['fiyat', 'kaynak'], axis=1, errors='ignore')
        
        # Veriyi böl
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Özellikleri ölçekle
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Veri bölündü - Eğitim: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
    
    def modelleri_egit(self):
        """Farklı modelleri eğitir"""
        if self.X_train is None:
            print("Önce veriyi bölün!")
            return
        
        print("Modeller eğitiliyor...")
        
        # 1. Linear Regression (Baseline)
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(self.X_train_scaled, self.y_train)
        
        # 2. Random Forest
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        
        # 3. XGBoost
        self.models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models['XGBoost'].fit(self.X_train, self.y_train)
        
        print("Modeller eğitildi!")
    
    def model_performanslarini_degerlendir(self):
        """Model performanslarını değerlendirir"""
        if not self.models:
            print("Önce modelleri eğitin!")
            return
        
        print("Model performansları değerlendiriliyor...")
        
        sonuclar = []
        
        for model_adi, model in self.models.items():
            # Tahminler
            if model_adi == 'Linear Regression':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Metrikler
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            sonuclar.append({
                'Model': model_adi,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            })
        
        # Sonuçları DataFrame olarak döndür
        performans_df = pd.DataFrame(sonuclar)
        performans_df = performans_df.sort_values('R²', ascending=False)
        
        print("\n=== MODEL PERFORMANS SONUÇLARI ===")
        print(performans_df.to_string(index=False))
        
        return performans_df
    
    def shap_analizi(self):
        """SHAP ile model açıklanabilirliği"""
        if 'Random Forest' not in self.models:
            print("Random Forest modeli bulunamadı!")
            return
        
        print("SHAP analizi yapılıyor...")
        
        # SHAP explainer oluştur
        explainer = shap.TreeExplainer(self.models['Random Forest'])
        shap_values = explainer.shap_values(self.X_test)
        
        # Özellik önem grafiği
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title('SHAP Özellik Önem Analizi')
        plt.tight_layout()
        plt.savefig('results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Özellik önem değerleri
        feature_importance = pd.DataFrame({
            'özellik': self.X_test.columns,
            'ortalama_shap': np.abs(shap_values).mean(0)
        }).sort_values('ortalama_shap', ascending=False)
        
        print("\n=== SHAP ÖZELLİK ÖNEM SIRALAMASI ===")
        print(feature_importance.head(10))
        
        return shap_values, feature_importance
    
    def lime_analizi(self, ornek_indeks=0):
        """LIME ile lokal açıklama"""
        if 'Random Forest' not in self.models:
            print("Random Forest modeli bulunamadı!")
            return
        
        print(f"LIME analizi yapılıyor (Örnek {ornek_indeks})...")
        
        # LIME explainer oluştur
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_test.columns,
            class_names=['fiyat'],
            mode='regression'
        )
        
        # Seçilen örnek için açıklama
        explanation = explainer.explain_instance(
            self.X_test.iloc[ornek_indeks].values,
            self.models['Random Forest'].predict,
            num_features=len(self.X_test.columns)
        )
        
        print(f"\n=== LIME AÇIKLAMA (Örnek {ornek_indeks}) ===")
        print(f"Gerçek fiyat: {self.y_test.iloc[ornek_indeks]:,.0f} TL")
        print(f"Tahmin edilen fiyat: {self.models['Random Forest'].predict(self.X_test.iloc[ornek_indeks:ornek_indeks+1])[0]:,.0f} TL")
        
        # LIME görselleştirmesi
        explanation.show_in_notebook(show_table=True)
        
        return explanation
    
    def sonuclari_raporla(self):
        """Tüm sonuçları raporlar"""
        print("\n" + "="*50)
        print("TÜRKIYE İKİNCİ EL ARAÇ FİYAT TAHMİNİ RAPORU")
        print("="*50)
        
        if self.df is not None:
            print(f"\nVeri Seti Bilgileri:")
            print(f"- Toplam araç sayısı: {len(self.df):,}")
            print(f"- Özellik sayısı: {self.df.shape[1]}")
            print(f"- Fiyat aralığı: {self.df['fiyat'].min():,.0f} - {self.df['fiyat'].max():,.0f} TL")
            print(f"- Ortalama fiyat: {self.df['fiyat'].mean():,.0f} TL")
        
        if self.models:
            performans = self.model_performanslarini_degerlendir()
            
            en_iyi_model = performans.iloc[0]['Model']
            en_iyi_r2 = performans.iloc[0]['R²']
            
            print(f"\nEn İyi Model: {en_iyi_model}")
            print(f"R² Skoru: {en_iyi_r2:.4f}")
        
        print("\nProje tamamlandı! Sonuçlar 'results/' klasöründe saklanmıştır.")

def main():
    """Ana fonksiyon"""
    print("Türkiye İkinci El Araç Fiyat Tahmini Projesi Başlatılıyor...")
    
    # Proje sınıfını başlat
    proje = ArabaFiyatTahmini()
    
    # 1. Veri yükleme
    if not proje.veri_yukle():
        print("Veri yükleme başarısız! Önce veri toplama scriptini çalıştırın.")
        return
    
    # 2. Veri önişleme
    proje.veri_onisle()
    
    # 3. Veri bölme
    proje.veri_bol()
    
    # 4. Model eğitimi
    proje.modelleri_egit()
    
    # 5. Performans değerlendirme
    proje.model_performanslarini_degerlendir()
    
    # 6. SHAP analizi
    try:
        proje.shap_analizi()
    except Exception as e:
        print(f"SHAP analizi hatası: {e}")
    
    # 7. LIME analizi
    try:
        proje.lime_analizi()
    except Exception as e:
        print(f"LIME analizi hatası: {e}")
    
    # 8. Sonuç raporu
    proje.sonuclari_raporla()

if __name__ == "__main__":
    main()

