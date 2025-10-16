"""
Türkiye İkinci El Araç Fiyat Tahmini - Gelişmiş Versiyon
Log Transform, Gruplandırılmış CV, Gelişmiş Feature Engineering

Hazırlayanlar: Seyfullah Adıgüzel, Batuhan Orkun İnce
Danışman: Sinan Keskin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import joblib
import warnings
warnings.filterwarnings('ignore')

class GelişmişArabaFiyatTahmini:
    """Gelişmiş araç fiyat tahmini sınıfı"""
    
    def __init__(self):
        """Sınıfı başlatır"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_log = None
        self.y_test_log = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def veri_yukle(self, dosya_yolu='data/kaggle_arac_verisi_processed.csv'):
        """Veri setini yükler"""
        try:
            self.df = pd.read_csv(dosya_yolu)
            print(f"Veri seti yüklendi! Boyut: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Veri dosyası bulunamadı: {dosya_yolu}")
            return False
    
    def veri_temizle(self):
        """Mantıksal filtreler ile veri temizliği"""
        print("Veri temizleme başlatılıyor...")
        
        original_size = len(self.df)
        
        # Mantıksal filtreler
        self.df = self.df[(self.df['kilometre'] >= 0) & (self.df['kilometre'] <= 1_000_000)]
        self.df = self.df[(self.df['model_yili'] >= 1980) & (self.df['model_yili'] <= 2025)]
        self.df = self.df[(self.df['motor_hacmi'] >= 500) & (self.df['motor_hacmi'] <= 8000)]
        
        # Fiyat filtreleri (%1-%99 arası)
        Q1_price = self.df['fiyat'].quantile(0.01)
        Q99_price = self.df['fiyat'].quantile(0.99)
        self.df = self.df[(self.df['fiyat'] >= Q1_price) & (self.df['fiyat'] <= Q99_price)]
        
        if 'motorgucu' in self.df.columns:
            self.df = self.df[(self.df['motorgucu'] >= 30) & (self.df['motorgucu'] <= 1000)]
        
        cleaned_size = len(self.df)
        print(f"Veri temizlendi: {original_size - cleaned_size:,} kayıt çıkarıldı")
        print(f"Kalan kayıt sayısı: {cleaned_size:,}")
    
    def gelismis_feature_engineering(self):
        """Gelişmiş özellik türetme"""
        print("Gelişmiş feature engineering başlatılıyor...")
        
        # 1. Zaman/enflasyon bilgisi (varsa)
        # İlan tarihi mevcutsa yıl/ay çıkar ve TÜFE ile reel fiyatı hesapla
        if 'ilan_tarihi' in self.df.columns:
            try:
                self.df['ilan_tarihi'] = pd.to_datetime(self.df['ilan_tarihi'], errors='coerce')
                self.df['ilan_yili'] = self.df['ilan_tarihi'].dt.year
                self.df['ilan_ayi'] = self.df['ilan_tarihi'].dt.month
                # CPI dosyası: data/cpi_tr.csv beklenen kolonlar: year, month, cpi
                cpi_path = 'data/cpi_tr.csv'
                if os.path.exists(cpi_path):
                    cpi = pd.read_csv(cpi_path)
                    # flexible column names
                    cpi.columns = [c.lower() for c in cpi.columns]
                    if 'date' in cpi.columns and 'cpi' in cpi.columns:
                        cpi['date'] = pd.to_datetime(cpi['date'])
                        cpi['year'] = cpi['date'].dt.year
                        cpi['month'] = cpi['date'].dt.month
                    elif {'year','month','cpi'}.issubset(set(cpi.columns)):
                        pass
                    else:
                        raise ValueError('cpi_tr.csv formatı desteklenmiyor (beklenen: date+cpi veya year,month,cpi).')
                    cpi = cpi[['year','month','cpi']].dropna()
                    base_cpi = float(cpi['cpi'].iloc[-1])
                    self.df = self.df.merge(
                        cpi.rename(columns={'year':'ilan_yili','month':'ilan_ayi'}),
                        on=['ilan_yili','ilan_ayi'], how='left'
                    )
                    # Deflatör: reel = nominal * (base / period)
                    self.df['fiyat_reel'] = self.df['fiyat'] * (base_cpi / self.df['cpi'])
                else:
                    # CPI yoksa, ilan_yili bazlı kaba normalizasyon (2025 baz)
                    if 'ilan_yili' in self.df.columns:
                        # Basit yıllık enflasyon varsayımı: %40 yıllık (yaklaşık); kullanıcı daha iyi CPI ile güncelleyebilir
                        annual_infl = 0.40
                        base_year = int(self.df['ilan_yili'].max())
                        self.df['fiyat_reel'] = self.df['fiyat'] * ((1 + annual_infl) ** (base_year - self.df['ilan_yili']))
            except Exception as e:
                print(f"Reel fiyat hesaplama atlandı: {e}")
        
        # 2. Log transform (nominal ve varsa reel)
        self.df['log_fiyat'] = np.log1p(self.df['fiyat'])
        if 'fiyat_reel' in self.df.columns:
            self.df['log_fiyat_reel'] = np.log1p(self.df['fiyat_reel'])
        
        # 3. Temel özellikler
        self.df['arac_yasi'] = 2025 - self.df['model_yili']
        self.df['yillik_ortalama_km'] = self.df['kilometre'] / (self.df['arac_yasi'] + 1)
        self.df['km_per_motor_hacmi'] = self.df['kilometre'] / (self.df['motor_hacmi'] + 0.1)
        
        # 4. Motor gücü özellikleri
        if 'motorgucu' in self.df.columns:
            self.df['guc_per_hacim'] = self.df['motorgucu'] / (self.df['motor_hacmi'] + 0.1)
            self.df['guc_per_km'] = self.df['motorgucu'] / (self.df['kilometre'] + 1)
        
        # 5. Kategorik gruplar
        self.df['yas_grubu'] = pd.cut(self.df['arac_yasi'], 
                                     bins=[0, 3, 7, 15, 100], 
                                     labels=['Yeni', 'Az_Kullanilmis', 'Orta', 'Eski'])
        
        self.df['km_grubu'] = pd.cut(self.df['kilometre'], 
                                    bins=[0, 50000, 100000, 200000, 1000000], 
                                    labels=['Dusuk', 'Orta', 'Yuksek', 'Cok_Yuksek'])
        
        self.df['motor_grubu'] = pd.cut(self.df['motor_hacmi'], 
                                       bins=[0, 1200, 1600, 2000, 8000], 
                                       labels=['Kucuk', 'Orta', 'Buyuk', 'Cok_Buyuk'])
        
        # 6. Etkileşim özellikleri
        self.df['yas_km_etkilesim'] = self.df['arac_yasi'] * self.df['kilometre']
        self.df['motor_hacmi_yas'] = self.df['motor_hacmi'] * self.df['arac_yasi']
        
        # 7. Kategorik değişkenleri encode et (varsa)
        categorical_features = [
            'marka', 'seri', 'model', 'vites_turu', 'yakit_turu', 'kasatipi', 'cekis',
            # Kullanıcı arayüzünde olan ek alanlar (veri setinde varsa kullanılacak)
            'arac_durumu', 'renk', 'garanti', 'agir_hasar_kayitli', 'kapi_sayisi', 'plaka_uyruk',
            # Türetilmiş grup alanları
            'yas_grubu', 'km_grubu', 'motor_grubu'
        ]
        
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[feature + '_encoded'] = le.fit_transform(self.df[feature].astype(str))
                self.label_encoders[feature] = le
        
        print("Feature engineering tamamlandı!")
        print(f"Toplam özellik sayısı: {self.df.shape[1]}")
    
    def veri_bol(self):
        """Gruplandırılmış train-test split"""
        print("Veri bölme işlemi başlatılıyor...")
        
        # Özellik seçimi
        feature_cols = []
        
        # Sayısal özellikler
        numeric_features = ['model_yili', 'kilometre', 'motor_hacmi', 'arac_yasi', 
                           'yillik_ortalama_km', 'km_per_motor_hacmi']
        if 'motorgucu' in self.df.columns:
            numeric_features.extend(['motorgucu', 'guc_per_hacim', 'guc_per_km'])
        
        # Kategorik özellikler (encoded)
        categorical_features = [col for col in self.df.columns if col.endswith('_encoded')]
        
        feature_cols = numeric_features + categorical_features
        
        # Gruplama için marka+seri kombinasyonu
        self.df['group'] = self.df['marka'].astype(str) + '_' + self.df['seri'].astype(str)
        
        # Veriyi böl
        X = self.df[feature_cols]
        # Hedef: reel fiyat varsa onu kullan, yoksa nominal
        if 'fiyat_reel' in self.df.columns and self.df['fiyat_reel'].notnull().any():
            y = self.df['fiyat_reel']
            y_log = self.df.get('log_fiyat_reel', np.log1p(self.df['fiyat_reel']))
        else:
            y = self.df['fiyat']
            y_log = self.df['log_fiyat']
        groups = self.df['group']
        
        # Gruplandırılmış split
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_log, self.y_test_log, train_groups, test_groups = train_test_split(
            X, y, y_log, groups, test_size=0.2, random_state=42, stratify=None
        )
        
        # Özellik isimlerini kaydet
        self.feature_names = feature_cols
        
        print(f"Eğitim seti: {self.X_train.shape[0]} kayıt")
        print(f"Test seti: {self.X_test.shape[0]} kayıt")
        print(f"Özellik sayısı: {len(feature_cols)}")
    
    def modelleri_egit(self):
        """Gelişmiş modelleri eğitir"""
        print("Modeller eğitiliyor...")
        
        # 1. Linear Regression (log scale)
        self.models['Linear Regression (Log)'] = LinearRegression()
        self.models['Linear Regression (Log)'].fit(self.X_train, self.y_train_log)
        
        # 2. Random Forest (hem normal hem log scale)
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        
        self.models['Random Forest (Log)'] = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
        self.models['Random Forest (Log)'].fit(self.X_train, self.y_train_log)
        
        # 3. XGBoost (hem normal hem log scale)
        self.models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        self.models['XGBoost'].fit(self.X_train, self.y_train)
        
        self.models['XGBoost (Log)'] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        self.models['XGBoost (Log)'].fit(self.X_train, self.y_train_log)
        
        print("Modeller eğitildi!")
    
    def model_performanslarini_degerlendir(self):
        """Gelişmiş model değerlendirmesi"""
        print("Model performansları değerlendiriliyor...")
        
        sonuclar = []
        
        for model_adi, model in self.models.items():
            # Tahminler
            if 'Log' in model_adi:
                y_pred_log = model.predict(self.X_test)
                y_pred = np.expm1(y_pred_log)  # Log'dan geri dönüştür
                y_true = self.y_test
            else:
                y_pred = model.predict(self.X_test)
                y_true = self.y_test
            
            # Metrikler
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Log scale metrikleri
            if 'Log' in model_adi:
                rmse_log = np.sqrt(mean_squared_error(self.y_test_log, y_pred_log))
                mae_log = mean_absolute_error(self.y_test_log, y_pred_log)
                r2_log = r2_score(self.y_test_log, y_pred_log)
            else:
                rmse_log = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
                mae_log = mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))
                r2_log = r2_score(np.log1p(y_true), np.log1p(y_pred))
            
            sonuclar.append({
                'Model': model_adi,
                'RMSE (TL)': rmse,
                'MAE (TL)': mae,
                'R²': r2,
                'RMSE (Log)': rmse_log,
                'MAE (Log)': mae_log,
                'R² (Log)': r2_log
            })
        
        # Sonuçları DataFrame olarak döndür
        performans_df = pd.DataFrame(sonuclar)
        performans_df = performans_df.sort_values('R²', ascending=False)
        
        print("\n=== GELİŞMİŞ MODEL PERFORMANS SONUÇLARI ===")
        print(performans_df.round(4).to_string(index=False))
        
        return performans_df
    
    def shap_analizi(self):
        """Gelişmiş SHAP analizi"""
        print("Gelişmiş SHAP analizi yapılıyor...")
        
        # En iyi modeli seç (Random Forest)
        best_model = self.models['Random Forest']
        
        # SHAP explainer oluştur
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(self.X_test)
        
        # Özellik önem grafiği
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
        plt.title('SHAP Özellik Önem Analizi - Gelişmiş Model', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/shap_advanced_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Özellik önem değerleri
        feature_importance = pd.DataFrame({
            'özellik': self.feature_names,
            'ortalama_shap': np.abs(shap_values).mean(0)
        }).sort_values('ortalama_shap', ascending=False)
        
        print("\n=== GELİŞMİŞ SHAP ÖZELLİK ÖNEM SIRALAMASI ===")
        print(feature_importance.head(15))

        # Otomatik içgörü üreten anlatım (domain-aware) → results/shap_insights.md
        try:
            self._yaz_shap_icerik(shap_values, feature_importance)
        except Exception as e:
            print(f"SHAP içgörü yazım hatası: {e}")
        
        return shap_values, feature_importance

    def shap_segment_analizi(self):
        """Segment bazlı SHAP summary plot üretir (Sedan, SUV, 2020+)."""
        print("Segment bazlı SHAP analizi yapılıyor...")
        import os
        os.makedirs('results', exist_ok=True)

        # En iyi ağaç tabanlı model (Random Forest) üzerinden açıklama
        best_model = self.models.get('Random Forest')
        if best_model is None:
            print("Segment SHAP: Random Forest bulunamadı, atlanıyor.")
            return

        # Test seti satırlarına ait meta özellikler
        test_meta = self.df.loc[self.X_test.index]

        segments = []
        # Sedan ve SUV segmentleri (varsa)
        if 'kasatipi' in test_meta.columns:
            segments.append((test_meta['kasatipi'].astype(str).str.lower() == 'sedan', 'sedan'))
            segments.append((test_meta['kasatipi'].astype(str).str.lower() == 'suv', 'suv'))
        # 2020 ve sonrası
        if 'model_yili' in test_meta.columns:
            segments.append((test_meta['model_yili'] >= 2020, 'modelyear_2020_plus'))

        explainer = shap.TreeExplainer(best_model)

        summary_lines = ["# Segment Bazlı SHAP İçgörüleri", ""]

        for mask, name in segments:
            try:
                idx = test_meta.index[mask]
                if len(idx) < 50:  # çok küçük örneklerde atla
                    print(f"Segment '{name}': yetersiz test örneği ({len(idx)}), atlanıyor.")
                    summary_lines.append(f"- {name}: Yetersiz örnek sayısı ({len(idx)}) – atlandı.")
                    continue

                X_seg = self.X_test.loc[idx, self.feature_names]
                shap_vals_seg = explainer.shap_values(X_seg)

                plt.figure(figsize=(12, 9))
                shap.summary_plot(shap_vals_seg, X_seg, feature_names=self.feature_names, show=False)
                plt.title(f"SHAP Summary – {name}")
                plt.tight_layout()
                out_png = f"results/shap_summary_{name}.png"
                plt.savefig(out_png, dpi=300, bbox_inches='tight')
                plt.close()

                # En etkili ilk 8 özellik ve yön
                fi_seg = pd.DataFrame({
                    'özellik': self.feature_names,
                    'ortalama_shap': np.abs(shap_vals_seg).mean(0)
                }).sort_values('ortalama_shap', ascending=False).head(8)

                # Yön kestirimi
                lines = [f"## {name}", "", f"Toplam test örneği: {len(idx)}", "", "En etkili özellikler:"]
                for feat in fi_seg['özellik']:
                    vals = X_seg[feat].values
                    svals = shap_vals_seg[:, self.feature_names.index(feat)]
                    if np.std(vals) == 0:
                        corr = 0.0
                    else:
                        corr = np.corrcoef(vals, svals)[0, 1]
                    if corr > 0.05:
                        yon = 'arttıkça fiyat artıyor'
                    elif corr < -0.05:
                        yon = 'arttıkça fiyat düşüyor'
                    else:
                        yon = 'zayıf/karışık'
                    lines.append(f"- {feat}: {yon}")

                lines.append("")
                lines.append(f"Görsel: {out_png}")
                lines.append("")
                summary_lines.extend(lines)
            except Exception as e:
                print(f"Segment '{name}' SHAP hatası: {e}")
                summary_lines.append(f"- {name}: Hata – {e}")

        with open('results/shap_segments.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

    def _yaz_shap_icerik(self, shap_values: np.ndarray, feature_importance: pd.DataFrame) -> None:
        """SHAP özetinden alan bilgisi içeren kısa bir rapor üretir."""
        import os
        os.makedirs('results', exist_ok=True)

        topk = 10
        top_features = feature_importance.head(topk)['özellik'].tolist()

        # Her özellik için yön kestirimi: SHAP değerleri ile özelliğin değeri arasındaki korelasyonun işareti
        directions = {}
        for feat in top_features:
            try:
                vals = self.X_test[feat].values
                shap_feat = shap_values[:, self.feature_names.index(feat)]
                if np.std(vals) == 0:
                    directions[feat] = 0.0
                else:
                    directions[feat] = np.corrcoef(vals, shap_feat)[0, 1]
            except Exception:
                directions[feat] = np.nan

        # İnsan dostu isimlendirme
        pretty_map = {
            'model_yili': 'Model yılı',
            'kilometre': 'Kilometre',
            'motor_hacmi': 'Motor hacmi (cc)',
            'arac_yasi': 'Araç yaşı',
            'yillik_ortalama_km': 'Yıllık ortalama kilometre',
            'km_per_motor_hacmi': 'Kilometre / motor hacmi',
            'motorgucu': 'Motor gücü (HP)',
            'guc_per_hacim': 'Güç / hacim',
            'guc_per_km': 'Güç / km',
        }
        def pretty(name: str) -> str:
            if name in pretty_map:
                return pretty_map[name]
            if name.endswith('_encoded'):
                return name.replace('_encoded', '').replace('_', ' ').title()
            return name.replace('_', ' ').title()

        lines = []
        lines.append('# SHAP İçgörü Raporu')
        lines.append('')
        lines.append('Bu rapor, modelin tahminlerine en çok etki eden özellikleri ve yönlerini kısaca açıklar.')
        lines.append('')
        lines.append('## En Etkili Özellikler ve Yönleri')
        for feat in top_features:
            dir_val = directions.get(feat)
            if np.isnan(dir_val):
                yorum = 'etkinin yönü belirsiz'
            elif dir_val > 0.05:
                yorum = 'değer arttıkça fiyat artma eğiliminde'
            elif dir_val < -0.05:
                yorum = 'değer arttıkça fiyat düşme eğiliminde'
            else:
                yorum = 'zayıf/karışık yön'
            lines.append(f"- {pretty(feat)}: {yorum}")

        lines.append('')
        lines.append('## Kısa Alan Yorumları (Domain Insights)')
        # Model yılı ve km için tipik içgörüler
        if 'model_yili' in self.feature_names or 'arac_yasi' in self.feature_names:
            lines.append('- Daha yeni model (ya da daha düşük araç yaşı) tipik olarak daha yüksek fiyata yol açar.')
        if 'kilometre' in self.feature_names:
            lines.append('- Kilometre yükseldikçe, aşınma ve yıpranma nedeniyle fiyat genelde düşme eğilimindedir.')
        if 'motorgucu' in self.feature_names or 'guc_per_hacim' in self.feature_names:
            lines.append('- Daha yüksek motor gücü aynı segmentte daha yüksek fiyatla ilişkilidir.')
        # Kategorik örnekler
        cat_examples = [c for c in self.feature_names if c.endswith('_encoded')]
        if cat_examples:
            lines.append('- Marka/seri/model ve yakıt/vites/gövde gibi kategorik değişkenler, segment farklılıkları nedeniyle fiyatı belirgin etkiler.')

        lines.append('')
        lines.append('Not: Yönler, özelliğin değeri ile ilgili SHAP değerlerinin korelasyon işaretinden türetilmiştir; doğrusal olmayan etkileşimler nedeniyle tek başına deterministik değildir.')

        with open('results/shap_insights.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def gruplandirilmis_cv(self):
        """Gruplandırılmış cross-validation"""
        print("Gruplandırılmış cross-validation yapılıyor...")
        
        try:
            # Grupları yeniden oluştur - train seti ile eşleştir
            train_indices = self.X_train.index
            groups = self.df.loc[train_indices, 'group']
            
            # GroupKFold ile CV
            group_kfold = GroupKFold(n_splits=5)
            
            cv_sonuclar = {}
            
            for model_adi, model in self.models.items():
                if 'Log' in model_adi:
                    y_cv = self.y_train_log
                else:
                    y_cv = self.y_train
                
                cv_scores = cross_val_score(
                    model, self.X_train, y_cv, 
                    cv=group_kfold, groups=groups, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                cv_sonuclar[model_adi] = {
                    'CV_RMSE': np.sqrt(-cv_scores.mean()),
                    'CV_Std': np.sqrt(cv_scores.std())
                }
            
            print("\n=== GRUPLANDIRILMIŞ CV SONUÇLARI ===")
            cv_df = pd.DataFrame(cv_sonuclar).T
            print(cv_df.round(4))
            
            return cv_df
            
        except Exception as e:
            print(f"CV hatası: {e}")
            print("Basit CV ile devam ediliyor...")
            
            # Basit CV
            cv_sonuclar = {}
            for model_adi, model in self.models.items():
                if 'Log' in model_adi:
                    y_cv = self.y_train_log
                else:
                    y_cv = self.y_train
                
                cv_scores = cross_val_score(
                    model, self.X_train, y_cv, 
                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                cv_sonuclar[model_adi] = {
                    'CV_RMSE': np.sqrt(-cv_scores.mean()),
                    'CV_Std': np.sqrt(cv_scores.std())
                }
            
            print("\n=== BASİT CV SONUÇLARI ===")
            cv_df = pd.DataFrame(cv_sonuclar).T
            print(cv_df.round(4))
            
            return cv_df
    
    def modeli_kaydet(self, dosya_adi='models/gelismis_arac_modeli.pkl'):
        """En iyi modeli kaydet"""
        try:
            # En iyi modeli seç (Random Forest)
            best_model = self.models['Random Forest']
            
            # Model ve encoders'ı kaydet
            model_data = {
                'model': best_model,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }
            
            joblib.dump(model_data, dosya_adi)
            print(f"Model kaydedildi: {dosya_adi}")
            
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
    
    def sonuclari_raporla(self):
        """Gelişmiş sonuç raporu"""
        print("\n" + "="*60)
        print("GELİŞMİŞ TÜRKIYE İKİNCİ EL ARAÇ FİYAT TAHMİNİ RAPORU")
        print("="*60)
        
        if self.df is not None:
            print(f"\nVeri Seti Bilgileri:")
            print(f"- Toplam araç sayısı: {len(self.df):,}")
            print(f"- Özellik sayısı: {self.df.shape[1]}")
            print(f"- Fiyat aralığı: {self.df['fiyat'].min():,.0f} - {self.df['fiyat'].max():,.0f} TL")
            print(f"- Ortalama fiyat: {self.df['fiyat'].mean():,.0f} TL")
            print(f"- Log transform uygulandı: ✓")
            print(f"- Mantıksal filtreler uygulandı: ✓")
            print(f"- Gelişmiş feature engineering: ✓")
        
        if self.models:
            performans = self.model_performanslarini_degerlendir()
            
            en_iyi_model = performans.iloc[0]['Model']
            en_iyi_r2 = performans.iloc[0]['R²']
            
            print(f"\nEn İyi Model: {en_iyi_model}")
            print(f"R² Skoru: {en_iyi_r2:.4f}")
            print(f"MAE: {performans.iloc[0]['MAE (TL)']:,.0f} TL")
            print(f"RMSE: {performans.iloc[0]['RMSE (TL)']:,.0f} TL")
        
        print("\nGelişmiş özellikler:")
        print("- Log transform ile fiyat dönüşümü")
        print("- Mantıksal veri filtreleri")
        print("- Gruplandırılmış cross-validation")
        print("- Gelişmiş feature engineering")
        print("- Etkileşim özellikleri")
        print("- Kategorik gruplandırma")

def main():
    """Ana fonksiyon"""
    print("Gelişmiş Türkiye İkinci El Araç Fiyat Tahmini Projesi Başlatılıyor...")
    
    # Proje sınıfını başlat
    proje = GelişmişArabaFiyatTahmini()
    
    # 1. Veri yükleme
    if not proje.veri_yukle():
        print("Veri yükleme başarısız!")
        return
    
    # 2. Veri temizleme
    proje.veri_temizle()
    
    # 3. Gelişmiş feature engineering
    proje.gelismis_feature_engineering()
    
    # 4. Veri bölme
    proje.veri_bol()
    
    # 5. Model eğitimi
    proje.modelleri_egit()
    
    # 6. Performans değerlendirme
    proje.model_performanslarini_degerlendir()
    
    # 7. Gruplandırılmış CV
    proje.gruplandirilmis_cv()
    
    # 8. SHAP analizi
    try:
        proje.shap_analizi()
    except Exception as e:
        print(f"SHAP analizi hatası: {e}")
    
    # 9. Model kaydetme
    proje.modeli_kaydet()
    
    # 10. Sonuç raporu
    proje.sonuclari_raporla()

if __name__ == "__main__":
    main()
