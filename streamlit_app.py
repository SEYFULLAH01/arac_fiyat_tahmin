"""
Türkiye İkinci El Araç Fiyat Tahmini - Streamlit Web Uygulaması
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Türkiye Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("🚗 Türkiye İkinci El Araç Fiyat Tahmini")
st.markdown("**Açıklanabilir Makine Öğrenmesi ile Araç Fiyat Tahmini**")
st.markdown("---")

# Sidebar - Kullanıcı girişleri
st.sidebar.header("📝 Araç Bilgilerini Girin")

# Marka seçimi
markalar = [
    'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Toyota', 'Ford', 'Renault', 
    'Opel', 'Hyundai', 'Fiat', 'Peugeot', 'Nissan', 'Honda', 'Mazda', 
    'Kia', 'Skoda', 'Seat', 'Citroen', 'Dacia', 'Chevrolet'
]

marka = st.sidebar.selectbox("Marka", markalar)

# Model yılı
model_yili = st.sidebar.slider("Model Yılı", min_value=1990, max_value=2025, value=2020)

# Kilometre
kilometre = st.sidebar.number_input("Kilometre", min_value=0, max_value=1000000, value=50000, step=1000)

# Motor hacmi
motor_hacmi = st.sidebar.selectbox("Motor Hacmi (L)", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0])

# Yakıt türü
yakit_turu = st.sidebar.selectbox("Yakıt Türü", ["Benzin", "Dizel", "LPG", "Hibrit", "Elektrik"])

# Vites türü
vites_turu = st.sidebar.selectbox("Vites Türü", ["Manuel", "Otomatik", "Yarı Otomatik"])

# Gövde tipi
govde_tipi = st.sidebar.selectbox("Gövde Tipi", ["Sedan", "Hatchback", "SUV", "Station Wagon", "Coupe", "Cabrio"])

# Çekiş tipi (opsiyonel, modelde varsa)
cekis = st.sidebar.selectbox("Çekiş", ["Önden", "Arkadan", "4x4"], index=0)

# Araç durumu (Sıfır / İkinci El)
arac_durumu = st.sidebar.selectbox("Araç Durumu", ["İkinci El", "Sıfır"], index=0)

# Renk
renk = st.sidebar.selectbox(
    "Renk",
    [
        "Beyaz", "Siyah", "Gri", "Gümüş", "Kırmızı", "Mavi", "Yeşil",
        "Bej", "Kahverengi", "Lacivert", "Turuncu", "Sarı", "Mor"
    ],
    index=0
)

# Garanti
garanti = st.sidebar.selectbox("Garanti", ["Yok", "Var"], index=0)

# Ağır hasar kaydı
agir_hasar_kayitli = st.sidebar.selectbox("Ağır Hasar Kayıtlı", ["Hayır", "Evet"], index=0)

# Kapı sayısı (opsiyonel)
kapi_sayisi = st.sidebar.selectbox("Kapı Sayısı", ["2", "3", "4", "5"], index=3)

# Plaka/Uyruk (opsiyonel)
plaka_uyruk = st.sidebar.selectbox("Plaka / Uyruk", ["Türkiye (TR) Plakalı", "Yabancı / Diğer"], index=0)

# Motor gücü (opsiyonel)
motor_gucu = st.sidebar.number_input("Motor Gücü (HP)", min_value=50, max_value=500, value=100, step=10)

# Tahmin butonu
if st.sidebar.button("🔮 Fiyat Tahmini Yap", type="primary"):
    
    # Kullanıcı girdilerini dataframe'e çevir
    # motor_hacmi litre → cc dönüşümü (eğitimde cc kullanılmış olabilir)
    motor_hacmi_cc = int(float(motor_hacmi) * 1000)

    user_input = pd.DataFrame({
        'marka': [marka],
        'seri': ['Genel'],
        'model': ['Standart'],
        'model_yili': [model_yili],
        'kilometre': [kilometre],
        'motor_hacmi': [motor_hacmi_cc],
        'yakit_turu': [yakit_turu],
        'vites_turu': [vites_turu],
        'kasatipi': [govde_tipi],
        'cekis': [cekis],
        'arac_durumu': [arac_durumu],
        'renk': [renk],
        'garanti': [garanti],
        'agir_hasar_kayitli': [agir_hasar_kayitli],
        'kapi_sayisi': [kapi_sayisi],
        'plaka_uyruk': [plaka_uyruk],
        'motorgucu': [motor_gucu]
    })
    
    # Feature engineering
    user_input['arac_yasi'] = 2025 - user_input['model_yili']
    user_input['yillik_ortalama_km'] = user_input['kilometre'] / (user_input['arac_yasi'] + 1)
    user_input['km_per_motor_hacmi'] = user_input['kilometre'] / (user_input['motor_hacmi'] + 0.1)
    user_input['guc_per_hacim'] = user_input['motorgucu'] / (user_input['motor_hacmi'] + 0.1)
    user_input['guc_per_km'] = user_input['motorgucu'] / (user_input['kilometre'] + 1)

    # Grup özellikleri (eğitimde kullanıldıysa)
    user_input['yas_grubu'] = pd.cut(
        user_input['arac_yasi'],
        bins=[0, 3, 7, 15, 100],
        labels=['Yeni', 'Az_Kullanilmis', 'Orta', 'Eski']
    )
    user_input['km_grubu'] = pd.cut(
        user_input['kilometre'],
        bins=[0, 50000, 100000, 200000, 1000000],
        labels=['Dusuk', 'Orta', 'Yuksek', 'Cok_Yuksek']
    )
    user_input['motor_grubu'] = pd.cut(
        user_input['motor_hacmi'],
        bins=[0, 1200, 1600, 2000, 8000],
        labels=['Kucuk', 'Orta', 'Buyuk', 'Cok_Buyuk']
    )
    
    # Ana içerik alanı
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Tahmin Sonuçları")

        # Gerçek eğitilmiş modeli yükle ve tahmin yap
        import os
        import shap
        from streamlit.components.v1 import html as st_html

        model_path_candidates = [
            'models/gelismis_arac_modeli.pkl',
            'models/hizli_arac_modeli.pkl'
        ]
        model_data = None
        for mp in model_path_candidates:
            if os.path.exists(mp):
                try:
                    model_data = joblib.load(mp)
                    break
                except Exception:
                    pass

        if model_data is None:
            st.error("Model dosyası bulunamadı. Lütfen önce modeli eğitip kaydedin.")
        else:
            model = model_data['model']
            label_encoders = model_data.get('label_encoders', {})
            feature_names = model_data.get('feature_names', [])

            # Encode kategorik alanlar eğitimdeki encoder'larla
            categorical_features = [
                'marka', 'seri', 'model', 'vites_turu', 'yakit_turu', 'kasatipi', 'cekis',
                'arac_durumu', 'renk', 'garanti', 'agir_hasar_kayitli', 'kapi_sayisi', 'plaka_uyruk',
                'yas_grubu', 'km_grubu', 'motor_grubu'
            ]
            user_proc = user_input.copy()
            for feat in categorical_features:
                if feat in label_encoders:
                    try:
                        user_proc[feat + '_encoded'] = label_encoders[feat].transform(user_proc[feat].astype(str))
                    except Exception:
                        user_proc[feat + '_encoded'] = 0
                else:
                    # Model bu özelliği bekliyor olabilir; yoksa 0 ile doldur
                    if feat + '_encoded' in feature_names and feat + '_encoded' not in user_proc.columns:
                        user_proc[feat + '_encoded'] = 0

            # Özellik hizalama
            if feature_names:
                X = user_proc[[c for c in feature_names if c in user_proc.columns]].copy()
            else:
                # Yedek: makul kolonlar
                base_cols = ['model_yili', 'kilometre', 'motor_hacmi', 'arac_yasi', 'yillik_ortalama_km', 'km_per_motor_hacmi', 'motorgucu', 'guc_per_hacim']
                enc_cols = [c for c in user_proc.columns if c.endswith('_encoded')]
                X = user_proc[[c for c in base_cols + enc_cols if c in user_proc.columns]].copy()

            # Tahmin
            try:
                y_pred = float(model.predict(X)[0])
                st.metric(label="🎯 Tahmini Fiyat", value=f"{y_pred:,.0f} TL")
                st.info(f"**Fiyat Aralığı:** {y_pred*0.9:,.0f} - {y_pred*1.1:,.0f} TL")
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")

            # SHAP Force Plot (tek örnek açıklaması)
            try:
                shap.initjs()
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                # Tek örnek için force plot
                fp = shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=False)
                # HTML olarak göm
                shap_html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
                st.subheader("🧠 SHAP Force Plot (Yerel Açıklama)")
                st_html(shap_html, height=300)
            except Exception as e:
                st.warning(f"SHAP force plot üretilemedi: {e}")
    
    with col2:
        st.subheader("📋 Araç Özeti")
        
        st.write(f"**Marka:** {marka}")
        st.write(f"**Model Yılı:** {model_yili}")
        st.write(f"**Araç Yaşı:** {user_input['arac_yasi'].iloc[0]} yıl")
        st.write(f"**Kilometre:** {kilometre:,} km")
        st.write(f"**Motor Hacmi:** {motor_hacmi}L")
        st.write(f"**Motor Gücü:** {motor_gucu} HP")
        st.write(f"**Yakıt Türü:** {yakit_turu}")
        st.write(f"**Vites:** {vites_turu}")
        st.write(f"**Gövde:** {govde_tipi}")
        
        # Öneriler
        st.subheader("💡 Öneriler")
        
        if user_input['arac_yasi'].iloc[0] > 10:
            st.warning("⚠️ 10 yaş üzeri araç - detaylı muayene önerilir")
        
        if kilometre > 200000:
            st.warning("⚠️ Yüksek kilometre - bakım geçmişi önemli")
        
        if motor_hacmi < 1.4:
            st.info("💡 Düşük motor hacmi - yakıt tasarrufu avantajı")
        
        if yakit_turu == "Elektrik":
            st.success("🌱 Elektrikli araç - çevre dostu seçim")

# Alt bilgi
st.markdown("---")
st.markdown("""
### 📚 Proje Hakkında

Bu uygulama, **Türkiye İkinci El Araç Fiyat Tahmini** projesi kapsamında geliştirilmiştir.

**Özellikler:**
- 🤖 Makine öğrenmesi tabanlı tahmin
- 🔍 Açıklanabilir AI (SHAP & LIME)
- 📊 Detaylı analiz ve görselleştirme
- 🇹🇷 Türkiye pazarına özgü veri

**Geliştiriciler:** Seyfullah Adıgüzel, Batuhan Orkun İnce  
**Danışman:** Sinan Keskin

---
*Not: Bu tahminler referans amaçlıdır. Gerçek fiyatlar piyasa koşullarına göre değişebilir.*
""")

# Footer
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 50px;'>
    <p>🚗 Türkiye Araç Fiyat Tahmini - Açıklanabilir AI Projesi</p>
</div>
""", unsafe_allow_html=True)
