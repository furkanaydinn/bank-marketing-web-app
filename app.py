import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Kaydedilmiş modeli, ölçekleyiciyi, özellik listesini ve giriş seçeneklerini yükle
try:
    model_pipeline = joblib.load('bank_deposit_model_selected_features.joblib')
    selected_features_list = joblib.load('selected_features_list.joblib')
    input_options = joblib.load('input_options.joblib')
    original_feature_order = joblib.load('original_feature_order.joblib')
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı! Lütfen önce `train_and_save_model.py` script'ini çalıştırın.")
    st.stop()


st.set_page_config(page_title="Banka Vadeli Mevduat Tahmini", layout="wide")
st.title("🏦 Banka Vadeli Mevduat Hesabı Tahmin Uygulaması")
st.markdown("""
Bu uygulama, bir müşterinin bankanın vadeli mevduat hesabına abone olup olmayacağını tahmin eder.
Lütfen aşağıdaki müşteri bilgilerini girin:
""")

# Kullanıcıdan girdi alma
# Streamlit'te daha dinamik bir form oluşturmak için input_options'ı kullanacağız.
# Formu iki sütuna bölebiliriz
col1, col2 = st.columns(2)
user_inputs = {}

# Orijinal df'teki sütun sırasına göre gidelim
# 'duration' gibi sızıntı yapan veya kampanya sonrası bilinen özellikleri hariç tutmamız gerekebilir.
# Bu örnekte orijinal veri setindeki tüm özellikler için girdi alacağız.
# Gerçek bir deploy'da 'duration' gibi özellikler çıkarılmalıdır.
# Eğer 'duration' en önemli özellikler arasındaysa, bu modelin pratik kullanımını sorgulatır.
# Eğitim scriptinde `duration`'ın önemini kontrol edin. Eğer yüksekse, dikkatli olunmalı.
# Bu örnekte, `duration`'ı dahil ediyoruz ancak gerçek senaryoda çıkarılabilir.

potential_leakage_features = ['duration'] # Bu özellikler genelde kampanya sonrası bilinir
st.sidebar.warning(f"**Not:** `{', '.join(potential_leakage_features)}` gibi özellikler genellikle kampanya sonrası bilindiğinden, bu demo amaçlı bir uygulamadır. Gerçek bir senaryoda bu özellikler tahmin aşamasında bilinmeyebilir.")


# Girdileri iki sütuna dağıtma
half_way = len(original_feature_order) // 2
features_col1 = original_feature_order[:half_way]
features_col2 = original_feature_order[half_way:]

with col1:
    st.subheader("Müşteri Bilgileri 1")
    for feature in features_col1:
        if feature in input_options:
            if isinstance(input_options[feature], list): # Kategorik
                user_inputs[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", options=input_options[feature], key=f"col1_{feature}")
            elif isinstance(input_options[feature], tuple): # Sayısal
                min_val, max_val, mean_val = input_options[feature]
                user_inputs[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=float(min_val), max_value=float(max_val), value=float(mean_val), step=1.0 if (max_val-min_val)>10 else 0.1, key=f"col1_{feature}")
        else:
            # Bu durum olmamalı eğer input_options doğru oluşturulduysa
            user_inputs[feature] = st.text_input(f"Enter {feature}", key=f"col1_{feature}")


with col2:
    st.subheader("Müşteri Bilgileri 2")
    for feature in features_col2:
        if feature in input_options:
            if isinstance(input_options[feature], list): # Kategorik
                user_inputs[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", options=input_options[feature], key=f"col2_{feature}")
            elif isinstance(input_options[feature], tuple): # Sayısal
                min_val, max_val, mean_val = input_options[feature]
                user_inputs[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=float(min_val), max_value=float(max_val), value=float(mean_val), step=1.0 if (max_val-min_val)>10 else 0.1, key=f"col2_{feature}")
        else:
            user_inputs[feature] = st.text_input(f"Enter {feature}", key=f"col2_{feature}")


# Tahmin butonu
if st.button("Tahmin Yap", key="predict_button"):
    # Kullanıcı girdilerini DataFrame'e dönüştürme
    input_df_original_features = pd.DataFrame([user_inputs])
    
    # One-hot encoding uygulama (eğitimdeki gibi)
    # Kategorik sütunları belirleyelim (input_options'tan yararlanabiliriz)
    categorical_to_encode = [col for col, opt in input_options.items() if isinstance(opt, list)]

    # --- YENİ EKLENEN BÖLÜM BAŞLANGICI ---
    # Kategorik sütunların veri tipini, eğitimdeki tüm kategorileri içerecek şekilde ayarla
    for col_name in categorical_to_encode:
        if col_name in input_df_original_features.columns:
            all_categories_for_col = input_options[col_name] # input_options'dan kategorileri al
            try:
                input_df_original_features[col_name] = pd.Categorical(
                    input_df_original_features[col_name],
                    categories=all_categories_for_col
                )
            except ValueError as ve:
                st.error(f"'{col_name}' sütunu için kategori ayarlarken hata: {ve}. Sağlanan değer: {input_df_original_features[col_name].iloc[0]}, Beklenen kategoriler: {all_categories_for_col}")
                st.stop() # Hata durumunda devam etme
    # --- YENİ EKLENEN BÖLÜM SONU ---
    
    try:
        input_df_processed = pd.get_dummies(input_df_original_features.copy(), columns=categorical_to_encode, drop_first=True)
    except Exception as e:
        st.error(f"One-hot encoding sırasında hata: {e}")
        st.error(f"Gelen kategorik sütunlar: {categorical_to_encode}")
        st.error(f"Input DataFrame sütunları: {input_df_original_features.columns.tolist()}")
        st.stop()

    # Modelin beklediği tüm one-hot encode edilmiş özelliklerin olduğundan emin olma
    # ve doğru sırada olduğundan emin olma. Eksik olanları 0 ile doldur.
    # ÖNEMLİ: `selected_features_list` eğitilmiş modelin kullandığı son özellik listesidir.
    # Model pipeline'ı (`bank_deposit_model_selected_features.joblib`) zaten ölçekleyiciyi içeriyor
    # ve sadece `selected_features_list`'teki özellikleri bekliyor.
    
    # Önce tüm olası işlenmiş özellikleri (eğitimde kullanılan) oluşturalım
    # Bu, train_and_save_model.py'deki `feature_names_processed` olmalıydı.
    # Şimdilik, en güvenli yol, `selected_features_list`'teki özelliklerin
    # `input_df_processed` içinde olmasını sağlamak.
    
    final_input_df = pd.DataFrame(columns=selected_features_list) # Boş bir df, doğru sütunlarla
    
    for feature in selected_features_list:
        if feature in input_df_processed.columns:
            final_input_df[feature] = input_df_processed[feature]
        else:
            # Bu, one-hot encoding sonucu oluşmayan bir sütunsa (örn: drop_first=True nedeniyle baz kategori)
            # veya orijinalde olmayan bir özellikse 0 ile doldurulur.
            final_input_df[feature] = 0 
            st.warning(f"'{feature}' özelliği girdide bulunamadı ve 0 olarak ayarlandı. Bu, one-hot encoding'den kaynaklanıyor olabilir.")

    # Sütunların sırasının modelin beklediği gibi olduğundan emin olalım (gerçi pipeline bunu halletmeli)
    final_input_df = final_input_df[selected_features_list]

    missing_in_processed_but_selected = [] # Hangi özelliklerin input_df_processed'de olmadığı ama selected_features_list'te olduğu

    for feature in selected_features_list:
        if feature in input_df_processed.columns:
            final_input_df[feature] = input_df_processed[feature]
        else:
            final_input_df[feature] = 0 
            # Bu uyarı hala görünebilir EĞER kullanıcı, seçilen özellik listesindeki
            # bir one-hot encoded değere karşılık GELMEYEN bir kategori seçtiyse.
            # Örneğin, selected_features_list'te 'month_mar' var ama kullanıcı 'month_jan' seçti.
            # Bu durumda 'month_mar' girdide bulunamaz ve 0 olması doğrudur.
            # Önemli olan, kullanıcının seçtiği değere karşılık gelen one-hot encoded sütunun
            # (örn: 'month_jan') input_df_processed'te oluşması ve final_input_df'e aktarılmasıdır.
            missing_in_processed_but_selected.append(feature) # Sadece listeye ekle, uyarıyı sonda verelim

    if missing_in_processed_but_selected:
        st.warning(f"Aşağıdaki önemli özellikler, girdilerinizden doğrudan oluşturulamadı ve model için 0 olarak ayarlandı (bu, farklı bir kategori seçmenizden veya one-hot encoding yapısından kaynaklanabilir): {', '.join(missing_in_processed_but_selected)}")

    # Tahmin yapma
    try:
        prediction_proba = model_pipeline.predict_proba(final_input_df)[0] # İlk (ve tek) örnek için olasılıklar
        prediction = model_pipeline.predict(final_input_df)[0] # Sınıf tahmini
        
        st.subheader("Tahmin Sonuçları")
        if prediction == 1:
            st.success(f"Bu müşterinin vadeli mevduat hesabı açma olasılığı: **{prediction_proba[1]*100:.2f}%** 🎉")
            st.balloons()
        else:
            st.info(f"Bu müşterinin vadeli mevduat hesabı açma olasılığı: **{prediction_proba[1]*100:.2f}%**")
            st.markdown("Model, bu müşterinin **vadeli hesap açmayacağını** tahmin ediyor.")
            
        # Olasılıkları göster
        # st.write("Sınıf Olasılıkları:", prediction_proba) # Hata ayıklama için

    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
        st.error(f"Modelin beklediği özellik sayısı: {len(selected_features_list)}")
        st.error(f"Sağlanan özellik sayısı: {final_input_df.shape[1]}")
        st.error(f"Sağlanan özellikler: {final_input_df.columns.tolist()}")
        # st.dataframe(final_input_df) # Hata ayıklama için
