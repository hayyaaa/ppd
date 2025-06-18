import streamlit as st
import pandas as pd
import joblib

# Inisialisasi state untuk setiap input
if 'pregnancies' not in st.session_state:
    st.session_state.pregnancies = 1
if 'glucose' not in st.session_state:
    st.session_state.glucose = 100
if 'blood_pressure' not in st.session_state:
    st.session_state.blood_pressure = 72
if 'skin_thickness' not in st.session_state:
    st.session_state.skin_thickness = 35
if 'insulin' not in st.session_state:
    st.session_state.insulin = 125
if 'bmi' not in st.session_state:
    st.session_state.bmi = 33.6
if 'dpf' not in st.session_state:
    st.session_state.dpf = 0.627
if 'age' not in st.session_state:
    st.session_state.age = 50

# Judul aplikasi
st.title('🩺 Prediksi Risiko Diabetes')
st.markdown('Aplikasi ini memprediksi risiko diabetes berdasarkan karakteristik medis pasien menggunakan model machine learning.')
st.image('https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?auto=format&fit=crop&w=1200', 
         caption='Kesehatan Diabetes - Sumber: Unsplash', width=600)

# Muat model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model_final.pkl')

model = load_model()
st.success("Model 'diabetes_model_final.pkl' berhasil dimuat!")

# Parameter threshold
OPTIMAL_THRESHOLD = st.slider(
    'Threshold Klasifikasi', 
    min_value=0.1, 
    max_value=0.9, 
    value=0.532, 
    step=0.01,
    help='Nilai ambang batas untuk menentukan prediksi diabetes (default: 0.53)'
)

# Input data pasien
st.header('📋 Data Pasien')
st.markdown('Masukkan data medis pasien:')

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input('Kehamilan (Pregnancies)', 0, 20, st.session_state.pregnancies)
    glucose = st.number_input('Glukosa (mg/dL)', 0, 300, st.session_state.glucose)
    blood_pressure = st.number_input('Tekanan Darah (mmHg)', 0, 200, st.session_state.blood_pressure)
    skin_thickness = st.number_input('Ketebalan Kulit (mm)', 0, 100, st.session_state.skin_thickness)

with col2:
    insulin = st.number_input('Insulin (μU/mL)', 0, 1000, st.session_state.insulin)
    bmi = st.number_input('BMI (kg/m²)', 10.0, 60.0, st.session_state.bmi, step=0.1)
    dpf = st.number_input('Riwayat Diabetes Keluarga', 0.0, 2.0, st.session_state.dpf, step=0.001)
    age = st.number_input('Usia', 0, 120, st.session_state.age)

# Hitung fitur rekayasa
glucose_bmi = (glucose * bmi) / 100
age_insulin = (age * insulin) / 100
bp_glucose_ratio = blood_pressure / glucose if glucose > 0 else 0

# Tombol prediksi
if st.button('Prediksi Risiko Diabetes', type='primary'):
    # Buat dataframe dari input
    patient_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age],
        'Glucose_BMI': [glucose_bmi],
        'Age_Insulin': [age_insulin],
        'BP_Glucose_Ratio': [bp_glucose_ratio]
    })
    
    st.divider()
    st.subheader('Hasil Prediksi')
    
    # Tampilkan data pasien
    st.markdown('**Data Pasien:**')
    st.dataframe(patient_data.style.format("{:.2f}"), hide_index=True)
    
    # Prediksi probabilitas
    prediction_proba = model.predict_proba(patient_data)
    risk_probability = prediction_proba[0][1]
    
    # Tampilkan hasil prediksi
    prob_col, result_col = st.columns(2)
    
    with prob_col:
        st.metric('Probabilitas Diabetes', f"{risk_probability*100:.2f}%")
        st.progress(int(risk_probability*100))
        
    with result_col:
        final_prediction = 1 if risk_probability >= OPTIMAL_THRESHOLD else 0
        if final_prediction == 1:
            st.error(f'**PREDIKSI: DIABETES** (Risiko > {OPTIMAL_THRESHOLD*100:.0f}%)')
        else:
            st.success(f'**PREDIKSI: TIDAK DIABETES** (Risiko < {OPTIMAL_THRESHOLD*100:.0f}%)')
    
    # Interpretasi klinis
    st.subheader('Interpretasi Klinis')
    
    if risk_probability >= 0.6:
        st.error('🟢 **RISIKO SANGAT TINGGI**')
        st.markdown('''
        - Pasien memiliki risiko diabetes sangat tinggi
        - Segera konsultasi dengan dokter spesialis
        - Lakukan pemeriksaan HbA1c dan tes toleransi glukosa
        - Perubahan gaya hidup mendesak diperlukan
        ''')
    elif risk_probability >= 0.45:
        st.warning('🟡 **RISIKO TINGGI**')
        st.markdown('''
        - Pasien berisiko tinggi terkena diabetes
        - Disarankan konsultasi dokter umum
        - Lakukan pemantauan glukosa rutin
        - Mulai perubahan pola makan dan aktivitas fisik
        ''')
    elif risk_probability >= 0.3:
        st.info('🟠 **RISIKO MODERAT**')
        st.markdown('''
        - Pasien memiliki risiko diabetes sedang
        - Perubahan gaya hidup preventif disarankan
        - Lakukan pemeriksaan kesehatan tahunan
        - Pantau kadar glukosa secara berkala
        ''')
    else:
        st.success('🔵 **RISIKO RENDAH**')
        st.markdown('''
        - Pasien memiliki risiko diabetes rendah
        - Pertahankan gaya hidup sehat
        - Lakukan aktivitas fisik teratur
        - Cek kesehatan rutin setiap tahun
        ''')

# Contoh data pasien
st.divider()
st.subheader('Contoh Data Pasien')

col1, col2 = st.columns(2)
with col1:
    if st.button('Pasien Berisiko Rendah', type='secondary'):
        st.session_state.pregnancies = 2
        st.session_state.glucose = 92
        st.session_state.blood_pressure = 74
        st.session_state.skin_thickness = 28
        st.session_state.insulin = 85
        st.session_state.bmi = 22.1
        st.session_state.dpf = 0.15
        st.session_state.age = 28
        st.rerun()

with col2:
    if st.button('Pasien Berisiko Tinggi', type='secondary'):
        st.session_state.pregnancies = 1
        st.session_state.glucose = 140
        st.session_state.blood_pressure = 72
        st.session_state.skin_thickness = 35
        st.session_state.insulin = 125
        st.session_state.bmi = 33.6
        st.session_state.dpf = 0.627
        st.session_state.age = 50
        st.rerun()

# Footer
st.divider()
st.caption('© 2025 Sistem Prediksi Diabetes - Model berbasis Random Forest')
st.caption('Catatan: Prediksi ini tidak menggantikan diagnosis medis profesional. Konsultasikan dengan dokter untuk penilaian lengkap.')
