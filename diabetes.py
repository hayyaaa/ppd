try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import joblib
import sys

# Calculate engineered features
def calculate_features(data: dict):
    """
    Menghitung fitur rekayasa tambahan:
    - Glucose_BMI: (Glucose * BMI) / 100
    - Age_Insulin: (Age * Insulin) / 100
    - BP_Glucose_Ratio: BloodPressure / Glucose (jika Glucose > 0)
    """
    glucose = data.get('Glucose', 0)
    bmi = data.get('BMI', 0)
    age = data.get('Age', 0)
    insulin = data.get('Insulin', 0)
    blood_pressure = data.get('BloodPressure', 0)

    glucose_bmi = (glucose * bmi) / 100 if glucose is not None and bmi is not None else 0
    age_insulin = (age * insulin) / 100 if age is not None and insulin is not None else 0
    bp_glucose_ratio = blood_pressure / glucose if glucose and glucose > 0 else 0
    return glucose_bmi, age_insulin, bp_glucose_ratio

if STREAMLIT_AVAILABLE:
    # --------------------------------------------------
    # Bagian Aplikasi Streamlit
    # --------------------------------------------------
    # Page configuration
    st.set_page_config(
        page_title="Prediksi Risiko Diabetes", layout="wide", initial_sidebar_state="collapsed"
    )

    # Inject custom CSS for modern design
    def inject_css():
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap" rel="stylesheet">
            <style>
            :root {
              --primary-color: #4A90E2;
              --secondary-color: #50E3C2;
              --bg-color: #F5F7FA;
              --text-color: #333333;
              --card-bg: #FFFFFF;
              --font-family: 'Roboto', sans-serif;
            }
            html, body, .stApp {
              background-color: var(--bg-color) !important;
              color: var(--text-color) !important;
              font-family: var(--font-family) !important;
            }
            .card {
              background-color: var(--card-bg) !important;
              box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
              border-radius: 12px !important;
              padding: 1.5rem !important;
              margin-bottom: 2rem !important;
            }
            .stButton>button {
              background-color: var(--primary-color) !important;
              color: #fff !important;
              border-radius: 8px !important;
              padding: 0.5rem 1.5rem !important;
              transition: background-color 0.3s ease, transform 0.2s ease;
              font-weight: 500 !important;
              font-size: 1rem !important;
            }
            .stButton>button:hover {
              background-color: var(--secondary-color) !important;
              transform: translateY(-2px) !important;
            }
            @media (max-width: 768px) {
              .card {
                padding: 1rem !important;
              }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @st.cache_resource
    def load_model():
        return joblib.load('diabetes_model_final.pkl')

    def main():
        inject_css()

        # Inisialisasi session_state
        defaults = {
            'pregnancies': 1,
            'glucose': 100,
            'blood_pressure': 72,
            'skin_thickness': 35,
            'insulin': 125,
            'bmi': 33.6,
            'dpf': 0.627,
            'age': 50
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # Judul dan header
        st.title('🩺 Prediksi Risiko Diabetes')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('Aplikasi prediksi risiko diabetes menggunakan model Random Forest.')
        st.image(
            'https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?auto=format&fit=crop&w=1200',
            caption='Kesehatan Diabetes', use_column_width='always'
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Load model
        model = load_model()
        st.sidebar.success("Model berhasil dimuat!")

        # Slider threshold di sidebar
        threshold = st.sidebar.slider('Threshold Klasifikasi', 0.1, 0.9, 0.532, 0.01)

        # Input Data Pasien
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header('📋 Data Pasien')
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input('Kehamilan', 0, 20, st.session_state.pregnancies)
            glucose = st.number_input('Glukosa (mg/dL)', 0, 300, st.session_state.glucose)
            blood_pressure = st.number_input('Tekanan Darah (mmHg)', 0, 200, st.session_state.blood_pressure)
        with col2:
            skin_thickness = st.number_input('Ketebalan Kulit (mm)', 0, 100, st.session_state.skin_thickness)
            insulin = st.number_input('Insulin (μU/mL)', 0, 1000, st.session_state.insulin)
            bmi = st.number_input('BMI (kg/m²)', 10.0, 60.0, st.session_state.bmi, step=0.1)
        with col3:
            dpf = st.number_input('Riwayat Diabetes Keluarga', 0.0, 2.0, st.session_state.dpf, step=0.001)
            age = st.number_input('Usia', 0, 120, st.session_state.age)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediksi
        if st.button('Prediksi Risiko Diabetes'):
            data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            g_bmi, a_ins, bp_gr = calculate_features(data)
            df = pd.DataFrame({**data,
                                'Glucose_BMI': g_bmi,
                                'Age_Insulin': a_ins,
                                'BP_Glucose_Ratio': bp_gr
            }, index=[0])

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader('Hasil Prediksi')
            st.markdown('**Data Pasien:**')
            st.dataframe(df.style.format("{:.2f}"), hide_index=True)

            proba = model.predict_proba(df)[0][1]
            c1, c2 = st.columns(2)
            with c1:
                st.metric('Probabilitas Diabetes', f"{proba*100:.2f}%")
                st.progress(int(proba*100))
            with c2:
                if proba >= threshold:
                    st.error('**PREDIKSI: DIABETES**')
                else:
                    st.success('**PREDIKSI: TIDAK DIABETES**')

            st.subheader('Interpretasi Klinis')
            if proba >= 0.6:
                st.error('🟢 RISIKO SANGAT TINGGI')
            elif proba >= 0.45:
                st.warning('🟡 RISIKO TINGGI')
            elif proba >= 0.3:
                st.info('🟠 RISIKO SEDANG')
            else:
                st.success('🔵 RISIKO RENDAH')
            st.markdown('</div>', unsafe_allow_html=True)

        # Preset contoh data
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Contoh Data Pasien')
        e1, e2 = st.columns(2)
        with e1:
            if st.button('Risiko Rendah', key='low'):
                st.session_state.pregnancies, st.session_state.glucose, st.session_state.blood_pressure = 2, 92, 74
                st.session_state.skin_thickness, st.session_state.insulin = 28, 85
                st.session_state.bmi, st.session_state.dpf, st.session_state.age = 22.1, 0.15, 28
                st.experimental_rerun()
        with e2:
            if st.button('Risiko Tinggi', key='high'):
                st.session_state.pregnancies, st.session_state.glucose, st.session_state.blood_pressure = 1, 140, 72
                st.session_state.skin_thickness, st.session_state.insulin = 35, 125
                st.session_state.bmi, st.session_state.dpf, st.session_state.age = 33.6, 0.627, 50
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.caption('© 2025 Sistem Prediksi Diabetes - Model berbasis Random Forest')
        st.caption('Catatan: Prediksi ini tidak menggantikan diagnosis medis profesional')

    if __name__ == "__main__":
        main()
else:
    # --------------------------------------------------
    # Fallback & Tests
    # --------------------------------------------------
    def main():
        print("Streamlit tidak terinstal. Silakan install menggunakan 'pip install streamlit'.")

    def _test_calculate_features():
        print("Menjalankan test calculate_features...")
        data = {'Glucose': 100, 'BMI': 20, 'Age': 50, 'Insulin': 100, 'BloodPressure': 80}
        gb, ai, bgr = calculate_features(data)
        assert gb == 20.0, f"Expected 20.0, got {gb}"
        assert ai == 50.0, f"Expected 50.0, got {ai}"
        assert abs(bgr - 0.8) < 1e-6, f"Expected ~0.8, got {bgr}"
        print("_test_calculate_features passed.")

    def _test_empty_data():
        print("Menjalankan test_empty_data...")
        gb, ai, bgr = calculate_features({})
        assert gb == 0, f"Expected 0, got {gb}"
        assert ai == 0, f"Expected 0, got {ai}"
        assert bgr == 0, f"Expected 0, got {bgr}"
        print("_test_empty_data passed.")

    if __name__ == "__main__":
        main()
        _test_calculate_features()
        _test_empty_data()