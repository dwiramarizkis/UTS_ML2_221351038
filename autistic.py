import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load label encoder untuk target (y) dan fitur (x)
label_encoder_y = joblib.load('label_encoder_y.pkl')
le_gender = joblib.load('label_encoder_gender.pkl')
le_ethnicity = joblib.load('label_encoder_ethnicity.pkl')
le_jundice = joblib.load('label_encoder_jundice.pkl')
le_austim = joblib.load('label_encoder_austim.pkl')
le_country = joblib.load('label_encoder_contry_of_res.pkl')
le_used_app = joblib.load('label_encoder_used_app_before.pkl')
le_relation = joblib.load('label_encoder_relation.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="autism-screening-on-adults.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Deteksi ASD pada Orang Dewasa")
st.write("Masukkan data berikut untuk memprediksi kemungkinan ASD.")

# Input pengguna
A1 = st.number_input("A1 Score", 0, 1, 1)
A2 = st.number_input("A2 Score", 0, 1, 1)
A3 = st.number_input("A3 Score", 0, 1, 1)
A4 = st.number_input("A4 Score", 0, 1, 1)
A5 = st.number_input("A5 Score", 0, 1, 1)
A6 = st.number_input("A6 Score", 0, 1, 1)
A7 = st.number_input("A7 Score", 0, 1, 1)
A8 = st.number_input("A8 Score", 0, 1, 1)
A9 = st.number_input("A9 Score", 0, 1, 1)
A10 = st.number_input("A10 Score", 0, 1, 1)

age = st.slider("Usia", 1, 100, 25)
gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
ethnicity = st.selectbox("Etnis", le_ethnicity.classes_)
jundice = st.selectbox("Penyakit Kuning?", le_jundice.classes_)
austim = st.selectbox("Riwayat ASD keluarga?", le_austim.classes_)
country = st.selectbox("Negara", le_country.classes_)
used_app = st.selectbox("Pernah pakai aplikasi ASD sebelumnya?", le_used_app.classes_)
result = st.slider("Hasil Tes (0.0 - 10.0)", 0.0, 10.0, 5.0)
relation = st.selectbox("Relasi pengisi formulir", le_relation.classes_)

# Encode fitur kategorikal
gender_enc = le_gender.transform([gender])[0]
ethnicity_enc = le_ethnicity.transform([ethnicity])[0]
jundice_enc = le_jundice.transform([jundice])[0]
austim_enc = le_austim.transform([austim])[0]
country_enc = le_country.transform([country])[0]
used_app_enc = le_used_app.transform([used_app])[0]
relation_enc = le_relation.transform([relation])[0]

# Gabungkan semua input
input_data = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
                        age, gender_enc, ethnicity_enc,
                        jundice_enc, austim_enc, country_enc,
                        used_app_enc,result, relation_enc]])

if st.button("Prediksi ASD"):
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    predicted_class = 1 if prediction >= 0.5 else 0
    result_label = label_encoder_y.inverse_transform([predicted_class])[0]

    st.success(f"Hasil Prediksi: **{result_label.upper()}** (Probabilitas: {prediction:.2f})")
