import streamlit as st
import joblib
import pandas as pd
import numpy as np

package = joblib.load("deployment_model/clustering_deployment.joblib")
model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means (RFM)")

recency = st.number_input("Recency (hari sejak belanja terakhir)", min_value=0, value=18)
frequency = st.number_input("Frequency (jumlah transaksi)", min_value=1, value=7)
monetary = st.number_input("Monetary (total belanja, £)", min_value=1.0, value=2570.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)],
    })
    cluster = model.predict(scaler.transform(data[feature_columns]))[0]
    st.success(f"Pelanggan masuk ke Cluster {cluster}")
