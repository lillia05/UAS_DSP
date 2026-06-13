from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model Kopi
try:
    # PASTIKAN NAMA FILE .pkl DI FOLDER model/ SUDAH SESUAI
    model = joblib.load('model/rf_coffee_model.pkl')
    print("Model Kopi berhasil dimuat!")
except Exception as e:
    model = None
    print(f"Error memuat model: {e}")

@app.route('/')
def index():
    return redirect(url_for('get_dashboard'))

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    prediction_result = None
    if request.method == 'POST':
        if model:
            try:
                # 1. Ambil data fisik & sensorik dari form HTML
                country = float(request.form.get('country', 0))
                processing = float(request.form.get('processing', 0))
                altitude = float(request.form.get('altitude', 0))
                cat_one = float(request.form.get('cat_one', 0))
                cat_two = float(request.form.get('cat_two', 0))
                moisture = float(request.form.get('moisture', 0))
                aroma = float(request.form.get('aroma', 0))
                flavor = float(request.form.get('flavor', 0))
                aftertaste = float(request.form.get('aftertaste', 0))
                acidity = float(request.form.get('acidity', 0))
                body = float(request.form.get('body', 0))
                balance = float(request.form.get('balance', 0))

                # 2. Susun menjadi array untuk ditebak oleh model
                input_data = np.array([[country, processing, altitude, cat_one, cat_two, moisture, 
                                        aroma, flavor, aftertaste, acidity, body, balance]])
                
                # 3. Prediksi
                pred = model.predict(input_data)[0]
                
                if pred == 1:
                    # Teks diubah agar langsung menunjuk ke nama Grade
                    prediction_result = "Specialty Grade 🌟"
                else:
                    prediction_result = "Commercial Grade ☕"
                    
            except Exception as e:
                prediction_result = f"Terjadi error saat prediksi: {e}"
        else:
            prediction_result = "Error: Model tidak ditemukan!"
            
    return render_template('form_prediction.html', prediction=prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)