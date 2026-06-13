import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib  # <-- Ini yang bikin web bisa baca modelnya!
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def run_rf_model_mlflow(df):
    # PERBAIKAN: Akhiran diubah dari .git menjadi .mlflow
    uri_dagshub = "https://dagshub.com/lillia05/PrediksiKualitas-Kopi-Arabika.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = "lillia05"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "db52c1460f54bc394b192cde0cff103f714d8706"

    experiment_name = "coffee_quality_prediction"
    client = mlflow.client.MlflowClient()

    try:
        experiment_id = client.create_experiment(name=experiment_name)
        print(f"Eksperimen '{experiment_name}' berhasil dibuat dengan ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Eksperimen '{experiment_name}' sudah ada dengan ID: {experiment_id}")

    print("Mempersiapkan data kopi...")
    # Membuang baris yang tidak memiliki skor kualitas
    df = df.dropna(subset=['Total.Cup.Points'])
    
    # Membuat Target (Label): Jika skor >= 80 maka Specialty (1), selain itu Commercial (0)
    df['Specialty_Grade'] = (df['Total.Cup.Points'] >= 80).astype(int)
    
    # 12 Faktor yang sesuai dengan input di Form HTML Web
    fitur_form = [
        'Country.of.Origin', 'Processing.Method', 'altitude_mean_meters', 
        'Category.One.Defects', 'Category.Two.Defects', 'Moisture',
        'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance'
    ]
    
    # Membersihkan NaN khusus di kolom yang mau dipakai
    df_model = df[fitur_form + ['Specialty_Grade']].dropna()

    y = df_model['Specialty_Grade']
    X = df_model[fitur_form].copy()
    
    print("Mengubah data teks (Negara & Proses) menjadi angka numerik...")
    le = LabelEncoder()
    if X['Country.of.Origin'].dtype == 'object':
        X['Country.of.Origin'] = le.fit_transform(X['Country.of.Origin'].astype(str))
    if X['Processing.Method'].dtype == 'object':
        X['Processing.Method'] = le.fit_transform(X['Processing.Method'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Nyalakan autolog agar training metrics muncul di DagsHub
    mlflow.sklearn.autolog() 

    with mlflow.start_run(run_name="rf-coffee-model", experiment_id=experiment_id) as run:
        # Menggunakan class_weight='balanced' agar prediksi adil
        model_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)

        # Metrik Uji (Testing)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1_score)

        # Mengirim Nama Model ke DagsHub
        model_signature = infer_signature(model_input=X_train, model_output=y_train)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                registered_model_name="rf_model_coffee", # <-- Nama Model Kopi
                signature=model_signature,
                input_example=X_train.head(1)
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                signature=model_signature,
                input_example=X_train.head(1)
            )

        # Simpan Model Lokal untuk Website
        os.makedirs("model", exist_ok=True)
        joblib.dump(model_rf, "model/rf_coffee_model.pkl")
        print("\n=> Model berhasil disimpan ke: model/rf_coffee_model.pkl")

        print("\n--- Proses Selesai ---")
        print("Silakan cek menu Experiments dan Models di DagsHub Anda!")

if __name__ == "__main__":
    # Ganti dengan nama file data kopimu yang ada di folder data/
    dataset_path = "data/Data_Kopi_Cleaned.csv"

    if os.path.exists(dataset_path):
        print("Memuat dataset lokal...")
        df = pd.read_csv(dataset_path)
        run_rf_model_mlflow(df)
    else:
        # Otomatis mengambil data asli dari internet jika file lokal tidak ditemukan
        print("Dataset lokal tidak ditemukan. Mendownload data kopi...")
        df = pd.read_csv("https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv")
        run_rf_model_mlflow(df)