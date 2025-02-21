import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    """
    Veriyi yükler, işler ve eğitim/test setlerine böler.
    
    Args:
        csv_path (str): Veri setinin dosya yolu.
    
    Returns:
        X_train, X_test, y_train, y_test: Eğitim ve test setleri.
    """
    df = pd.read_csv(csv_path)

    # Duygu (emotion) sütununu hedef değişken olarak al
    X = df.drop(columns=['emotion']).values
    y = df['emotion'].values

    # Kategorik etiketleri sayısal hale getir
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def train_random_forest(X_train, y_train):
    """
    Random Forest modeli oluşturur ve eğitir.
    
    Args:
        X_train (np.ndarray): Eğitim özellikleri.
        y_train (np.ndarray): Eğitim etiketleri.
    
    Returns:
        RandomForestClassifier: Eğitilmiş model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_emotion(model, X_test, y_test, label_encoder):
    """
    Model ile tahmin yapar ve doğruluk oranını hesaplar.
    
    Args:
        model: Eğitilmiş Random Forest modeli.
        X_test: Test özellikleri.
        y_test: Gerçek test etiketleri.
        label_encoder: Etiketleri geri dönüştürmek için kullanılan LabelEncoder.
    
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Doğruluk Oranı: {accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    csv_path = "combined_dataset.csv"  # Özellik çıkarımı sonrası elde edilen veri seti
    
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(csv_path)
    
    rf_model = train_random_forest(X_train, y_train)
    
    predict_emotion(rf_model, X_test, y_test, label_encoder)
