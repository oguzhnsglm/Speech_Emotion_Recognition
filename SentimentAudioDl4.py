import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(csv_path):
    """ Veriyi yükler, işler ve eğitim/test setlerine böler. """
    df = pd.read_csv(csv_path)
    features = df.drop(columns=['emotion']).values
    labels = df['emotion'].values
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    labels_one_hot = one_hot_encoder.fit_transform(labels_encoded.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train_3d, X_test_3d, y_train, y_test

def build_cnn_model(input_shape, num_classes):
    """ CNN modelini oluşturur ve derler. """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])    
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """ Modeli eğitir ve test başarımını döndürür. """
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

def main():
    """ Ana program akışı. """
    csv_path = 'selected_features.csv'
    X_train_3d, X_test_3d, y_train, y_test = load_and_preprocess_data(csv_path)
    
    input_shape = (X_train_3d.shape[1], 1)
    num_classes = y_train.shape[1]
    
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_accuracy = train_and_evaluate_model(cnn_model, X_train_3d, y_train, X_test_3d, y_test, "CNN")
    
    print("\nModel Accuracy Comparison:")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")

if __name__ == "__main__":
    main()
