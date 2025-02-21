import os
import pandas as pd
import librosa
import numpy as np

def extract_features(file_path):
    """
    Verilen ses dosyasından özellik çıkarır.

    Args:
        file_path (str): Ses dosyasının yolu.

    Returns:
        np.ndarray: Özellikler içeren numpy dizisi.
    """
    try:
        # Ses dosyasını yükle
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Zero crossing rate
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

        # Spectral özellikler
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(spectral_contrast, axis=1)  # Her bir bandın ortalaması
        contrast_std = np.std(spectral_contrast, axis=1)   # Her bir bandın standart sapması

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)  # Her bir chroma kanalının ortalaması
        chroma_stft_std = np.std(chroma_stft, axis=1)    # Her bir chroma kanalının standart sapması

        # RMS enerji
        rms_mean = np.mean(librosa.feature.rms(y=audio))

        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        melspectrogram_mean = np.mean(mel_spectrogram)
        melspectrogram_std = np.std(mel_spectrogram)

        # Flatness (Düzlük)
        flatness_mean = np.mean(librosa.feature.spectral_flatness(y=audio))

        # Polynomyal özellikler
        poly_features = librosa.feature.poly_features(y=audio, sr=sample_rate, order=1)
        poly_mean = np.mean(poly_features, axis=1)

        # MFCC özellikleri
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)  # Her bir MFCC'nin ortalaması
        mfcc_std = np.std(mfcc, axis=1)   # Her bir MFCC'nin standart sapması

        # Enerji (toplam enerji hesaplama)
        energy = np.sum(audio ** 2)

        # Özellikleri birleştir
        features = np.hstack([
            zero_crossing, spectral_centroid, spectral_rolloff, spectral_bandwidth,
            contrast_mean, contrast_std, chroma_stft_mean, chroma_stft_std,
            rms_mean, melspectrogram_mean, melspectrogram_std, flatness_mean,
            poly_mean, mfcc_mean, mfcc_std, energy
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_features(input_csv, output_csv):
    """
    CREMA-D veri seti üzerinde özellik çıkarımı yapar ve bir CSV dosyasına kaydeder.

    Args:
        input_csv (str): CREMA-D dataset CSV dosyasının yolu.
        output_csv (str): Özellikler için oluşturulacak CSV dosyasının yolu.
    """
    crema_df = pd.read_csv(input_csv)

    features_list = []
    for index, row in crema_df.iterrows():
        file_path = row['Path']
        emotion = row['Emotions']
        features = extract_features(file_path)
        if features is not None:
            features_list.append([*features, emotion])

    # Özellikleri DataFrame'e çevirme
    columns = (
        ['zero_crossing', 'centroid_mean', 'rolloff_mean', 'bandwidth_mean'] +
        [f'contrast_mean_{i}' for i in range(7)] +
        [f'contrast_std_{i}' for i in range(7)] +
        [f'chroma_stft_mean_{i}' for i in range(12)] +
        [f'chroma_stft_std_{i}' for i in range(12)] +
        ['rms_mean', 'melspectrogram_mean', 'melspectrogram_std', 'flatness_mean'] +
        [f'poly_mean_{i}' for i in range(2)] +
        [f'mfcc_mean_{i}' for i in range(40)] +
        [f'mfcc_std_{i}' for i in range(40)] +
        ['energy', 'emotion']
    )
    
    # Özellik listesiyle DataFrame oluştur
    features_df = pd.DataFrame(features_list, columns=columns)

    # Özellik DataFrame'ini kaydetme
    features_df.to_csv(output_csv, index=False)
    print(f"Özellik çıkarımı tamamlandı ve '{output_csv}' dosyasına kaydedildi!")

if __name__ == "__main__":
    input_csv = "ravdess_emotions.csv"
    process_features(input_csv, "ravdess_new_features_extended.csv")