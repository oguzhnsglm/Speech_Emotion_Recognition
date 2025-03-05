# Speech_Emotion_Recognition

Açıklama

Bu proje, SAVE, TESS ve RAVDESS veri setleri üzerinde özellik çıkarımı, özellik seçimi (feature selection) ve farklı makine öğrenmesi ile derin öğrenme algoritmalarının uygulanmasını içeren bir duygu tanıma sistemidir.

Proje Adımları

1️⃣ Özellik Çıkarımı

Veri setlerindeki ses dosyalarından aşağıdaki akustik özellikler çıkarılır:

MFCC (Mel-Frequency Cepstral Coefficients)

Chroma Features

Mel-Spectrogram

Spectral Features (Centroid, Rolloff, Bandwidth, Contrast)

Zero Crossing Rate

RMS Energy

2️⃣ Özellik Seçimi (Feature Selection)

Tüm çıkarılan özellikler arasından en önemli olanlarını belirlemek için 4 farklı özellik seçme algoritması kullanılır:

Information Gain (IG)

Correlation Feature Selection (CFS)

Relief-F (REL)

Symmetrical Uncertainty (SU)

Özellik seçim süreci:

Her algoritma, en iyi 100 özelliği seçer.

Seçilmeyen 20 özellik elenir.

Her seçilen özelliğe sıralama puanı (Rank Score) verilir:

Örneğin IG yöntemi, en önemli özelliğe 100 puan, ikinciye 99 puan, ... son seçilene 1 puan verir.

Aynı işlem CFS, REL ve SU için de yapılır.

Dört algoritmanın verdiği sıralama puanları toplanır ve en yüksek skora sahip 100 özellik seçilir.

3️⃣ Makine Öğrenmesi ve Derin Öğrenme Algoritmaları

Seçilen özellikler kullanılarak ML ve DL modelleri eğitilir ve performansları karşılaştırılır:

📌 Makine Öğrenmesi Algoritmaları:

Random Forest (RF)

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

📌 Derin Öğrenme Algoritmaları:

Convolutional Neural Network (CNN)

Long Short-Term Memory (LSTM)

Gated Recurrent Unit (GRU)

4️⃣ Rank-Based Adaptive Model (RAM) Yöntemi

Bu projede RAM algoritması kullanılarak en iyi özellikler belirlenir. RAM algoritması şu mantıkla çalışır:

Tüm özellikler için feature selection yapılır ve her özelliğe önem puanı atanır.

En yüksek puana sahip 100 özellik sıralanarak seçilir.

Bu özelliklerle ML ve DL algoritmaları çalıştırılır.
