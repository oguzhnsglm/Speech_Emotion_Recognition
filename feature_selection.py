import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mutual_info_score

# Information Gain (IG) için fonksiyon
def information_gain(X, y):
    scores = []
    for i in range(X.shape[1]):
        mutual_info = mutual_info_score(y, X[:, i])
        scores.append(mutual_info)
    return np.array(scores)

# Simetrik belirsizlik için fonksiyon
def symmetrical_uncertainty(X, y):
    scores = []
    for i in range(X.shape[1]):
        hx = mutual_info_score(X[:, i], X[:, i])
        hy = mutual_info_score(y, y)
        hxy = mutual_info_score(X[:, i], y)
        sym_uncertainty = 2 * hxy / (hx + hy) if (hx + hy) != 0 else 0
        scores.append(sym_uncertainty)
    return np.array(scores)

# Özellikleri discretize etme
def discretize_features(X):
    return np.apply_along_axis(
        lambda col: np.digitize(col, bins=np.histogram_bin_edges(col, bins='auto')), axis=0, arr=X
    )

# Veriyi yükleme ve ön işleme
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['emotion']).values
    y = df['emotion'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, df.drop(columns=['emotion']).columns, df['emotion']

# Özellik seçim algoritmalarını çalıştırma
def feature_selection_algorithms(X, y, feature_names):
    feature_sets = []
    feature_ranks = {name: 0 for name in feature_names}

    X_binned = discretize_features(X)

    # Information Gain (IG)
    ig_scores = information_gain(X_binned, y)
    ig_indices = np.argsort(ig_scores)[-100:]
    for idx, feature_idx in enumerate(ig_indices):
        feature_ranks[feature_names[feature_idx]] += (100 - idx)
    feature_sets.append(set(ig_indices))

    # RFE (Recursive Feature Elimination)
    rfe_model = LogisticRegression(max_iter=500, random_state=42)
    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=100, step=1)
    rfe_selector.fit(X, y)
    rfe_features = rfe_selector.get_support(indices=True)
    for idx, feature_idx in enumerate(rfe_features):
        feature_ranks[feature_names[feature_idx]] += (100 - idx)
    feature_sets.append(set(rfe_features))

    # SelectKBest
    selectkbest_selector = SelectKBest(k=100)
    selectkbest_selector.fit(X, y)
    selectkbest_features = selectkbest_selector.get_support(indices=True)
    for idx, feature_idx in enumerate(selectkbest_features):
        feature_ranks[feature_names[feature_idx]] += (100 - idx)
    feature_sets.append(set(selectkbest_features))

    # ANOVA F-Value
    anova_selector = SelectKBest(score_func=f_classif, k=100)
    anova_selector.fit(X, y)
    anova_features = anova_selector.get_support(indices=True)
    for idx, feature_idx in enumerate(anova_features):
        feature_ranks[feature_names[feature_idx]] += (100 - idx)
    feature_sets.append(set(anova_features))

    # VarianceThreshold (Low variance features removal)
    variance_selector = VarianceThreshold(threshold=0.01)
    variance_selector.fit(X)
    variance_features = set(np.where(variance_selector.get_support())[0])
    feature_sets.append(variance_features)

    # RandomForest Feature Selection (using RandomForestClassifier)
    tree_model = RandomForestClassifier(n_estimators=100, random_state=42)
    tree_model.fit(X, y)
    tree_selector = SelectFromModel(estimator=tree_model, prefit=True, threshold="mean")
    tree_features = tree_selector.get_support(indices=True)
    for idx, feature_idx in enumerate(tree_features):
        feature_ranks[feature_names[feature_idx]] += (100 - idx)
    feature_sets.append(set(tree_features))

    # Puanları toplayıp sıralama yapma
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1], reverse=True)

    # İlk 100 özellik
    top_100_features = sorted_features[:126]
    top_100_feature_names = [feature[0] for feature in top_100_features]
    top_100_feature_scores = [feature[1] for feature in top_100_features]

    return top_100_feature_names, top_100_feature_scores

# Seçilen özellikleri kaydetme
def save_selected_features(X, top_100_feature_names, emotions):
    # Seçilen 100 özelliklerin indekslerini buluyoruz
    feature_indices = [i for i, name in enumerate(feature_names) if name in top_100_feature_names]
    
    # Seçilen 100 özelliklerin X_top_100 ile ilgisini kuruyoruz
    X_top_100 = X[:, feature_indices]
    
    # Seçilen özellikler ve hedef (emotion) değerleri ile DataFrame oluşturuyoruz
    top_100_features_df = pd.DataFrame(X_top_100, columns=top_100_feature_names)
    top_100_features_df['emotion'] = emotions.values
    
    # CSV dosyasına kaydediyoruz
    top_100_features_df.to_csv("selected_features.csv", index=False)
    print("Seçilen 100 özellik başarıyla kaydedildi!")


if __name__ == "__main__":
    csv_path = "combined_dataset.csv"
    X_scaled, y_encoded, feature_names, emotions = load_and_preprocess_data(csv_path)

    top_100_feature_names, top_100_feature_scores = feature_selection_algorithms(
        X_scaled, y_encoded, feature_names
    )

    print(f"En İyi 100 Seçilen Özellik Sayısı: {len(top_100_feature_names)}")
    print("\nEn İyi 100 Özellikler:")
    print(top_100_feature_names)
    print("\nPuanlar:")
    print(top_100_feature_scores)

    save_selected_features(X_scaled, top_100_feature_names, emotions)