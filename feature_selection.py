import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector, f_classif, VarianceThreshold
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def symmetrical_uncertainty(X, y):
    scores = []
    for i in range(X.shape[1]):
        hx = mutual_info_score(X[:, i], X[:, i])
        hy = mutual_info_score(y, y)
        hxy = mutual_info_score(X[:, i], y)
        sym_uncertainty = 2 * hxy / (hx + hy) if (hx + hy) != 0 else 0
        scores.append(sym_uncertainty)
    return np.array(scores)

def information_gain(X, y):
    scores = []
    for i in range(X.shape[1]):
        mutual_info = mutual_info_score(y, X[:, i])
        scores.append(mutual_info)
    return np.array(scores)

def discretize_features(X):
    return np.apply_along_axis(
        lambda col: np.digitize(col, bins=np.histogram_bin_edges(col, bins='auto')), axis=0, arr=X
    )

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['emotion']).values
    y = df['emotion'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, df.drop(columns=['emotion']).columns, df['emotion']

def feature_selection_algorithms(X, y, feature_names, min_algorithms=2):
    feature_sets = []

    X_binned = discretize_features(X)

    ig_scores = information_gain(X_binned, y)
    ig_indices = np.argsort(ig_scores)[-100:]
    feature_sets.append(set(ig_indices))

    rfe_model = LogisticRegression(max_iter=500, random_state=42)
    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=100, step=1)
    rfe_selector.fit(X, y)
    rfe_features = rfe_selector.get_support(indices=True)
    feature_sets.append(set(rfe_features))

    selectkbest_selector = SelectKBest(k=100)
    selectkbest_selector.fit(X, y)
    selectkbest_features = selectkbest_selector.get_support(indices=True)
    feature_sets.append(set(selectkbest_features))

    tree_model = RandomForestClassifier(n_estimators=100, random_state=42)
    tree_model.fit(X, y)
    tree_selector = SelectFromModel(estimator=tree_model, prefit=True, threshold="mean")
    tree_features = tree_selector.get_support(indices=True)
    feature_sets.append(set(tree_features))

    anova_selector = SelectKBest(score_func=f_classif, k=100)
    anova_selector.fit(X, y)
    anova_features = anova_selector.get_support(indices=True)
    feature_sets.append(set(anova_features))

    variance_selector = VarianceThreshold(threshold=0.01)
    variance_selector.fit(X)
    variance_features = set(np.where(variance_selector.get_support())[0])
    feature_sets.append(variance_features)

    feature_counts = {}
    for feature_set in feature_sets:
        for feature in feature_set:
            if feature not in feature_counts:
                feature_counts[feature] = 0
            feature_counts[feature] += 1

    intersect_features = {feature for feature, count in feature_counts.items() if count >= min_algorithms}

    intersect_feature_names = [feature_names[i] for i in intersect_features]

    return intersect_features, intersect_feature_names

def save_selected_features(X, intersect_features, intersect_feature_names, emotions):
    X_intersect = X[:, list(intersect_features)]
    intersect_features_df = pd.DataFrame(X_intersect, columns=intersect_feature_names)
    intersect_features_df['emotion'] = emotions.values
    intersect_features_df.to_csv("selected_features.csv", index=False)
    print("Seçilen özellikler başarıyla kaydedildi!")

if __name__ == "__main__":
    csv_path = "combined_dataset.csv"
    X_scaled, y_encoded, feature_names, emotions = load_and_preprocess_data(csv_path)

    intersect_features, intersect_feature_names = feature_selection_algorithms(
        X_scaled, y_encoded, feature_names
    )

    print(f"Intersect Seçilen Özellik Sayısı: {len(intersect_features)}")
    print("\nIntersect Özellikler:")
    print(intersect_feature_names)

    save_selected_features(X_scaled, intersect_features, intersect_feature_names, emotions)