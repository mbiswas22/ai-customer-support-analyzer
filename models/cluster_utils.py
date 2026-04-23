import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def build_cluster_features(df):
    """Encode categorical + numeric columns into a feature matrix for clustering."""
    data = df[['category', 'sentiment', 'priority', 'urgent', 'response_time_hours']].copy()

    for col in ['category', 'sentiment', 'priority']:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    data['urgent'] = data['urgent'].astype(int)
    return StandardScaler().fit_transform(data)


def run_clustering(df, n_clusters=3):
    X = build_cluster_features(df)

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    result = df[['text', 'category', 'sentiment', 'priority', 'urgent', 'response_time_hours']].copy().reset_index(drop=True)
    result['cluster'] = labels.astype(str)
    result['pca_x'] = coords[:, 0]
    result['pca_y'] = coords[:, 1]
    return result
