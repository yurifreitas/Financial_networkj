import numpy as np
from sklearn.decomposition import PCA

def compute_pca(series: np.ndarray, n_components=1):
    series_reshaped = series.reshape(-1, 1)
    pca = PCA(n_components=n_components)
    principal = pca.fit_transform(series_reshaped)
    return principal.flatten().tolist()
