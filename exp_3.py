import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    # High-dimensional data
    data = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50],
        [25, 35, 45, 55],
        [30, 40, 50, 60],
    ], dtype=float)

    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("PCA requires at least 2 samples and 2 features.")

    print("Original Data:")
    print(data)

    # Step 1: Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Step 2: Apply PCA
    # Reduce to at most 2 dimensions, while keeping the request valid for the dataset.
    n_components = min(2, data_scaled.shape[1], data_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_scaled)

    # Display reduced data
    print("\nReduced Data after PCA:")
    print(reduced_data)

    # Display explained variance
    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)


if __name__ == "__main__":
    main()