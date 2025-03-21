# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 2: Load and Prepare Your Data
# Load the eGeMAPs features from CSV
df = pd.read_csv('predict_feature.csv', index_col=0)

# Separate features and labels
X = df.drop('label', axis=1)  # All columns except 'label'
y = df['label']  # The label column

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA()  # By default, it keeps all components
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Step 4: Plot Explained Variance
plt.figure(figsize=(10, 6))

# Individual explained variance
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')

# Cumulative explained variance
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')

# Highlight the point where cumulative variance reaches 95%
target_variance = 0.95
n_components_95 = np.argmax(cumulative_explained_variance >= target_variance) + 1
plt.axvline(n_components_95, color='r', linestyle='--', label=f'{target_variance*100:.0f}% Variance (n_components={n_components_95})')

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components (eGeMAPs Features)')
plt.legend(loc='best')
plt.grid()
plt.savefig('pca_test_variance_plot.png')
plt.show()

# Step 5: Print Component Information
print("\nPCA Component Information:")
for i, (ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    print(f"PC{i+1}: {ratio*100:.2f}% (Cumulative: {cum_ratio*100:.2f}%)")

# Step 6: Retain Selected Components
# Re-fit PCA with the selected number of components
pca_reduced = PCA(n_components=32)
X_pca_reduced = pca_reduced.fit_transform(X_scaled)

# Create DataFrame with reduced PCA components and labels
pca_df = pd.DataFrame(
    X_pca_reduced,
    columns=[f'PC{i+1}' for i in range(32)],
    index=X.index
)
pca_df['label'] = y  # Add back the labels

# Save the reduced PCA results
pca_df.to_csv('pca_results_reduced.csv')
print(f"\nReduced PCA results saved to: pca_results_reduced.csv")
print(f"Number of components to capture {target_variance*100:.0f}% variance: {n_components_95}")
print(f"Reduced data shape: {X_pca_reduced.shape}")

# Step 7: Plot PCA Components
plt.figure(figsize=(10, 6))
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Components 1 vs 2 (Colored by ADHD Label)')
plt.colorbar(label='ADHD Label')
plt.savefig('pca_test_components_plot.png')
plt.show()