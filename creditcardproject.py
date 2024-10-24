# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


# Load the dataset
file_path = 'creditcard.csv'
creditcard_data = pd.read_csv(file_path)


# Display basic dataset information
print("Dataset Info:")
creditcard_data.info()


print("\nFirst few rows of the dataset:")
print(creditcard_data.head())


# Initial data exploration: Check for missing values and basic statistical summary
missing_values = creditcard_data.isnull().sum().sum()
summary_statistics = creditcard_data.describe()


print("\nTotal Missing Values:", missing_values)
print("\nSummary Statistics:")
print(summary_statistics)


# Normalizing 'Amount' and 'Time' features
scaler = StandardScaler()
creditcard_data[['Amount', 'Time']] = scaler.fit_transform(creditcard_data[['Amount', 'Time']])


# Displaying the first few rows after normalization
print("\nFirst few rows after normalization:")
print(creditcard_data[['Time', 'Amount']].head())


# Define features for clustering (excluding 'Class' for unsupervised learning)
features = creditcard_data.drop(columns=['Class'])


# Run K-Means with a range of cluster values to determine the optimal number using the elbow method
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)


# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal K (K-Means)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# Applying K-Means with the optimal number of clusters
optimal_k = 3  # Example value, adjust based on the elbow method output
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
creditcard_data['KMeans_Cluster'] = kmeans.fit_predict(features)


# Summary of K-Means clusters
print("\nK-Means Clustering Results:")
print(creditcard_data['KMeans_Cluster'].value_counts())


# Applying DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
creditcard_data['DBSCAN_Cluster'] = dbscan.fit_predict(features)


# Summary of DBSCAN clusters
print("\nDBSCAN Clustering Results:")
print(creditcard_data['DBSCAN_Cluster'].value_counts())


# Applying PCA for dimensionality reduction (for visualization)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
creditcard_data['PCA1'] = pca_result[:, 0]
creditcard_data['PCA2'] = pca_result[:, 1]


# Scatter plot of PCA results with K-Means clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='KMeans_Cluster', data=creditcard_data, palette='viridis')
plt.title('PCA Visualization of Clusters (K-Means)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# Applying t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)
creditcard_data['TSNE1'] = tsne_result[:, 0]
creditcard_data['TSNE2'] = tsne_result[:, 1]


# Scatter plot of t-SNE results with K-Means clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='KMeans_Cluster', data=creditcard_data, palette='viridis')
plt.title('t-SNE Visualization of Clusters (K-Means)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()


# Final Insights and Next Steps
print("\nFinal Insights and Recommendations:")
print("1. The optimal number of clusters was determined using the elbow method.")
print("2. K-Means and DBSCAN clusters were created, and patterns observed in PCA and t-SNE visualizations.")
print("3. Further analysis may include adding additional features, fine-tuning clustering parameters, or testing other clustering algorithms.")


# python creditcardproject.py
