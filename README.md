Clustering Analysis of Credit Card Transactions
Project Overview
This project applies unsupervised learning techniques to a dataset of credit card transactions to identify patterns and group similar observations. By clustering transactions, we aim to uncover insights into different types of behaviors, which could potentially help in identifying fraudulent activities. The analysis explores various clustering models such as K-Means, DBSCAN, and uses dimensionality reduction techniques like PCA and t-SNE to visually interpret the data.

Dataset
Total Records: 284,807 transactions
Features:
Time: Time elapsed between the transaction and the first transaction.
V1 to V28: Principal Component Analysis (PCA) transformed features.
Amount: The transaction amount.
Class: Target variable where '0' represents legitimate transactions and '1' indicates fraudulent transactions.
The dataset is highly imbalanced, with relatively few fraudulent transactions. Our clustering analysis focuses on identifying natural groupings without using the Class labels.

Data Exploration and Preprocessing
Checked for missing values; none were found.
Normalized Time and Amount features to avoid bias in clustering.
Standardized data for better performance of clustering algorithms.
Models and Techniques
K-Means Clustering

Used the Elbow method to determine the optimal number of clusters.
Identified three main groups, though the clusters were not clearly distinguishable.
DBSCAN Clustering

No requirement to specify a set number of clusters.
Detected a high level of noise, with many points labeled as outliers.
Dimensionality Reduction

PCA: Reduced data to 2 components for visualization, showing some separation of clusters.
t-SNE: Visualized non-linear patterns, providing insights that PCA did not capture.
Key Findings
Clustering models revealed natural groupings within the transactions.
PCA and t-SNE visualizations suggested patterns indicating different behaviors in transaction processing.
Further feature engineering or domain-specific data could improve clustering results.
Recommendations and Next Steps
Additional Feature Engineering: Incorporate more domain-specific features or aggregate transactions over time.
Model Improvement: Fine-tune hyperparameters or experiment with other clustering techniques like Gaussian Mixture Models.
Supervised Learning Integration: Use the identified clusters to build separate models for fraud detection.
How to Run the Code
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/creditcard-clustering-analysis.git
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Download the dataset and place it in the root directory.
Run the script:
bash
Copy code
python creditcardproject.py
Results
K-Means: Showed a reasonable clustering with three groups.
DBSCAN: Identified noise and outliers but was less effective in defining clear clusters.
Visualizations: PCA and t-SNE plots helped in understanding the data distribution and clustering performance.
Conclusion
This project demonstrates the use of clustering techniques to analyze credit card transactions. Although K-Means was preferred, there's potential for further improvement with better feature engineering and parameter tuning.

License
This project is licensed under the MIT License - see the LICENSE file for details.

