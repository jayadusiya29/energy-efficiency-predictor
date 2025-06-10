# vehicle_clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# Generate synthetic dataset for vehicles
np.random.seed(0)
data_size = 300
data = {
    'Weight': np.random.randint(1000, 3000, data_size),
    'EngineSize': np.random.uniform(1.0, 4.0, data_size),
    'Horsepower': np.random.randint(50, 300, data_size)
}
df = pd.DataFrame(data)

# Prepare feature matrix
X = df

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Plot clusters (2D: Weight vs Horsepower)
plt.figure(figsize=(8, 6))
plt.scatter(df['Weight'], df['Horsepower'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Vehicle Clusters Based on Weight and Horsepower')
plt.grid(True)
plt.show()
