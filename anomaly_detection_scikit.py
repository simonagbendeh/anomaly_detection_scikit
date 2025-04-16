
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulate network traffic data (normal and anomalous)
np.random.seed(42)
normal_traffic = np.random.normal(loc=50, scale=5, size=(100, 2))   # Normal traffic
anomalous_traffic = np.random.normal(loc=70, scale=1, size=(10, 2)) # Anomalies

# Combine data
data = np.vstack((normal_traffic, anomalous_traffic))

# Fit Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(data)
predictions = model.predict(data)  # -1 for anomalies, 1 for normal

# Visualize the results
plt.figure(figsize=(8, 6))
for i, point in enumerate(data):
    if predictions[i] == -1:
        plt.scatter(point[0], point[1], color='red', label='Anomaly' if i == 100 else "")
    else:
        plt.scatter(point[0], point[1], color='blue', label='Normal' if i == 0 else "")
plt.title("Anomaly Detection in Simulated Network Traffic")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
