import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Your raw confusion matrix
cm = np.array([
    [6, 13, 30],
    [7, 2372, 1147],
    [21, 65, 95]
])

# Class labels
class_names = ['Erosion', 'Normal', 'Ulcer']

# Normalize the confusion matrix row-wise
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# # Plot raw confusion matrix
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.title("Confusion Matrix (Raw Counts)")
# plt.xlabel("Predicted")
# plt.ylabel("True")

# Plot normalized confusion matrix
plt.subplot(1, 1,1)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()
