import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
name = "M8-21-2025-moving_hands_full"
model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r") as f:
    class_names = f.read().splitlines()
predicted_classes = []
df = pd.read_csv("data/Moving_hands3.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  
predictions = model.predict(X)
predicted_index = np.argmax(predictions, axis=1)
for id in predicted_index:
    predicted_classes.append(class_names[id])

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have predicted_classes and true_labels
cm = confusion_matrix(y, predicted_classes)

# Optional: Plot the confusion matrix for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()