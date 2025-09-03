import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

date="M9-3-2025"
name="tree100"

# Load your CSV
df = pd.read_csv("data/main100.csv")

# Split features and labels
X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values  
y=[]
for i in range(99):
    for j in range(30):
        y.append(i)
for l in range(28):
    y.append(100)
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

tabnet_model = TabNetClassifier()
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['accuracy'],
    max_epochs=1000,
    patience=50,
    batch_size=128,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

y_pred_tabnet = tabnet_model.predict(X_test)
acc_tabnet = accuracy_score(y_test, y_pred_tabnet)
print(f"TabNet Test Accuracy: {acc_tabnet:.4f}")

tabnet_model.save_model(f"ML-model/{date}-{name}/tabnet_model.zip")
with open(f"ML-model/{date}-{name}/text.txt", "w") as f:
    for label in le.classes_:
        f.write(str(label) + "\n")
