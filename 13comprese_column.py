import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import joblib

date="M8-12-2025"
name="moving_hands_try"

# Load your CSV
df = pd.read_csv("data/Moving_hands.csv")

# Split features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_categorical, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)
    # Dropout(0.3),
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_pca.shape[1],)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stop])
# , callbacks=[early_stop]

joblib.dump(pca, "pca.pkl")
model.save(f"ML-model/{date}-{name}/model.h5")

with open(f"ML-model/{date}-{name}/text.txt", "w") as f:
    for label in le.classes_:
        f.write(str(label) + "\n")

joblib.dump(pca, f"ML-model/{date}-{name}/pca.pkl")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("keras_loss_plot.png")
plt.show()