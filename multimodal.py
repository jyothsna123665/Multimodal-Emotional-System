import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Example training data (replace with your collected predictions)
# Format: [speech_prob, text_prob, face_prob]

X = []
y = []

# 🔴 You should collect real predictions later
# For now dummy data
for _ in range(500):
    sp = np.random.rand()
    tp = np.random.rand()
    fp = np.random.rand()

    X.append([sp, tp, fp])
    y.append(np.argmax([sp, tp, fp]))  # dummy label

X = np.array(X)
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)

# Save model
with open("fusion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Fusion model trained & saved")