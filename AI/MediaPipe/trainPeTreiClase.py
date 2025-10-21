import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
df = pd.read_csv("date_balansate.csv")

X = df.drop(columns=['eticheta']).values
y = df['eticheta'].values

# 3. Codificare etichete (ex: corect = 0, mediu = 1, slab = 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump((model, le), "../run_app/model_clasificare_xgb.joblib")

print("=== Raport de clasificare ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("=== Matrice de confuzie ===")
cm = confusion_matrix(y_test, y_pred)
labels = le.classes_

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - XGBoost Classification')
plt.tight_layout()
plt.show()

