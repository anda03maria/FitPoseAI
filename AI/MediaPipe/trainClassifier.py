import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib
import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def mse_distance(a, b):
    return np.mean((np.array(a) - np.array(b))**2)

df = pd.read_csv('barbellrow_dataset.csv')

X = df[[
    'back_angle_L', 'knee_angle_L', 'arm_angle_L', 'trunk_angle_L', 'torso_bend_L',
    'back_angle_R', 'knee_angle_R', 'arm_angle_R', 'trunk_angle_R', 'torso_bend_R'
]]
y = df['label']

corect_samples = X[y == 1]
mean_angles = corect_samples.mean().values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Incorect', 'Corect'], yticklabels=['Incorect', 'Corect'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - inițial')
plt.show()

dists = [mse_distance(sample, mean_angles) for sample in X_test.values]

plt.hist(dists, bins=50)
plt.title('Distribuția distanței față de execuția ideală')
plt.xlabel('MSE fata de medie (clasa corecta)')
plt.ylabel('Frecvență')
plt.show()

#threshold = np.percentile(dists, 90)
threshold = np.percentile(dists, 75)

adjusted_preds = []
for pred, dist in zip(y_pred, dists):
    if pred == 1 and dist > threshold:
        adjusted_preds.append(0)
    else:
        adjusted_preds.append(pred)

print("\n[CLASSIFICATION REPORT - după ajustare cu prag de distanță]")
print(classification_report(y_test, adjusted_preds))

cm_adj = confusion_matrix(y_test, adjusted_preds)
sns.heatmap(cm_adj, annot=True, fmt='d', cmap='Purples', xticklabels=['Incorect', 'Corect'], yticklabels=['Incorect', 'Corect'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - după ajustare')
plt.show()

joblib.dump(model, '../run_app/xgboost_barbellrow_model.pkl')
