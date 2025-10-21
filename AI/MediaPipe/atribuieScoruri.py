import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("date_cu_scoruri_fara_label.csv")

# if 'label' and 'id' in df.columns:
#     df = df.drop(columns=['label'])
#     df = df.drop(columns=['id'])

df = df.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

# 3. Clustering (ex: 3 clustere: execuție bună, medie, slabă)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. Calcul distanță față de centroidul „ideal” (media tuturor execuțiilor)
ideal = np.mean(X_scaled, axis=0)

def distance_to_score(x, ideal, max_dist):
    dist = np.mean(np.abs(x - ideal))
    score = 10 - (dist / max_dist) * 9
    return max(1, min(10, score))

distances = np.array([np.mean(np.abs(x - ideal)) for x in X_scaled])
max_dist = np.max(distances)
scores = [distance_to_score(x, ideal, max_dist) for x in X_scaled]

df['score'] = scores
df.drop(columns=['cluster'], inplace=True)

def scor_to_eticheta(scor):
    if scor > 8:
        return "corect"
    elif scor >= 5:
        return "mediu"
    else:
        return "slab"

df['eticheta'] = df['score'].apply(scor_to_eticheta)
df.drop(columns=['score'], inplace=True)

df_slab = df[df['eticheta'] == 'slab']
df_mediu = df[df['eticheta'] == 'mediu']
df_corect = df[df['eticheta'] == 'corect']

# Target: vrem ~7000 exemple în fiecare
target_size = len(df_mediu)

# Oversample (alege cu înlocuire)
df_slab_oversampled = df_slab.sample(n=target_size, replace=True, random_state=42)
df_corect_oversampled = df_corect.sample(n=target_size, replace=True, random_state=42)

# Combinăm toate
df_balanced = pd.concat([df_mediu, df_slab_oversampled, df_corect_oversampled])

# Amestecăm rândurile
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Salvăm noul fișier
df_balanced.to_csv("date_balansate.csv", index=False)
print("✔ Fișierul cu date balansate a fost salvat.")

