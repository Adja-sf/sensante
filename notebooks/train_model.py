import pandas as pd
import numpy as np

# =========================
# CHARGEMENT DATASET
# =========================

df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# =========================
# ENCODAGE
# =========================

from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

# =========================
# SPLIT
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# =========================
# MODELE
# =========================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})

print(comparison)

# =========================
# METRIQUES
# =========================

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

print("\nMatrice de confusion :")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# VISUALISATION
# =========================

import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')

plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("Figure sauvegardee dans figures/confusion_matrix.png")

# =========================
# SAUVEGARDE ← CORRIGÉ
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/le_sexe.pkl")        # ← corrigé
joblib.dump(le_region, "models/le_region.pkl")    # ← corrigé
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Modele + encodeurs sauvegardes")

# =========================
# RECHARGEMENT ← CORRIGÉ
# =========================

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/le_sexe.pkl")      # ← corrigé
le_region_loaded = joblib.load("models/le_region.pkl")  # ← corrigé

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# =========================
# TEST NOUVEAU PATIENT
# =========================

nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Resultat ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

# =========================
# IMPORTANCES
# =========================

importances = model.feature_importances_

for name, imp in sorted(
    zip(feature_cols, importances),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:20s} : {imp:.3f}")

# =========================
# TEST MULTI-PATIENTS
# =========================

p1 = pd.DataFrame([[18, 1, 36.5, 120, 0, 0, 0, 0]], columns=feature_cols)
p2 = pd.DataFrame([[35, 0, 39.8, 130, 1, 1, 1, 1]], columns=feature_cols)
p3 = pd.DataFrame([[70, 1, 38.2, 140, 1, 0, 0, 1]], columns=feature_cols)

patients = [p1, p2, p3]

for i, patient in enumerate(patients, 1):
    prediction = model_loaded.predict(patient)[0]
    print(f"Patient {i} → Diagnostic : {prediction}")