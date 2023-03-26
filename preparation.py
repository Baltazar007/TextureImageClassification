import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


# Lecture du fichier CSV
df = pd.read_csv('features.csv', header=None)

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# Normalisation des données avec StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Normalisation des données avec MinMaxScaler
scaler = MinMaxScaler()
X_train_mm = scaler.fit_transform(X_train)
X_test_mm = scaler.transform(X_test)

# Entraînement du modèle avec les données originales
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
joblib.dump(dtc, 'dtc_original.pkl')

svm = SVC()
svm.fit(X_train, y_train)
joblib.dump(svm, 'svm_original.pkl')

catboost = CatBoostClassifier()
catboost.fit(X_train, y_train)
joblib.dump(catboost, 'catboost_original.pkl')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, 'knn_original.pkl')

# Entraînement du modèle avec les données normalisées avec StandardScaler
dtc = DecisionTreeClassifier()
dtc.fit(X_train_std, y_train)
joblib.dump(dtc, 'dtc_std.pkl')

svm = SVC()
svm.fit(X_train_std, y_train)
joblib.dump(svm, 'svm_std.pkl')

catboost = CatBoostClassifier()
catboost.fit(X_train_std, y_train)
joblib.dump(catboost, 'catboost_std.pkl')

knn = KNeighborsClassifier()
knn.fit(X_train_std, y_train)
joblib.dump(knn, 'knn_std.pkl')

# Entraînement du modèle avec les données normalisées avec MinMaxScaler
dtc = DecisionTreeClassifier()
dtc.fit(X_train_mm, y_train)
joblib.dump(dtc, 'dtc_mm.pkl')

svm = SVC()
svm.fit(X_train_mm, y_train)
joblib.dump(svm, 'svm_mm.pkl')

catboost = CatBoostClassifier()
catboost.fit(X_train_mm, y_train)
joblib.dump(catboost, 'catboost_mm.pkl')

knn = KNeighborsClassifier()
knn.fit(X_train_mm, y_train)
joblib.dump(knn, 'knn_mm.pkl')
