import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load pre-extracted dataset
X = np.load("model_files/X_features.npy")
y = np.load("model_files/y_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model_files/scaler.pkl")

#KNN 
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

joblib.dump(knn, "model_files/model_knn.pkl")

print("\nKNN Accuracy:", accuracy_score(y_test, knn.predict(X_test_scaled)))

# SVM
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train_scaled, y_train)

joblib.dump(svm, "model_files/model_svm.pkl")

print("\nSVM Accuracy:", accuracy_score(y_test, svm.predict(X_test_scaled)))

#Random Forest
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)

joblib.dump(rf, "model_files/model_rf.pkl")

print("\nRandomForest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

#Gradient Boosted Trees
gbt = GradientBoostingClassifier()
gbt.fit(X_train, y_train)

joblib.dump(gbt, "model_files/model_gbt.pkl")

print("\nGBT Accuracy:", accuracy_score(y_test, gbt.predict(X_test)))
