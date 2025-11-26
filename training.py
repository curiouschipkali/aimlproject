import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# Hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("Running GridSearchCV...")
grid.fit(X_train_scaled, y_train)

print("\nBest Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Final tuned model
best_knn = grid.best_estimator_

# Evaluate
y_pred = best_knn.predict(X_test_scaled)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_knn, "model_knn.pkl")

print("\nSaved model_knn.pkl and scaler.pkl")
