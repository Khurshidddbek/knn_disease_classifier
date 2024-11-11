import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets
symptoms_df = pd.read_csv('data/DiseaseAndSymptoms.csv')
precautions_df = pd.read_csv('data/Disease precaution.csv')

# Fill NaN values in symptoms with 'None'
symptoms_df.fillna('None', inplace=True)

# Prepare binary features for symptoms
symptoms = symptoms_df.iloc[:, 1:].values.flatten()
symptoms = symptoms[symptoms != 'None']
unique_symptoms = np.unique(symptoms)

# Initialize DataFrame for binary features
features = pd.DataFrame(0, index=range(symptoms_df.shape[0]), columns=unique_symptoms)

# Populate binary features for each symptom
for index, row in symptoms_df.iterrows():
    for symptom in row[1:]:
        if symptom != 'None':
            features.at[index, symptom] = 1

# Merge features with target variable (Disease)
final_df = pd.concat([symptoms_df['Disease'], features], axis=1)

# Encode Disease labels as integers
le = LabelEncoder()
final_df['Disease'] = le.fit_transform(final_df['Disease'])

# Prepare data for model training
X = final_df.drop('Disease', axis=1)
y = final_df['Disease']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize k-NN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Optimal k selection
k_range = range(1, 21)
accuracy_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

# Plot accuracy vs. k values
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, marker='o', linestyle='dashed', color='b')
plt.title('Accuracy vs. k values')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Find the optimal k value
optimal_k = k_range[np.argmax(accuracy_scores)]
print(f'Optimal k: {optimal_k} with accuracy {max(accuracy_scores) * 100:.2f}%')

# Train final model with optimal k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)

# Predict and evaluate with the optimal model
y_pred_optimal = knn_optimal.predict(X_test_scaled)
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Optimal model accuracy with k={optimal_k}: {accuracy_optimal * 100:.2f}%')

# Confusion matrix for optimal model
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix for k={optimal_k}')
plt.show()

# Classification report for optimal model
print(classification_report(y_test, y_pred_optimal, target_names=le.classes_))

# Load precautions data and create a dictionary for recommendations
precautions_df.fillna('', inplace=True)
precautions_dict = {
    row['Disease']: [
        row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']
    ] for _, row in precautions_df.iterrows()
}

# Example of retrieving precautions for a predicted disease
predicted_disease_id = y_pred_optimal[0]  # Example: First prediction in the test set
predicted_disease = le.inverse_transform([predicted_disease_id])[0]
precautions = precautions_dict.get(predicted_disease, ["No precautions available."])

print(f'Precautions for {predicted_disease}:')
for precaution in precautions:
    print(f'- {precaution}')
