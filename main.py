import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, \
    recall_score
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# Load Data
data = pd.read_csv('heart_attack_dataset.csv')

# Encode Binary Categorical Features
data['Outcome'] = data['Outcome'].map({'No Heart Attack': 0, 'Heart Attack': 1})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Medication'] = data['Medication'].map({'Yes': 1, 'No': 0})
data['ExerciseInducedAngina'] = data['ExerciseInducedAngina'].map({'Yes': 1, 'No': 0})

# Normalize Continuous Variables
scaler = MinMaxScaler()
data[['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'Income', 'Age', 'NumberOfMajorVessels', 'PhysicalActivity',
      'StressLevel', 'AlcoholConsumption', 'MaxHeartRate', 'ST_Depression']] = (
    scaler.fit_transform(
        data[['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'Income', 'Age', 'NumberOfMajorVessels',
              'PhysicalActivity', 'StressLevel', 'AlcoholConsumption', 'MaxHeartRate', 'ST_Depression']]))

# Encode Multi-Class Categorical Features
OneHotEncoder_Scaler = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = OneHotEncoder_Scaler.fit_transform(
    data[['Diet', 'Ethnicity', 'EducationLevel', 'ChestPainType', 'ECGResults', 'Slope', 'Thalassemia',
          'Residence', 'EmploymentStatus', 'MaritalStatus']])
encoded_features = pd.DataFrame(encoded_features, columns=OneHotEncoder_Scaler.get_feature_names_out(
    ['Diet', 'Ethnicity', 'EducationLevel', 'ChestPainType', 'ECGResults', 'Slope', 'Thalassemia',
     'Residence', 'EmploymentStatus', 'MaritalStatus']))
data = data.drop(['Diet', 'Ethnicity', 'EducationLevel', 'ChestPainType', 'ECGResults', 'Slope', 'Thalassemia',
                  'Residence', 'EmploymentStatus', 'MaritalStatus'], axis=1)

# Merge Encoded Features
data = pd.concat([data, encoded_features], axis=1)
data_target = data['Outcome']
data = data.drop(['Outcome'], axis=1)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(data, data_target, test_size=0.3, random_state=42)
# Define Models
models = {'Decision Tree': DecisionTreeClassifier(max_depth=8),
          'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=1000,
                                          learning_rate='adaptive', max_iter=200, activation='logistic'),
          'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.3, min_samples_split=1000),
          'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=8, max_leaf_nodes=1000)}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_train_prediction = model.predict(x_train)
    y_test_prediction = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred=y_test_prediction, labels=model.classes_)
    accuracy_train = accuracy_score(y_train, y_pred=y_train_prediction)
    accuracy_test = accuracy_score(y_test, y_pred=y_test_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot()
    plt.title(f'Confusion Matrix: {name}')
    plt.show()
