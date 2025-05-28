import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class UserHealthData(db.Model):
    __tablename__ = "user_health_data"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, unique=True, nullable=False)
    hemoglobin = db.Column(db.Float, nullable=True)
    systolic = db.Column(db.Integer, nullable=True)
    diastolic = db.Column(db.Integer, nullable=True)
    total_cholesterol = db.Column(db.Float, nullable=True)
    hba1c = db.Column(db.Float, nullable=True)

# -------------------- ANEMIA MODEL --------------------
def load_anemia_model():
    data = pd.read_csv('small/Anemia_Dataset.csv')
    # Encode categorical features and target
    anemia_le_gender = LabelEncoder()
    data['Gender_encoded'] = anemia_le_gender.fit_transform(data['Gender'])
    anemia_le_status = LabelEncoder()
    data['Anemia_Status_encoded'] = anemia_le_status.fit_transform(data['Anemia_Status'])
    # Define features and target
    X = data[['Age', 'Gender_encoded', 'Hemoglobin Level (g/dL)']]
    y = data['Anemia_Status_encoded']
    # Split the data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    anemia_scaler = StandardScaler()
    X_train_scaled = anemia_scaler.fit_transform(X_train)
    X_test_scaled = anemia_scaler.transform(X_test)
    # Build and train the model
    anemia_model = LogisticRegression(class_weight='balanced')
    anemia_model.fit(X_train_scaled, y_train)
    return anemia_model, anemia_le_gender, anemia_le_status, anemia_scaler

def predict_anemia_status(model, le_gender, le_status, scaler, age, gender, hemoglobin):
    # Domain override: if hemoglobin > 18.5, classify as Polycythemia
    if hemoglobin > 18.5:
        return "Polycythemia"
    gender_enc = le_gender.transform([gender])[0]
    features = np.array([[age, gender_enc, hemoglobin]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return le_status.inverse_transform(prediction)[0]

# -------------------- CHOLESTEROL MODEL --------------------
def load_cholesterol_model():
    chol_data = pd.read_csv('small/cholesterol_dataset.csv')
    # Map Gender to numeric values
    chol_data['Gender'] = chol_data['Gender'].map({'Male': 0, 'Female': 1})
    # Define features and target
    X = chol_data[['Age', 'Gender', 'Total Cholesterol', 'LDL Cholesterol', 'HDL Cholesterol']]
    y = chol_data['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a RandomForest classifier
    chol_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    chol_clf.fit(X_train, y_train)
    return chol_clf

def predict_cholesterol_status(clf, age, gender, total_chol, ldl, hdl):
    # Convert gender if provided as a string
    if isinstance(gender, str):
        gender = 0 if gender.lower() == 'male' else 1
    sample = pd.DataFrame([[age, gender, total_chol, ldl, hdl]],
                          columns=['Age', 'Gender', 'Total Cholesterol', 'LDL Cholesterol', 'HDL Cholesterol'])
    return clf.predict(sample)[0]

# -------------------- BLOOD PRESSURE MODEL --------------------
def load_blood_pressure_model():
    bp_df = pd.read_csv('small/Blood_Pressure_Dataset.csv')
    # Encode categorical columns
    bp_le_gender = LabelEncoder()
    bp_df['Gender'] = bp_le_gender.fit_transform(bp_df['Gender'])
    bp_le_status = LabelEncoder()
    bp_df['Status'] = bp_le_status.fit_transform(bp_df['Status'])
    # Define features and target
    bp_features = ['Age', 'Gender', 'Systolic (mmHg)', 'Diastolic (mmHg)']
    X = bp_df[bp_features]
    y = bp_df['Status']
    # Scale features
    bp_scaler = StandardScaler()
    X[bp_features] = bp_scaler.fit_transform(X[bp_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Decision Tree classifier
    bp_model = DecisionTreeClassifier(random_state=42)
    bp_model.fit(X_train, y_train)
    return bp_model, bp_le_gender, bp_le_status, bp_scaler, bp_features

def predict_bp_status(model, le_gender, le_status, scaler, features, age, gender, systolic, diastolic):
    gender_encoded = le_gender.transform([gender])[0]
    input_data = pd.DataFrame([[age, gender_encoded, systolic, diastolic]], columns=features)
    input_data[features] = scaler.transform(input_data[features])
    pred = model.predict(input_data)
    return le_status.inverse_transform(pred)[0]

# -------------------- DIABETES MODEL --------------------
def load_diabetes_model():
    diabetes_df = pd.read_csv('small/Diabetes_Dataset.csv')
    # Convert Gender to numeric
    diabetes_df['Gender'] = diabetes_df['Gender'].apply(lambda x: 0 if x.lower() == 'male' else 1)
    # Encode 'Status' if needed
    if diabetes_df['Status'].dtype == 'object':
        diabetes_le = LabelEncoder()
        diabetes_df['Status'] = diabetes_le.fit_transform(diabetes_df['Status'])
    else:
        diabetes_le = LabelEncoder()  # dummy encoder if not needed
    # Define features and target
    X = diabetes_df[['Age', 'Gender', 'HbA1c']]
    y = diabetes_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    diabetes_model = RandomForestClassifier(random_state=42)
    diabetes_model.fit(X_train, y_train)
    return diabetes_model, diabetes_le

def predict_diabetes(model, le, age, gender, hba1c):
    gender_encoded = 0 if gender.lower() == 'male' else 1
    data_instance = [[age, gender_encoded, hba1c]]
    prediction = model.predict(data_instance)
    return le.inverse_transform(prediction)[0]

def main():
    age = int(input("Enter Age: "))
    gender = input("Enter Gender (Male/Female): ")
    hemoglobin = float(input("Enter Hemoglobin Level (g/dL): "))
    total_chol = float(input("Enter Total Cholesterol: "))
    ldl = float(input("Enter LDL Cholesterol: "))
    hdl = float(input("Enter HDL Cholesterol: "))
    systolic = float(input("Enter Systolic (mmHg): "))
    diastolic = float(input("Enter Diastolic (mmHg): "))
    hba1c = float(input("Enter HbA1c: "))

    
    # Anemia prediction
    anemia_model, anemia_le_gender, anemia_le_status, anemia_scaler = load_anemia_model()
    anemia_result = predict_anemia_status(anemia_model, anemia_le_gender, anemia_le_status, anemia_scaler,
                                          age, gender, hemoglobin)
    
    # Cholesterol prediction
    chol_model = load_cholesterol_model()
    chol_result = predict_cholesterol_status(chol_model, age, gender, total_chol, ldl, hdl)
    
    # Blood pressure prediction
    bp_model, bp_le_gender, bp_le_status, bp_scaler, bp_features = load_blood_pressure_model()
    bp_result = predict_bp_status(bp_model, bp_le_gender, bp_le_status, bp_scaler, bp_features,
                                  age, gender, systolic, diastolic)
    
    # Diabetes prediction
    diabetes_model, diabetes_le = load_diabetes_model()
    diabetes_result = predict_diabetes(diabetes_model, diabetes_le, age, gender, hba1c)
    if anemia_result == "Normal":
        print("anemia test is normal")
    else:
        print( anemia_result)
    if chol_result == "Healthy":
        print("Cholestrol test is normal")
    else:
        print(chol_result)
    if bp_result == "Healthy":
        print("anemia test is normal")
    else:
        print(bp_result)
    if anemia_result == "Healthy":
        print("Diabetis test is normal")
    else:
        print(diabetes_result)

if __name__ == "__main__":
    main()
