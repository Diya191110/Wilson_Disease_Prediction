import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("data/Wilson_disease_dataset.csv")

# Drop irrelevant columns
columns_to_drop = ['Name', 'Patient_ID', 'Phone_No', 'Address']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Label encode object columns
label_encoders = {}
label_value_hints = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    label_value_hints[col] = list(le.classes_)

# Keep only important features
cor_matrix = df.corr(numeric_only=True)['Is_Wilson_Disease'].abs()
important_cols = cor_matrix[cor_matrix > 0.05].index.tolist()
df = df[important_cols]

# Features and target
X = df.drop(columns='Is_Wilson_Disease')
y = df['Is_Wilson_Disease']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
rf = RandomForestClassifier(n_estimators=150, random_state=42)
svm = SVC(kernel='rbf', C=2, gamma='scale', probability=True, random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
lr = LogisticRegression()

# Train and evaluate individual models
models = {'Random Forest': rf, 'SVM': svm, 'XGBoost': xgb}
print("\n--- Individual Model Evaluations ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Stacking Ensemble
stack = StackingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], final_estimator=lr, cv=5)
stack.fit(X_train, y_train)
y_stack_pred = stack.predict(X_test)

# Stacking Evaluation
print("\n--- Stacking Ensemble Performance ---")
print("Accuracy:", accuracy_score(y_test, y_stack_pred))
print("F1 Score:", f1_score(y_test, y_stack_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_stack_pred))
print("Classification Report:\n", classification_report(y_test, y_stack_pred))

# Real-Time Prediction from User
def predict_from_user():
    print("\n--- Enter New Patient Data (with normal value ranges) ---")

    normal_ranges = {
        'Age': "years (Normal: <40)",
        'Sex': "(Enter: Male/Female)",
        'Serum_Ceruloplasmin': "mg/dL (Normal: 20–35)",
        'Serum_Copper': "mcg/dL (Normal: 70–140)",
        '24_hr_urinary_copper': "mcg/day (Normal: <40, Wilson's: >100)",
        'Kayser_Fleischer_Rings': "(Enter: Yes/No; Normal: No)",
        'Liver_Enzymes_ALT': "U/L (Normal: <55)",
        'Liver_Enzymes_AST': "U/L (Normal: <40)",
        'Total_Bilirubin': "mg/dL (Normal: 0.1–1.2)",
        'Neuro_Psych_Symptoms': "(Enter: Yes/No; Normal: No)",
        'Slit_Lamp_Exam': "(Enter: Positive/Negative; Normal: Negative)",
        'MRI_Brain_Findings': "(Enter: Normal/Abnormal)",
        'Family_History': "(Enter: Yes/No)",
        'Genetic_Test_Result': "(Enter: Positive/Negative)"
    }

    user_input = {}
    for feature in X.columns:
        hint = normal_ranges.get(feature, "")
        if feature in label_value_hints:
            options = '/'.join(label_value_hints[feature])
            val = input(f"{feature} {hint}: ").strip()
            le = label_encoders[feature]
            try:
                val = le.transform([val])[0]
            except:
                print(f"❌ Invalid value. Choose from: {options}")
                return
        else:
            try:
                val = float(input(f"{feature} {hint}: "))
            except:
                print("❌ Please enter a valid number.")
                return
        user_input[feature] = val

    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = stack.predict(user_scaled)[0]
    print("\n✅ Prediction Result:")
    print("🩺 Patient is", "**Positive** for Wilson Disease" if prediction == 1 else "**Negative** for Wilson Disease")

# Start prediction
predict_from_user()
