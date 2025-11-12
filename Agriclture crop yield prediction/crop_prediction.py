import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

# ===========================================
# 2️⃣ Upload or Load Dataset - MODIFIED FOR LOCAL FILE SYSTEM
# ===========================================

csv_filename = 'yield_df.csv'
print(f"📄 Loading dataset from local file: {csv_filename}")

try:
    data = pd.read_csv(csv_filename)
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print(f"\n❌ Error: The file '{csv_filename}' was not found.")
    print("Please make sure 'yield_df.csv' is in the same directory as this script.")
    exit()
    
print("\nFirst 5 rows of the dataset:")
print(data.head())

# ===========================================
# 3️⃣ Basic Data Cleaning
# ===========================================
print("\nChecking for missing values...")
print(data.isnull().sum())

data = data.dropna()
print("Missing values handled.")

# ===========================================
# 4️⃣ Feature Selection
# ===========================================
feature_cols = ['Area', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
target_col = 'Item'

missing_cols = [col for col in feature_cols + [target_col] if col not in data.columns]
if missing_cols:
    print(f"❌ Error: Missing columns in the dataset: {missing_cols}")
    print("Available columns:", data.columns.tolist())
    raise KeyError(f"Missing columns in the dataset: {missing_cols}")


X = data[feature_cols]
y = data[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nCrop types encoded to numerical labels.")

# ===========================================
# 5️⃣ Data Visualization (light version)
# ===========================================
if len(data) > 1:
    plt.figure(figsize=(8,5))
    numeric_data = data.select_dtypes(include=np.number)
    if not numeric_data.empty:
      sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu')
      plt.title("🌡 Feature Correlation Heatmap")
      plt.show()
    else:
      print("\n Skipping heatmap: No numeric data to compute correlation.")
else:
    print("\n Skipping heatmap: Not enough data to compute correlation.")


if len(y) > 0:
    plt.figure(figsize=(10,6))
    sns.countplot(y=y, order = y.value_counts().index)
    plt.title("🌾 Distribution of Crop Types")
    plt.show()
else:
    print("\n Skipping crop distribution plot: No crop data available.")


# ===========================================
# 6️⃣ Split + Scale Data
# ===========================================
if len(data) > 1:
    X_categorical = X[['Area']]
    X_numerical = X[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]

    X_categorical_encoded = pd.get_dummies(X_categorical, columns=['Area'], prefix='Area')

    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=X_numerical.columns, index=X_numerical.index)

    X_processed = pd.concat([X_categorical_encoded, X_numerical_scaled], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
    print("\nData split and preprocessed successfully.")
else:
    print("\n Skipping data splitting and scaling: Not enough data.")


# ===========================================
# 7️⃣ Train Model (Random Forest Classifier)
# ===========================================
if 'X_train' in locals() and 'y_train' in locals() and len(X_train) > 0:
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    print("\nModel trained successfully.")
else:
    print("\n Skipping model training: Not enough training data.")


# ===========================================
# 8️⃣ Predict + Evaluate
# ===========================================
if 'rf' in locals() and 'X_test' in locals() and 'y_test' in locals() and len(X_test) > 0:
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"Accuracy Score: {accuracy:.2f}")
else:
    print("\n Skipping model prediction and evaluation: Model not trained or not enough test data.")


# ===========================================
# 9️⃣ Feature Importance
# ===========================================
if 'rf' in locals() and len(X_processed.columns) > 0:
    feat_imp = pd.Series(rf.feature_importances_, index=X_processed.columns)
    feat_imp.nlargest(10).sort_values().plot(kind='barh', color='limegreen')
    plt.title(" Top 10 Feature Importance in Crop Prediction")
    plt.show()
else:
    print("\n Skipping feature importance plot: Model not trained or no features defined.")

# ===========================================
# 🔟 Predict New Data from User Input
# ===========================================
if 'rf' in locals() and 'scaler' in locals() and 'le' in locals():
    print("\n🔮 Ready to predict new data. Please provide the following details:")
    
    try:
        input_area = input("Enter Area (e.g., India): ")
        input_rainfall = float(input("Enter average rainfall (mm per year): "))
        input_pesticides = float(input("Enter pesticides use (tonnes): "))
        input_temp = float(input("Enter average temperature (Celsius): "))

        sample_data = {
            'Area': [input_area],
            'average_rain_fall_mm_per_year': [input_rainfall],
            'pesticides_tonnes': [input_pesticides],
            'avg_temp': [input_temp]
        }
        sample_df = pd.DataFrame(sample_data)

        sample_encoded = pd.get_dummies(sample_df, columns=['Area'], prefix='Area')
        
        missing_cols_sample = set(X_train.columns) - set(sample_encoded.columns)
        for c in missing_cols_sample:
            sample_encoded[c] = 0
            
        sample_processed = sample_encoded[X_train.columns]
        
        pred_class_encoded = rf.predict(sample_processed)[0]
        pred_crop = le.inverse_transform([pred_class_encoded])[0]

        print(f"\n Predicted Crop for your input: {pred_crop}")
        
    except ValueError:
        print("\n Invalid input. Please ensure you enter numerical values for rainfall, pesticides, and temperature.")
else:
    print("\n Skipping sample prediction: Model or scaler not available.")