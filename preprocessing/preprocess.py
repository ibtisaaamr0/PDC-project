
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    
    # Fill missing BMI values
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    
    # Encode categorical features
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # Features and target
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test
