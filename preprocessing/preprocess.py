# preprocessing/preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path):
    """
    Load dataset, preprocess, split into train/test, scale features.
    Returns: X_train, X_test, y_train, y_test
    Also saves a combined preprocessed CSV for reference.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop ID column if exists
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    
    # Fill missing  values 
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    
    # Encode categorical columns
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # Separate features and target
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Create combined DataFrame for saving
    train_df = X_train_scaled.copy()
    train_df["stroke"] = y_train
    train_df["split"] = "train"
    
    test_df = X_test_scaled.copy()
    test_df["stroke"] = y_test
    test_df["split"] = "test"
    
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # Make results folder if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save preprocessed data
    preprocessed_file = os.path.join(results_dir, "preprocessed_data.csv")
    combined_df.to_csv(preprocessed_file, index=False)
    
    # Print info
    print("=== Preprocessing Output ===")
    print("Full dataframe shape:", combined_df.shape)
    print("Train/Test distribution:\n", combined_df['split'].value_counts())
    print("\nFirst 5 rows:\n", combined_df.head())
    print(f"\nPreprocessed data saved to: {preprocessed_file}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test
