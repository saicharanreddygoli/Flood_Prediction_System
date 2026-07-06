import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_pipeline():
    if not os.path.exists("Dataset/Historical_Flood_Data.csv"):
        raise FileNotFoundError("Dataset missing. Please run Dataset/generate_data.py first.")
        
    df = pd.read_csv("Dataset/Historical_Flood_Data.csv")
    
    X = df[["Annual_Rainfall", "Cloud_Visibility", "Seasonal_Rainfall"]]
    y = df["Flood_Occurred"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f"XGBoost Model trained with Test Accuracy: {accuracy * 100:.2f}%")
    
    os.makedirs("Saved_Models", exist_ok=True)
    with open("Saved_Models/XGBoost_Model.pkl", "wb") as m_file:
        pickle.dump(model, m_file)
    with open("Saved_Models/Preprocessor_Scaler.pkl", "wb") as s_file:
        pickle.dump(scaler, s_file)
        
    print("Model and Scaler successfully saved to 'Saved_Models/'!")

if __name__ == "__main__":
    train_pipeline()
