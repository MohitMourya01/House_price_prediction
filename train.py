import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Configuration ---
MODEL_FILE = 'housing_model.joblib'

# --- Feature Engineering & Model Training ---
def train_model():
    """
    Loads California housing data, trains a linear regression model, 
    and saves it.
    """
    print("Starting model training...")
    
    # 1. Load Data from sklearn
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    print(f"Loaded {len(df)} rows of California housing data.")

    # 2. Define Features (X) and Target (y)
    # Features: 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    # 'Population', 'AveOccup', 'Latitude', 'Longitude'
    # Target: 'MedHouseVal' (Median House Value in $100,000s)
    
    FEATURES = data.feature_names
    TARGET = 'MedHouseVal' # This is the target variable in the dataframe

    X = df[FEATURES]
    y = df[TARGET]

    # 3. Create the Model
    # Since all features are numerical, we don't need the 
    # complex OneHotEncoder/ColumnTransformer pipeline from before.
    # We can just use the LinearRegression model directly.
    # For a more advanced model, you might add a StandardScaler here
    # in a pipeline, but for now, this is the simplest change.
    
    model = LinearRegression()
    
    # 4. Split Data and Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model (Optional, but good practice)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    # Target is in $100,000s, so RMSE is in the same unit.
    print(f"Model training complete. Test RMSE: {rmse:.2f} (in $100,000s)")

    # 6. Save the Model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    train_model()

