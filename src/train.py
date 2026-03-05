import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_model():
    # 1. Load Data
    df = pd.read_csv('data/final_training_set.csv')
    # Remove any weird index columns if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 2. Split Features and Target
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # 3. Train/Test Split (80% training, 20% evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} games...")

    # 4. Initialize XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 5. Fit Model
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Performance ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # 7. Save Model
    joblib.dump(model, 'models/nba_model.joblib')
    print("\nModel saved to models/nba_model.joblib")

if __name__ == "__main__":
    train_model()