import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, brier_score_loss

def run_backtest():
    # 1. Load the model and the final training set
    model = joblib.load('models/nba_model.joblib')
    df = pd.read_csv('data/final_training_set.csv')
    
    # 2. Split Features and Target
    # This automatically includes CORE_INJURY_DIFF because it's in the CSV now
    X = df.drop(columns=['TARGET'])
    y_true = df['TARGET']
    
    # Safety check: Ensure X has the exact columns the model expects
    # (Sometimes pandas adds an 'Unnamed: 0' index column when saving)
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    
    # 3. Generate Predictions and Probabilities
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    # 4. Calculate Simulation Metrics
    results = pd.DataFrame({
        'Actual': y_true,
        'Predicted': preds,
        'Probability': probs
    })
    
    # 5. Simulate a $10 Flat Betting Strategy
    def calculate_profit(row):
        bet_amount = 10
        if row['Predicted'] == 1: # Model predicted Home Win
            if row['Actual'] == 1:
                return bet_amount * 0.91 # Win $9.10 (Standard -110 odds)
            else:
                return -bet_amount # Lose $10.00
        return 0 

    results['Profit'] = results.apply(calculate_profit, axis=1)
    
    # 6. Output Results
    print("--- Backtest Results (Season to Date) ---")
    print(f"Total Games Analyzed: {len(results)}")
    print(f"Model Accuracy: {accuracy_score(y_true, preds):.2%}")
    print(f"Total Profit/Loss: ${results['Profit'].sum():.2f}")
    
    # Calculate ROI based only on games where a bet was placed
    total_invested = results[results['Predicted'] == 1].count()['Profit'] * 10
    roi = (results['Profit'].sum() / total_invested) if total_invested > 0 else 0
    print(f"Return on Investment (ROI): {roi:.2%}")
    
    # Brier Score: Measures how close probabilities are to the truth (Lower is better)
    print(f"Brier Score: {brier_score_loss(y_true, probs):.4f}")

if __name__ == "__main__":
    run_backtest()