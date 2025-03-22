import pandas as pd
import numpy as np
import joblib

def load_and_prepare_data(file_path):
    """
    Load and prepare new data for prediction
    """
    # Load the new features
    df = pd.read_csv(file_path, index_col=0)
    
    # Load the scaler
    scaler = joblib.load('scaler.joblib')
    
    # Scale the features
    X_scaled = scaler.transform(df)
    
    return X_scaled, df.index

def predict(model, data):
    """
    Make predictions using the trained SVM model
    
    Args:
        model: The trained SVM model
        data: Scaled input data
    """
    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    
    return predictions, probabilities[:, 1]  # Return probability of positive class

def main():
    # Load the trained model
    print("Loading trained SVM model...")
    model = joblib.load('adhd_classifier.joblib')
    
    # Load and prepare new data
    print("Loading new features...")
    X_scaled, file_names = load_and_prepare_data('features.csv')
    
    # Print data shape for debugging
    print(f"Input data shape: {X_scaled.shape}")
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict(model, X_scaled)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'File': file_names,
        'Prediction': predictions,
        'Probability': probabilities
    })
    
    # Add interpretation
    results_df['Interpretation'] = results_df['Prediction'].map({1: 'ADHD', 0: 'Non-ADHD'})
    
    # Save results
    results_df.to_csv('prediction_results.csv')
    print("\nResults saved to 'prediction_results.csv'")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total samples: {len(predictions)}")
    print(f"ADHD predictions: {sum(predictions)}")
    print(f"Non-ADHD predictions: {len(predictions) - sum(predictions)}")
    
    # Print detailed results
    print("\nDetailed Results:")
    for _, row in results_df.iterrows():
        print(f"\nFile: {row['File']}")
        print(f"Prediction: {row['Interpretation']}")
        print(f"Probability: {row['Probability']:.4f}")

if __name__ == "__main__":
    main() 