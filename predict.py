import joblib
from create_predict_data import process_audio_files
def predict_adhd(features_df):
    """
    Predict ADHD from features DataFrame
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        
    Returns:
        dict: Dictionary containing prediction results
            {
                'prediction': 'ADHD' or 'Non-ADHD',
                'probability': float (0-1),
                'percentage': float (0-100),
                'message': str
            }
    """
    try:
        # Load the trained model and scaler
        model = joblib.load('adhd_classifier.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Scale the features
        X_scaled = scaler.transform(features_df)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        print(probabilities)
        # Calculate average probability for ADHD
        avg_probability = probabilities[:, 1].mean()
        
        # Determine final prediction
        final_prediction = 1 if avg_probability >= 0.5 else 0
        
        # Prepare result
        result = {
            'success':True,
            'prediction': f"prediction: {'ADHD' if final_prediction == 1 else 'Non-ADHD'}",
            'probability': f"Probability of ADHD: {avg_probability:.2%}",
            'percentage': float(avg_probability * 100),
        }
        
        return result
        
    except Exception as e:
        return {
            'prediction': 'Error',
            'probability': 0.0,
            'percentage': 0.0,
            'message': f'Error processing features: {str(e)}'
        }

def main():
    # Example usage
    # Process audio file
    features_df = process_audio_files(
        input_file="dataset/predict_non_ADHD.mp3"
    )
    # Make prediction
    result = predict_adhd(features_df)
    
    print("\nPrediction Results:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Percentage: {result['percentage']:.2f}%")
    print(f"Message: {result['message']}")

if __name__ == "__main__":
    main() 