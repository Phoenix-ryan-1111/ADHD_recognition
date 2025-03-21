import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from neural_network import ADHDClassifier, ADHDDataset

def load_and_prepare_data(file_path):
    """
    Load and prepare new data for prediction
    """
    # Load the new features
    df = pd.read_csv(file_path, index_col=0)
    
    # Scale the features using the same scaler as training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return X_scaled, df.index

def predict(model, data, device):
    """
    Make predictions using the trained model
    
    Args:
        model: The trained PyTorch model
        data: DataLoader containing the input data
        device: The device to run the model on (CPU/GPU)
    """
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, _ in data:  # Unpack the tuple from DataLoader
            inputs = inputs.to(device)  # Move inputs to the correct device
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            
            # Handle both single and batch predictions
            if len(probs.shape) == 0:  # Single prediction
                probs = np.array([probs])
            
            # Convert probabilities to predictions
            batch_predictions = (probs > 0.5).astype(int)
            
            # Append results
            predictions.extend(batch_predictions.tolist())
            probabilities.extend(probs.tolist())
    
    return predictions, probabilities

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare new data
    print("Loading new features...")
    X_scaled, file_names = load_and_prepare_data('pca_predict_results.csv')
    
    # Print data shape for debugging
    print(f"Input data shape: {X_scaled.shape}")
    
    # Create model with correct input dimensions
    input_dim = X_scaled.shape[1]
    model = ADHDClassifier(input_dim=input_dim).to(device)
    
    # Load the trained model
    try:
        model.load_state_dict(torch.load('adhd_classifier.pth'))
        print(f"Model loaded successfully with input dimension: {input_dim}")
    except RuntimeError as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure the input data has the same number of features as the training data")
        return
    
    # Create dataset and dataloader for prediction
    dataset = ADHDDataset(X_scaled, np.zeros(len(X_scaled)))  # Dummy labels
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict(model, dataloader, device)
    
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