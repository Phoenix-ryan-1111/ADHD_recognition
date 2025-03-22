import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_prepare_data():
    """
    Load PCA-reduced data and prepare it for training
    """
    # Load the PCA-reduced dataset
    df = pd.read_csv('train_feature.csv', index_col=0)
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train SVM model
    """
    # Initialize SVM classifier with RBF kernel
    svm_model = SVC(
        kernel='rbf',
        C=1.0,  # Regularization parameter
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    return svm_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and generate visualizations
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return y_pred, y_pred_proba

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"ADHD samples in training: {sum(y_train)}")
    print(f"ADHD samples in testing: {sum(y_test)}")
    
    # Train model
    print("\nTraining SVM model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'adhd_classifier.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model saved as 'adhd_classifier.joblib'")
    print("Scaler saved as 'scaler.joblib'")

if __name__ == "__main__":
    main() 