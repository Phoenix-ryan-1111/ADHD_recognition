import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import opensmile
import librosa
import soundfile as sf

def get_feature_names():
    """
    Get feature names directly from opensmile
    
    Returns:
        list: List of feature names
    """
    # Initialize eGeMAPs feature extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=16000
    )
    
    # Get feature names from opensmile
    return smile.feature_names

def resample_audio(audio_file, target_sr=16000):
    """
    Resample audio file to target sampling rate
    
    Args:
        audio_file (str): Path to the audio file
        target_sr (int): Target sampling rate (default 16000 Hz)
        
    Returns:
        tuple: (resampled audio data, target sampling rate)
    """
    # Load audio file
    y, sr = librosa.load(audio_file)
    
    # Resample if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    return y, target_sr

def extract_egemaps(audio_file):
    """
    Extract eGeMAPs features from an audio file
    
    Args:
        audio_file (str): Path to the audio file
        
    Returns:
        numpy.ndarray: Array of eGeMAPs features
    """
    # Initialize eGeMAPs feature extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=16000
    )
    
    # Extract features
    features = smile.process_file(audio_file)
    return features.values[0]

# def get_label(filename):
#     """
#     Get label based on filename (1 for ADHD, 0 for non-ADHD)
    
#     Args:
#         filename (str): Name of the audio file
        
#     Returns:
#         int: Label (1 for ADHD, 0 for non-ADHD)
#     """
#     return 1 if 'adhd' in filename.lower() else 0

def process_audio_directory(input_dir, output_file):
    """
    Process all audio files in a directory and extract eGeMAPs features
    
    Args:
        input_dir (str): Directory containing audio files
        output_file (str): Path to save the features CSV file
        
    Returns:
        tuple: (DataFrame with features, list of labels)
    """
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.wav'))]
    
    # Initialize lists to store features, labels, and file names
    all_features = []
    # all_labels = []
    file_names = []
    
    # Process each audio file
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        file_path = os.path.join(input_dir, audio_file)
        
        try:
            # Extract eGeMAPs features
            features = extract_egemaps(file_path)
            # label = get_label(audio_file)
            
            all_features.append(features)
            # all_labels.append(label)
            file_names.append(audio_file)
            print(f"Successfully processed: {audio_file}")
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    if not all_features:
        raise ValueError("No features were successfully extracted from any files")
    
    # Get feature names from opensmile
    feature_names = get_feature_names()
    
    # Convert to DataFrame with named columns
    df = pd.DataFrame(all_features, index=file_names, columns=feature_names)
    
    # Add label column to the DataFrame
    # df['label'] = all_labels
    
    # Save features to CSV
    df.to_csv(output_file)
    print(f"\nFeatures saved to: {output_file}")
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    # print(f"ADHD samples: {sum(all_labels)}")
    # print(f"Non-ADHD samples: {len(all_labels) - sum(all_labels)}")
    
    return df

def main():
    # Specify your directories and parameters
    input_dir = r"dataset\predict_16k"  # Directory containing audio files
    features_file = "predict_feature.csv"  # Where to save the features
    
    try:
        # Extract eGeMAPs features and get labels
        print("Extracting eGeMAPs features...")
        features_df, labels = process_audio_directory(input_dir, features_file)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()