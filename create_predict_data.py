import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import opensmile
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def split_audio(input_file, output_dir, segment_length_seconds=60):
    """
    Split an audio file into segments of specified length
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save the split audio files
        segment_length_seconds (int): Length of each segment in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    print(f"Loading audio file: {input_file}")
    y, sr = librosa.load(input_file)
    
    # Calculate segment length in samples
    segment_length_samples = int(segment_length_seconds * sr)
    
    # Calculate number of segments
    total_segments = len(y) // segment_length_samples + (1 if len(y) % segment_length_samples != 0 else 0)
    
    print(f"Total duration: {len(y)/sr:.2f} seconds")
    print(f"Number of segments: {total_segments}")
    
    # Get the file extension
    file_extension = os.path.splitext(input_file)[1].lower()
    
    # Split and save segments
    for i in tqdm(range(total_segments), desc="Splitting audio"):
        start_sample = i * segment_length_samples
        end_sample = min((i + 1) * segment_length_samples, len(y))
        
        # Extract segment
        segment = y[start_sample:end_sample]
        # Generate output filename
        output_filename = f"segment_{i+1:03d}{file_extension}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Export segment
        sf.write(output_path, segment, sr)

def resample_audio(input_file, output_file, target_sr=16000):
    """
    Resample audio file to target sampling rate
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the resampled audio file
        target_sr (int): Target sampling rate
    """
    # Load audio file
    y, sr = librosa.load(input_file)
    
    # Resample if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Save resampled audio
    sf.write(output_file, y, target_sr)

def process_directory(input_dir, output_dir, target_sr=16000):
    """
    Process all audio files in a directory and resample them
    
    Args:
        input_dir (str): Directory containing audio files
        output_dir (str): Directory to save resampled files
        target_sr (int): Target sampling rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.wav'))]
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Resampling audio files"):
        input_path = os.path.join(input_dir, audio_file)
        output_path = os.path.join(output_dir, audio_file)
        
        try:
            resample_audio(input_path, output_path, target_sr)
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")

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

def process_audio_files(input_file, output_dir, segment_length=60):
    """
    Process audio file: split, resample, and extract features
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Base directory for outputs
        segment_length (int): Length of each segment in seconds
    """
    # Create necessary directories
    split_dir = os.path.join(output_dir, 'split')
    resampled_dir = os.path.join(output_dir, 'resampled')
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(resampled_dir, exist_ok=True)
    
    # Step 1: Split audio
    print("\nStep 1: Splitting audio file...")
    split_audio(input_file, split_dir, segment_length)
    
    # Step 2: Resample segments
    print("\nStep 2: Resampling segments to 16kHz...")
    process_directory(split_dir, resampled_dir, target_sr=16000)
    
    # Step 3: Extract features
    print("\nStep 3: Extracting eGeMAPs features...")
    all_features = []
    file_names = []
    
    audio_files = [f for f in os.listdir(resampled_dir) if f.endswith(('.mp3', '.wav'))]
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        file_path = os.path.join(resampled_dir, audio_file)
        try:
            features = extract_egemaps(file_path)
            all_features.append(features)
            file_names.append(audio_file)
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Create DataFrame with features
    feature_names = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=16000
    ).feature_names
    
    df = pd.DataFrame(all_features, index=file_names, columns=feature_names)
    
    # Save features
    features_file = os.path.join(output_dir, 'features.csv')
    df.to_csv(features_file)
    print(f"\nFeatures saved to: {features_file}")
    
    return df

def perform_pca(features_df, n_components=None):
    """
    Perform PCA on the features with manual input of number of components
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        n_components (int, optional): Number of components to keep. If None, will ask for input.
        
    Returns:
        tuple: (PCA object, transformed features, explained variance ratio)
    """
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # If n_components is not provided, ask for input
    if n_components is None:
        max_components = min(features_df.shape[0], features_df.shape[1])
        print(f"\nMaximum possible components: {max_components}")
        while True:
            try:
                n_components = int(input("Enter the number of components to keep: "))
                if 1 <= n_components <= max_components:
                    break
                else:
                    print(f"Please enter a number between 1 and {max_components}")
            except ValueError:
                print("Please enter a valid number")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)
    
    # Print explained variance information
    print("\nPCA Results:")
    print(f"Number of components: {n_components}")
    print("\nExplained variance ratio for each component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_, 1):
        print(f"PC{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    
    print(f"\nCumulative explained variance: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    return pca, pca_result, pca.explained_variance_ratio_

def plot_pca_results(pca_result, explained_variance_ratio, output_dir):
    """
    Plot PCA results including explained variance and component visualization
    
    Args:
        pca_result (np.ndarray): PCA transformed features
        explained_variance_ratio (np.ndarray): Explained variance ratio for each component
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.title('Explained Variance Ratio by Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.savefig(os.path.join(plots_dir, 'explained_predict_variance.png'))
    plt.close()
    
    # Plot 2: Cumulative explained variance
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cumulative_predict_variance.png'))
    plt.close()
    
    # Plot 3: First two components scatter plot (if at least 2 components)
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title('First Two Principal Components')
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'pca_scatter.png'))
        plt.close()

def main():
    # Specify your input and output paths
    input_file = r"dataset\predict.mp3"  # Change this to your input file path
    output_dir = r"dataset\processed"  # Change this to your desired output directory
    segment_length = 60  # Length of each segment in seconds
    
    try:
        # Process audio files
        print("\nProcessing audio files...")
        features_df = process_audio_files(input_file, output_dir, segment_length)
        
        print("\nProcessing completed successfully!")
        print(f"Total segments processed: {len(features_df)}")
        print(f"Number of features extracted: {len(features_df.columns)}")
        
        # Perform PCA
        print("\nPerforming PCA analysis...")
        pca, pca_result, explained_variance_ratio = perform_pca(features_df)
        
        # Create PCA results DataFrame
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
            index=features_df.index
        )
        
        # Save PCA results
        pca_file = os.path.join(output_dir, 'pca_predict_results.csv')
        pca_df.to_csv(pca_file)
        print(f"\nPCA results saved to: {pca_file}")
        
        # Plot PCA results
        print("\nGenerating PCA plots...")
        plot_pca_results(pca_result, explained_variance_ratio, output_dir)
        print(f"Plots saved to: {os.path.join(output_dir, 'plots')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 