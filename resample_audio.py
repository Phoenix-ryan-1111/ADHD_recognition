import os
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_audio(input_file, output_file, target_sr=16000):
    """
    Resample audio file to target sampling rate
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the resampled audio file
        target_sr (int): Target sampling rate (default 16000 Hz)
    """
    # Load audio file
    y, sr = librosa.load(input_file)
    
    # Resample if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        print(f"Resampled from {sr}Hz to {target_sr}Hz")
    
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

def main():
    # Specify your directories
    input_dir = r"dataset\train"  # Directory containing original audio files
    output_dir = r"dataset\train_16k"  # Directory to save resampled files
    target_sr = 16000  # Target sampling rate
    
    try:
        print(f"Starting resampling process to {target_sr}Hz...")
        process_directory(input_dir, output_dir, target_sr)
        print("\nResampling completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 