import os
import librosa
import soundfile as sf
import numpy as np

def split_audio(input_file, output_dir, segment_length_seconds=60):
    """
    Split an audio file into segments of specified length
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save the split audio files
        segment_length_seconds (int): Length of each segment in seconds (default 60 seconds = 1 minute)
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
    for i in range(total_segments):
        start_sample = i * segment_length_samples
        end_sample = min((i + 1) * segment_length_samples, len(y))
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Generate output filename
        output_filename = f"segment_{i+1:03d}{file_extension}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Export segment
        sf.write(output_path, segment, sr)
        print(f"Saved segment {i+1}/{total_segments}: {output_filename}")

def main():
    # Specify your input and output paths here
    input_file = r"dataset\test.mp3"  # Change this to your input file path
    output_dir = r"dataset\adhd_test"  # Change this to your desired output directory
    segment_length = 60  # Length of each segment in seconds (1 minute)
    
    try:
        split_audio(input_file, output_dir, segment_length)
        print("\nAudio splitting completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 