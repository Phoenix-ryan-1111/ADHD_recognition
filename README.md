# ADHD Voice Analysis Project

This project analyzes voice characteristics of ADHD and non-ADHD individuals using eGeMAPs features and Principal Component Analysis (PCA).

## Project Structure

```
.
├── dataset/               # Audio dataset (not included in repository)
├── egemaps_pca.py        # Main script for eGeMAPs feature extraction and PCA
├── pca.py                # PCA analysis and visualization
├── resample_audio.py     # Audio resampling script
└── requirements.txt      # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your audio dataset in the `dataset` directory:
   - ADHD samples should have "adhd" in their filename
   - Non-ADHD samples should not have "adhd" in their filename

## Usage

1. Resample audio files to 16kHz:
```bash
python resample_audio.py
```

2. Extract eGeMAPs features:
```bash
python egemaps_pca.py
```

3. Perform PCA analysis:
```bash
python pca.py
```

## Output Files

- `egemaps_features.csv`: Raw eGeMAPs features
- `pca_results_reduced.csv`: PCA-transformed features
- `pca_variance_plot.png`: Explained variance plot
- `pca_components_plot.png`: PCA components visualization

## Dependencies

- librosa
- soundfile
- opensmile
- scikit-learn
- numpy
- pandas
- matplotlib

## License

[Your chosen license] 