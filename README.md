# ADHD Voice Analysis

This project analyzes voice characteristics to detect ADHD using machine learning techniques. It processes audio files, extracts eGeMAPs features, and uses a neural network for classification.

## Features

- Audio file processing and feature extraction
- eGeMAPs feature extraction using OpenSMILE
- PCA dimensionality reduction
- Neural network classification using PyTorch
- Support for both training and prediction

## Project Structure

```
.
├── dataset/              # Directory for audio files (not included in repo)
├── neural_network.py     # Neural network model implementation
├── predict.py           # Script for making predictions
├── create_train_test_data.py  # Script for data preparation
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adhd-voice-analysis.git
cd adhd-voice-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your audio files in the `dataset/adhd` directory
2. Run the data preparation script:
```bash
python create_train_test_data.py
```

### Training

1. Run the neural network training script:
```bash
python neural_network.py
```

### Prediction

1. Place new audio files in the appropriate directory
2. Run the prediction script:
```bash
python predict.py
```

## Dependencies

- Python 3.8+
- PyTorch
- librosa
- soundfile
- opensmile
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 