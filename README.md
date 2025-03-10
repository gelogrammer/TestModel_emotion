# Real-Time Speech Rate and Emotion Feedback Using Deep Reinforcement Learning

This repository contains the implementation of a speech emotion recognition system that uses deep reinforcement learning to provide real-time feedback on speech rate and emotional content. The system is designed to improve over time through continuous interaction and feedback.

## Project Overview

The system uses reinforcement learning to learn from feedback, allowing it to adapt specifically to your voice and speaking patterns without requiring large pre-labeled datasets. It analyzes speech in real-time, extracting acoustic features and providing immediate feedback on both emotional content and speech rate.

## Key Features

- Real-time speech emotion recognition
- Speech rate analysis
- Self-improvement through reinforcement learning
- No pre-labeled dataset required
- Visualization tools for tracking progress
- Multi-modal reinforcement learning capabilities

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended for faster training)
- Microphone for audio input

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speech-emotion-rl.git
   cd speech-emotion-rl
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/raw data/processed models logs
   ```

### Package Requirements

The project requires the following main packages (see requirements.txt for full details):

```
# Core libraries
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Deep Learning
torch>=1.10.0
torchaudio>=0.10.0
transformers>=4.15.0

# Audio processing
librosa>=0.8.1
sounddevice>=0.4.4
pyaudio>=0.2.11
webrtcvad>=2.0.10

# Machine Learning & Reinforcement Learning
scikit-learn>=1.0.0
tensorboard>=2.7.0
gym>=0.21.0
stable-baselines3>=1.4.0

# Visualization
plotly>=5.3.0
streamlit>=1.2.0
```

### Windows-specific installation notes

On Windows, you might encounter issues installing some packages:

1. For PyAudio:
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

2. For webrtcvad, you may need Microsoft Visual C++ Build Tools:
   - Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Then install webrtcvad: `pip install webrtcvad`

3. For torch with CUDA support:
   ```bash
   # Check the PyTorch website for the command specific to your CUDA version
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Audio Setup

Ensure your microphone is properly configured:

1. On Windows: Set your microphone as the default recording device in Sound settings
2. On macOS: Allow microphone access in System Preferences > Security & Privacy
3. On Linux: Check your microphone is recognized with `arecord -l`

## Usage

### Training the Model

To start training the model from scratch:

```bash
python src/train.py --mode both --samples 20 --epochs 50 --duration 5
```

This will:
1. Collect 20 initial labeled samples (you'll be prompted to express specific emotions)
2. Train the model for 50 epochs on these samples
3. Begin continuous learning mode where you provide feedback on the model's predictions
4. Save the trained model to the `models/` directory (typically as `emotion_model_latest.pt`)

### Options:

- `--mode`: Choose between `initial` (bootstrap training only), `continuous` (feedback-based training only), or `both`
- `--samples`: Number of initial samples to collect
- `--epochs`: Number of training epochs for the initial training phase
- `--duration`: Duration of each recording in seconds
- `--model`: Path to a pre-trained model to start from (optional)

### Evaluating the Model

Before evaluating, ensure you have trained the model first:

```bash
# First train a model
python src/train.py --mode initial --samples 10 --epochs 30 --duration 5
```

This will create a model file in the models/ directory. Then you can evaluate it:

```bash
# Use the correct path to your trained model
python src/evaluate.py --model models/emotion_model_latest.pt --mode evaluate --samples 10
```

If you encounter a "Model file not found" error, check that:
1. You've successfully completed the training step
2. The model filename matches exactly what was generated during training
3. You're running the command from the project root directory

### Modes:

- `evaluate`: Test the model on a set number of samples and calculate performance metrics
- `monitor`: Continuously monitor emotions for a specified duration
- `analyze`: Analyze speech rate over time, either from a recorded file or in real-time

### Real-Time Feedback Application

Once your model is trained, you can run the real-time feedback application:

```bash
python src/run_feedback.py
```

## Project Structure

```
.
├── data/                      # Directory for storing audio samples and training data
├── models/                    # Saved model checkpoints
├── src/
│   ├── audio_processing.py    # Audio preprocessing functions
│   ├── rl_agent.py            # Reinforcement learning agent
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── notebooks/                 # Jupyter notebooks for exploration and visualization
├── logs/                      # Training logs and tensorboard files
├── INSTRUCTIONS.md            # Detailed instructions for model training
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## How It Works

1. **Audio Processing**: 
   - Captures raw audio from microphone
   - Removes silence and normalizes audio
   - Extracts features using MFCCs, pitch, energy, etc.

2. **Reinforcement Learning**:
   - State: Audio feature vector
   - Actions: Emotion classifications
   - Reward: Feedback on correct/incorrect predictions
   - Agent: Deep Q-Network that learns to map audio features to emotions

3. **Continuous Improvement**:
   - Each prediction gets feedback
   - Model updates its weights based on this feedback
   - Over time, adapts specifically to your voice and expression patterns

## Performance Metrics

The system monitors and visualizes:
- Learning progress over time
- Confusion matrix for emotion predictions
- Speech rate analysis trends
- Feature importance visualization

## Tips for Best Results

1. **Initial Training**: Express emotions clearly and consistently during the initial training phase
2. **Varied Samples**: Provide a diverse range of emotional expressions
3. **Consistent Feedback**: Give accurate feedback for the model to learn effectively
4. **Regular Training**: Train the model regularly to improve its performance
5. **Quiet Environment**: Record in a quiet environment for better audio quality

## Troubleshooting

### Model Not Found Errors
- Ensure you've trained a model before evaluation
- Check that the model path is correct
- Run commands from the project root directory

### TensorFlow Warnings
You may see warnings like:
```
tensorflow/core/util/port.cc:113 oneDNN custom operations are on...
```
These are harmless TensorFlow configuration warnings and can be safely ignored.

### Common Issues
- **Poor accuracy**: Collect more diverse speech samples
- **Slow response time**: Optimize feature extraction pipeline
- **Inconsistent learning**: Adjust reward function and exploration rate
- **Overfitting to specific voice patterns**: Introduce variations in speech samples

## Future Development

1. Expand emotion categories
2. Implement additional speech metrics (clarity, filler words)
3. Develop enhanced user interface for feedback visualization
4. Create API for integration with other applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PyTorch team for the deep learning framework
- librosa developers for audio processing tools
- The research community for advancements in speech emotion recognition 