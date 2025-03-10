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

### Options:

- `--mode`: Choose between `initial` (bootstrap training only), `continuous` (feedback-based training only), or `both`
- `--samples`: Number of initial samples to collect
- `--epochs`: Number of training epochs for the initial training phase
- `--duration`: Duration of each recording in seconds
- `--model`: Path to a pre-trained model to start from (optional)

### Evaluating the Model

To evaluate a trained model:

```bash
python src/evaluate.py --model models/emotion_model_final.pt --mode evaluate --samples 10
```

### Modes:

- `evaluate`: Test the model on a set number of samples and calculate performance metrics
- `monitor`: Continuously monitor emotions for a specified duration
- `analyze`: Analyze speech rate over time, either from a recorded file or in real-time

## Project Structure

```
.
├── data/                      # Directory for storing audio samples and training data
├── models/                    # Saved model checkpoints
├── src/
│   ├── audio_processing.py    # Audio preprocessing functions
│   ├── feature_extraction.py  # Speech feature extraction
│   ├── emotion_model.py       # Emotion classification model
│   ├── rl_agent.py            # Reinforcement learning agent
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
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

## Tips for Best Results

1. **Initial Training**: Express emotions clearly and consistently during the initial training phase
2. **Varied Samples**: Provide a diverse range of emotional expressions
3. **Consistent Feedback**: Give accurate feedback for the model to learn effectively
4. **Regular Training**: Train the model regularly to improve its performance
5. **Quiet Environment**: Record in a quiet environment for better audio quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PyTorch team for the deep learning framework
- librosa developers for audio processing tools
- The research community for advancements in speech emotion recognition 