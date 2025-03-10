"""
Training Script for Speech Emotion Recognition using Reinforcement Learning

This script implements the training loop for the speech emotion recognition model.
It handles data collection, feature extraction, and model training using reinforcement learning.
"""

import os
import sys
import argparse
import numpy as np
import time
import random
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle  # Added this import

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.audio_processing import (
    record_audio, extract_audio_features, remove_silence, 
    calculate_speech_rate, plot_waveform, load_audio
)
from src.rl_agent import EmotionDQNAgent, EMOTIONS
from src.data_augmentation import augment_audio

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_feature_vector(audio, sample_rate):
    """
    Extract a feature vector from audio for model input.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        
    Returns:
        numpy.ndarray: Feature vector
    """
    # Clean audio
    audio = remove_silence(audio, sample_rate)
    if len(audio) == 0:
        # No speech detected, return zeros
        return np.zeros(128)  # Expanded feature vector size
    
    # Extract features
    features = extract_audio_features(audio, sample_rate, n_mfcc=20, n_mels=60)
    
    # Create an expanded feature vector with more emotional cues
    feature_vector = []
    
    # MFCC features (mean and std)
    feature_vector.extend(features['mfcc_mean'])
    feature_vector.extend(features['mfcc_std'])  # Adding standard deviation
    
    # Spectral features
    feature_vector.append(features['spectral_centroid_mean'])
    feature_vector.append(features['spectral_centroid_std'])
    feature_vector.append(features['spectral_rolloff_mean'])
    feature_vector.append(features['spectral_rolloff_std'])
    
    # Pitch features
    feature_vector.append(features['f0_mean'])
    feature_vector.append(features['f0_std'])
    feature_vector.append(features.get('f0_min', 0))  # Min pitch
    feature_vector.append(features.get('f0_max', 0))  # Max pitch
    feature_vector.append(features.get('f0_range', 0))  # Pitch range
    
    # Energy features
    feature_vector.append(features['rms_mean'])
    feature_vector.append(features.get('rms_std', 0))
    feature_vector.append(features.get('rms_max', 0))
    
    # Rhythm features
    feature_vector.append(features['speech_rate'])
    
    # Voice quality features (if available)
    feature_vector.append(features.get('jitter', 0))
    feature_vector.append(features.get('shimmer', 0))
    feature_vector.append(features.get('hnr', 0))  # Harmonics-to-noise ratio
    
    # Zero-crossing rate
    feature_vector.append(features['zcr_mean'])
    feature_vector.append(features.get('zcr_std', 0))
    
    # Spectrogram statistics
    if 'spectral_contrast_mean' in features:
        feature_vector.extend(features['spectral_contrast_mean'])
    
    # Chroma features if available
    if 'chroma_mean' in features:
        feature_vector.extend(features['chroma_mean'])
    
    # Normalize feature vector to have unit variance
    feature_vector = np.array(feature_vector)
    
    # Handle NaN or infinite values that might occur in feature extraction
    feature_vector = np.nan_to_num(feature_vector)
    
    # Pad or truncate to fixed size for model input (128 features)
    if len(feature_vector) < 128:
        feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
    else:
        feature_vector = feature_vector[:128]
    
    # Z-score normalization for numerical stability
    epsilon = 1e-10  # To avoid division by zero
    feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + epsilon)
    
    return feature_vector


def collect_initial_samples(num_samples=20, duration=5):
    """
    Collect initial labeled samples for bootstrapping the model.
    
    Args:
        num_samples (int): Number of samples to collect
        duration (int): Duration of each sample in seconds
        
    Returns:
        list: List of (feature_vector, emotion_index) tuples
    """
    samples = []
    
    print(f"Collecting {num_samples} initial samples...")
    print("For each recording, you'll need to express a specific emotion.")
    
    for i in range(num_samples):
        # Randomly select an emotion to express
        emotion_index = random.randrange(len(EMOTIONS))
        emotion = EMOTIONS[emotion_index]
        
        print(f"\nSample {i+1}/{num_samples}")
        print(f"Please express the emotion: {emotion.upper()}")
        print(f"Recording will start in 3 seconds...")
        time.sleep(3)
        
        # Record audio
        filename = f"initial_sample_{i}_{emotion}"
        audio = record_audio(duration=duration, filename=filename)
        
        # Extract features
        feature_vector = get_feature_vector(audio, 16000)
        
        # Save sample
        samples.append((feature_vector, emotion_index))
        
        # Pause between recordings
        time.sleep(1)
    
    print(f"Collected {len(samples)} initial samples.")
    
    # Save samples to disk using pickle instead of numpy
    samples_file = os.path.join(DATA_DIR, f"initial_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    with open(samples_file, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Samples saved to {samples_file}")
    
    return samples


def initial_training(agent, samples, epochs=50):
    """
    Perform initial training using collected samples.
    
    Args:
        agent (EmotionDQNAgent): The agent to train
        samples (list): List of (feature_vector, emotion_index) tuples
        epochs (int): Number of training epochs
        
    Returns:
        list: Training losses
    """
    print(f"Starting initial training for {epochs} epochs...")
    
    losses = []
    
    # Augment training data to create a more diverse dataset
    print("Augmenting training data...")
    augmented_samples = []
    
    # Check if audio directory exists
    audio_dir = os.path.join(DATA_DIR, 'audio')
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} does not exist. Skipping augmentation.")
    else:
        # Process original audio samples to extract raw audio for augmentation
        audio_files = [f for f in os.listdir(audio_dir) if f.startswith('initial_sample_') and f.endswith('.wav')]
        
        # If we have the original audio files, augment them
        if audio_files:
            print(f"Found {len(audio_files)} original audio files for augmentation")
            for audio_file in tqdm(audio_files, desc="Augmenting samples"):
                # Extract emotion index from filename
                parts = audio_file.split('_')
                if len(parts) >= 3:
                    try:
                        # Try to parse emotion from filename
                        emotion = parts[-1].replace('.wav', '')
                        emotion_index = EMOTIONS.index(emotion)
                        
                        # Load audio
                        filepath = os.path.join(audio_dir, audio_file)
                        audio, sr = load_audio(filepath)
                        
                        # Create 3 augmented versions of each sample
                        augmented_audios = augment_audio(audio, sr, num_augmentations=3)
                        
                        # Extract features from each augmented audio
                        for aug_audio in augmented_audios:
                            feature_vector = get_feature_vector(aug_audio, sr)
                            augmented_samples.append((feature_vector, emotion_index))
                    except (ValueError, IndexError):
                        print(f"Couldn't parse emotion from {audio_file}, skipping")
        else:
            print("No initial sample audio files found. Generating augmented samples from existing feature vectors.")
            # If no audio files, create synthetic variations of the existing feature vectors
            for feature_vector, emotion_index in samples:
                # Create 2 synthetic feature variations by adding small random noise
                for _ in range(2):
                    # Add small random noise to the feature vector (5% variation)
                    noise = np.random.normal(0, 0.05 * np.abs(feature_vector).mean(), feature_vector.shape)
                    augmented_feature = feature_vector + noise
                    augmented_samples.append((augmented_feature, emotion_index))
    
    # Combine original and augmented samples
    all_samples = samples + augmented_samples
    print(f"Training with {len(samples)} original samples + {len(augmented_samples)} augmented samples = {len(all_samples)} total samples")
    
    # Ensure balanced class distribution by duplicating underrepresented classes
    emotion_counts = {emotion_idx: 0 for emotion_idx in range(len(EMOTIONS))}
    for _, emotion_idx in all_samples:
        emotion_counts[emotion_idx] += 1
    
    # Find the emotion with the most samples (to balance towards)
    max_count = max(emotion_counts.values())
    
    # Duplicate samples from underrepresented classes
    balanced_samples = all_samples.copy()
    
    for emotion_idx, count in emotion_counts.items():
        if count < max_count:
            # Find all samples of this emotion
            emotion_samples = [(feat, emo_idx) for feat, emo_idx in all_samples if emo_idx == emotion_idx]
            
            # Duplicate these samples to reach max_count (or close to it)
            duplications_needed = max_count - count
            
            # If we need more duplications than available samples, we'll loop through multiple times
            for _ in range(duplications_needed):
                # Add a randomly selected sample of this emotion
                if emotion_samples:
                    balanced_samples.append(random.choice(emotion_samples))
    
    # Final training samples
    training_samples = balanced_samples
    print(f"After balancing: {len(training_samples)} total training samples")
    
    # Print class distribution
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    for _, emotion_idx in training_samples:
        emotion_counts[EMOTIONS[emotion_idx]] += 1
    
    print("Class distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion.upper()}: {count} samples ({count/len(training_samples)*100:.1f}%)")
    
    # Train with balanced samples
    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        
        # Shuffle samples
        random.shuffle(training_samples)
        
        for feature_vector, emotion_index in training_samples:
            # Create initial state
            state = feature_vector
            
            # Select action (emotion prediction)
            action = agent.select_action(state)
            
            # Calculate reward based on correctness
            reward = 1.0 if action == emotion_index else -1.0
            
            # Use same state as next_state for simplicity
            next_state = state
            done = True
            
            # Add to memory
            agent.add_to_memory(state, emotion_index, reward, next_state, done)
            
            # Train step
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                epoch_losses.append(loss)
            
            # Update emotion history
            agent.update_emotion_history(action, emotion_index)
        
        # Calculate mean loss for the epoch
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            losses.append(mean_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")
    
    print("Initial training completed.")
    
    # Save the model
    agent.save_model("emotion_model_initial.pt")
    
    return losses


def continuous_learning(agent, duration=30, num_episodes=50):
    """
    Perform continuous learning with real-time feedback.
    
    Args:
        agent (EmotionDQNAgent): The agent to train
        duration (int): Duration of each recording in seconds
        num_episodes (int): Number of episodes (recordings) to train on
        
    Returns:
        list: Training accuracies
    """
    print(f"Starting continuous learning for {num_episodes} episodes...")
    
    accuracies = []
    losses = []
    
    # Track emotion distribution to ensure balanced training
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Record audio
        filename = f"continuous_sample_{episode}"
        audio = record_audio(duration=duration, filename=filename)
        
        # Extract features
        feature_vector = get_feature_vector(audio, 16000)
        
        # Predict emotion
        action = agent.select_action(feature_vector, training=False)
        predicted_emotion = EMOTIONS[action]
        
        print(f"Predicted emotion: {predicted_emotion.upper()}")
        
        # Get emotion statistics
        print("Current emotion distribution:")
        total_samples = sum(emotion_counts.values()) or 1  # Avoid division by zero
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {emotion.upper()}: {count} samples ({percentage:.1f}%)")
        
        # Ask for feedback
        print("\nWas this prediction correct? If not, what was the actual emotion?")
        print("Emotions:", ", ".join(f"{i}: {emotion.upper()}" for i, emotion in enumerate(EMOTIONS)))
        print("Enter the correct emotion index, or -1 to skip:")
        
        try:
            actual_emotion_index = int(input("Enter the correct emotion index (or -1 to skip): "))
            if actual_emotion_index < 0 or actual_emotion_index >= len(EMOTIONS):
                print("Skipping this episode...")
                continue
        except ValueError:
            print("Invalid input. Skipping this episode...")
            continue
        
        actual_emotion = EMOTIONS[actual_emotion_index]
        emotion_counts[actual_emotion] += 1
        
        # Calculate reward with more nuanced scoring
        # Base reward for correct prediction
        correct_prediction = (action == actual_emotion_index)
        
        # Scale rewards to emphasize learning underrepresented emotions
        emotion_frequency = emotion_counts[actual_emotion] / sum(emotion_counts.values())
        rarity_bonus = 1.0 - emotion_frequency  # Bonus for rare emotions
        
        # Calculate final reward
        if correct_prediction:
            reward = 1.0 + (rarity_bonus * 0.5)  # Higher reward for rare emotions
        else:
            # Smaller penalty for incorrect more common emotions
            reward = -1.0 * (1.0 - (rarity_bonus * 0.3))
        
        print(f"Reward: {reward:.2f} (correct: {correct_prediction}, rarity bonus: {rarity_bonus:.2f})")
        
        # Add to memory
        next_state = feature_vector  # Use same state as next state
        done = True
        agent.add_to_memory(feature_vector, actual_emotion_index, reward, next_state, done)
        
        # Train multiple steps to learn better from this example
        if len(agent.memory) >= agent.batch_size:
            # Train multiple steps for better learning
            total_loss = 0
            num_train_steps = 3  # Train more on each sample for better learning
            for _ in range(num_train_steps):
                loss = agent.train_step()
                total_loss += loss
            avg_loss = total_loss / num_train_steps
            losses.append(avg_loss)
            print(f"Training loss: {avg_loss:.4f}")
        
        # Update emotion history
        agent.update_emotion_history(action, actual_emotion_index)
        
        # Calculate accuracy
        emotion_accuracies = agent.get_emotion_accuracies()
        overall_accuracy = sum(emotion_accuracies.values()) / len(emotion_accuracies)
        accuracies.append(overall_accuracy)
        
        print(f"Overall accuracy: {overall_accuracy:.2f}")
        print("Per-emotion accuracies:")
        for emotion, accuracy in emotion_accuracies.items():
            print(f"  {emotion.upper()}: {accuracy:.2f}")
        
        # Save checkpoint
        if (episode + 1) % 5 == 0 or episode == num_episodes - 1:
            agent.save_model(f"emotion_model_continuous_{episode+1}.pt")
    
    print("Continuous learning completed.")
    
    # Save the final model
    agent.save_model("emotion_model_final.pt")
    
    return accuracies


def plot_training_metrics(losses=None, accuracies=None):
    """
    Plot training metrics.
    
    Args:
        losses (list, optional): List of training losses
        accuracies (list, optional): List of training accuracies
    """
    plt.figure(figsize=(12, 5))
    
    if losses:
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    if accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(DATA_DIR, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Training metrics plot saved to {plot_file}")


def main():
    """Main function to execute the training process."""
    parser = argparse.ArgumentParser(description='Train speech emotion recognition model.')
    parser.add_argument('--mode', type=str, choices=['initial', 'continuous', 'both'], default='both',
                        help='Training mode (initial, continuous, or both)')
    parser.add_argument('--samples', type=int, default=20, help='Number of initial samples to collect')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs for initial training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes for continuous learning')
    parser.add_argument('--duration', type=int, default=5, help='Duration of each recording in seconds')
    parser.add_argument('--model', type=str, help='Path to model file to load (optional)')
    args = parser.parse_args()
    
    # Initialize agent
    input_dim = 128  # Based on our feature extraction
    agent = EmotionDQNAgent(input_dim=input_dim)
    
    # Load model if specified
    if args.model and os.path.exists(args.model):
        agent.load_model(args.model)
    
    losses = None
    accuracies = None
    
    # Initial training
    if args.mode in ['initial', 'both']:
        samples = collect_initial_samples(num_samples=args.samples, duration=args.duration)
        losses = initial_training(agent, samples, epochs=args.epochs)
    
    # Continuous learning
    if args.mode in ['continuous', 'both']:
        accuracies = continuous_learning(agent, duration=args.duration, num_episodes=args.episodes)
    
    # Plot training metrics
    plot_training_metrics(losses, accuracies)


if __name__ == "__main__":
    main() 