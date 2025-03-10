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
from src.audio_processing import record_audio, extract_audio_features, remove_silence, calculate_speech_rate, plot_waveform
from src.rl_agent import EmotionDQNAgent, EMOTIONS

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
        return np.zeros(40)
    
    # Extract features
    features = extract_audio_features(audio, sample_rate)
    
    # Flatten MFCC features and combine with other features
    feature_vector = []
    feature_vector.extend(features['mfcc_mean'])
    feature_vector.append(features['spectral_centroid_mean'])
    feature_vector.append(features['spectral_rolloff_mean'])
    feature_vector.append(features['f0_mean'])
    feature_vector.append(features['f0_std'])
    feature_vector.append(features['zcr_mean'])
    feature_vector.append(features['rms_mean'])
    feature_vector.append(features['speech_rate'])
    
    # Ensure consistent feature vector length
    if len(feature_vector) < 40:
        feature_vector.extend([0] * (40 - len(feature_vector)))
    
    # Normalize features
    feature_vector = np.array(feature_vector[:40])  # Limit to 40 features
    
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
    
    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        
        # Shuffle samples
        random.shuffle(samples)
        
        for feature_vector, emotion_index in samples:
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
        
        # Ask for feedback
        print("Was this prediction correct? If not, what was the actual emotion?")
        print("Emotions:", ", ".join(f"{i}: {emotion}" for i, emotion in enumerate(EMOTIONS)))
        
        try:
            actual_emotion_index = int(input("Enter the correct emotion index (or -1 to skip): "))
            if actual_emotion_index < 0 or actual_emotion_index >= len(EMOTIONS):
                print("Skipping this episode...")
                continue
        except ValueError:
            print("Invalid input. Skipping this episode...")
            continue
        
        actual_emotion = EMOTIONS[actual_emotion_index]
        
        # Calculate reward
        reward = 1.0 if action == actual_emotion_index else -1.0
        print(f"Reward: {reward:.1f}")
        
        # Add to memory
        next_state = feature_vector  # Use same state as next state
        done = True
        agent.add_to_memory(feature_vector, actual_emotion_index, reward, next_state, done)
        
        # Train step
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()
            print(f"Training loss: {loss:.4f}")
        
        # Update emotion history
        agent.update_emotion_history(action, actual_emotion_index)
        
        # Calculate accuracy
        emotion_accuracies = agent.get_emotion_accuracies()
        overall_accuracy = sum(emotion_accuracies.values()) / len(emotion_accuracies)
        accuracies.append(overall_accuracy)
        
        print(f"Overall accuracy: {overall_accuracy:.2f}")
        
        # Save checkpoint
        if (episode + 1) % 10 == 0:
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
    input_dim = 40  # Based on our feature extraction
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