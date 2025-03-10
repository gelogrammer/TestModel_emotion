"""
Evaluation Script for Speech Emotion Recognition Model

This script evaluates the performance of the trained emotion recognition model.
It provides real-time emotion prediction and calculates performance metrics.
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.audio_processing import record_audio, extract_audio_features, remove_silence, plot_waveform
from src.rl_agent import EmotionDQNAgent, EMOTIONS
from src.train import get_feature_vector

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def real_time_evaluation(agent, num_samples=10, duration=5):
    """
    Evaluate the model in real-time with immediate feedback.
    
    Args:
        agent (EmotionDQNAgent): The trained agent
        num_samples (int): Number of test samples
        duration (int): Duration of each recording in seconds
        
    Returns:
        tuple: (y_true, y_pred) for performance metrics calculation
    """
    y_true = []
    y_pred = []
    
    print(f"Real-time evaluation with {num_samples} samples...")
    
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}")
        print("Please express any emotion from the list:")
        print(", ".join(EMOTIONS))
        print(f"Recording will start in 3 seconds...")
        time.sleep(3)
        
        # Record audio
        filename = f"eval_sample_{i}"
        audio = record_audio(duration=duration, filename=filename)
        
        # Process audio
        start_time = time.time()
        feature_vector = get_feature_vector(audio, 16000)
        
        # Predict emotion
        action = agent.select_action(feature_vector, training=False)
        predicted_emotion = EMOTIONS[action]
        processing_time = time.time() - start_time
        
        # Display result
        print(f"Predicted emotion: {predicted_emotion.upper()}")
        print(f"Processing time: {processing_time:.3f} seconds")
        
        # Ask for ground truth
        print("What was the actual emotion you expressed?")
        for idx, emotion in enumerate(EMOTIONS):
            print(f"{idx}: {emotion}")
        
        try:
            actual_emotion_index = int(input("Enter the correct emotion index: "))
            if actual_emotion_index < 0 or actual_emotion_index >= len(EMOTIONS):
                print("Invalid input. Skipping this sample...")
                continue
        except ValueError:
            print("Invalid input. Skipping this sample...")
            continue
        
        actual_emotion = EMOTIONS[actual_emotion_index]
        print(f"Actual emotion: {actual_emotion.upper()}")
        
        # Record results
        y_true.append(actual_emotion_index)
        y_pred.append(action)
        
        # Pause between recordings
        time.sleep(1)
    
    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    """
    Calculate and display performance metrics.
    
    Args:
        y_true (list): Ground truth emotion indices
        y_pred (list): Predicted emotion indices
    """
    if not y_true:
        print("No evaluation data available.")
        return
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=EMOTIONS)
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    cm_file = os.path.join(DATA_DIR, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(cm_file)
    plt.show()
    print(f"Confusion matrix saved to {cm_file}")
    
    return accuracy


def continuous_emotion_monitoring(agent, duration=60, interval=1.0):
    """
    Monitor emotions continuously for a period of time.
    
    Args:
        agent (EmotionDQNAgent): The trained agent
        duration (int): Total duration in seconds
        interval (float): Prediction interval in seconds
    """
    print(f"Continuous emotion monitoring for {duration} seconds...")
    print("Speak continuously and the system will predict your emotions.")
    print("Press Ctrl+C to stop early.")
    
    # Initialize variables
    emotions_detected = []
    timestamps = []
    
    try:
        # Start recording
        print("Recording started...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Record a short audio segment
            audio = record_audio(duration=interval, filename=None)
            
            # Process audio
            feature_vector = get_feature_vector(audio, 16000)
            
            # Predict emotion
            action = agent.select_action(feature_vector, training=False)
            predicted_emotion = EMOTIONS[action]
            
            # Store result
            current_time = time.time() - start_time
            emotions_detected.append(predicted_emotion)
            timestamps.append(current_time)
            
            # Display result
            print(f"[{current_time:.1f}s] Detected emotion: {predicted_emotion.upper()}")
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    # Plot emotion timeline
    plt.figure(figsize=(12, 6))
    
    # Create numeric values for emotions for plotting
    emotion_indices = [EMOTIONS.index(emotion) for emotion in emotions_detected]
    
    plt.scatter(timestamps, emotion_indices, c=emotion_indices, cmap='viridis', s=100)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Timeline')
    plt.grid(True, axis='y')
    
    # Save timeline
    timeline_file = os.path.join(DATA_DIR, f"emotion_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(timeline_file)
    plt.show()
    print(f"Emotion timeline saved to {timeline_file}")


def analyze_speech_rate(agent, audio_file=None, duration=10):
    """
    Analyze speech rate over time.
    
    Args:
        agent (EmotionDQNAgent): The trained agent
        audio_file (str, optional): Path to audio file
        duration (int): Duration to record if no file provided
    """
    from src.audio_processing import load_audio, calculate_speech_rate
    
    if audio_file and os.path.exists(audio_file):
        # Load audio file
        audio, sample_rate = load_audio(audio_file)
        print(f"Loaded audio file: {audio_file}")
    else:
        # Record audio
        print(f"Recording {duration} seconds of speech for analysis...")
        print("Please speak continuously at different rates.")
        print("Recording will start in 3 seconds...")
        time.sleep(3)
        audio = record_audio(duration=duration, filename="speech_rate_analysis")
        sample_rate = 16000
    
    # Analyze speech rate over time
    window_size = int(1.0 * sample_rate)  # 1-second window
    hop_length = int(0.5 * sample_rate)   # 0.5-second hop
    
    speech_rates = []
    timestamps = []
    emotions = []
    
    for i in range(0, len(audio) - window_size, hop_length):
        # Extract window
        window = audio[i:i+window_size]
        
        # Calculate speech rate
        speech_rate = calculate_speech_rate(window, sample_rate)
        
        # Extract features and predict emotion
        feature_vector = get_feature_vector(window, sample_rate)
        action = agent.select_action(feature_vector, training=False)
        emotion = EMOTIONS[action]
        
        # Store results
        timestamp = i / sample_rate
        speech_rates.append(speech_rate)
        timestamps.append(timestamp)
        emotions.append(emotion)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot speech rate
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, speech_rates, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speech Rate (syllables/second)')
    plt.title('Speech Rate Over Time')
    plt.grid(True)
    
    # Plot emotions
    plt.subplot(2, 1, 2)
    emotion_indices = [EMOTIONS.index(emotion) for emotion in emotions]
    plt.scatter(timestamps, emotion_indices, c=emotion_indices, cmap='viridis', s=100)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Over Time')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save analysis
    analysis_file = os.path.join(DATA_DIR, f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(analysis_file)
    plt.show()
    print(f"Speech analysis saved to {analysis_file}")


def main():
    """Main function to execute the evaluation process."""
    parser = argparse.ArgumentParser(description='Evaluate speech emotion recognition model.')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'monitor', 'analyze'], default='evaluate',
                        help='Evaluation mode (evaluate, monitor, or analyze)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--duration', type=int, default=5, help='Duration of each recording in seconds')
    parser.add_argument('--audio_file', type=str, help='Path to audio file for analysis')
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # Initialize agent
    input_dim = 40  # Based on our feature extraction
    agent = EmotionDQNAgent(input_dim=input_dim)
    
    # Load model
    if not agent.load_model(args.model):
        print("Failed to load model.")
        return
    
    # Run evaluation
    if args.mode == 'evaluate':
        y_true, y_pred = real_time_evaluation(agent, num_samples=args.samples, duration=args.duration)
        calculate_metrics(y_true, y_pred)
    elif args.mode == 'monitor':
        continuous_emotion_monitoring(agent, duration=args.duration, interval=1.0)
    elif args.mode == 'analyze':
        analyze_speech_rate(agent, audio_file=args.audio_file, duration=args.duration)


if __name__ == "__main__":
    main() 