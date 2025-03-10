"""
Audio Processing Module for Speech Emotion Recognition

This module contains functions for recording, preprocessing, and analyzing speech audio.
Optimized for Python 3.11 compatibility.
"""

import os
import numpy as np
import librosa
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
SAMPLE_RATE = 16000  # 16 kHz, standard for speech processing
FRAME_DURATION_MS = 30  # 30ms frames for VAD
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'audio')

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)


def record_audio(duration=5, sample_rate=SAMPLE_RATE, filename=None):
    """
    Record audio from the microphone.
    
    Args:
        duration (float): Duration of recording in seconds
        sample_rate (int): Sample rate in Hz
        filename (str, optional): If provided, save audio to this file
        
    Returns:
        numpy.ndarray: Recorded audio samples
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = audio.flatten()  # Convert to mono
    
    if filename:
        if not filename.endswith('.wav'):
            filename += '.wav'
        filepath = os.path.join(AUDIO_DIR, filename)
        wavfile.write(filepath, sample_rate, audio)
        print(f"Audio saved to {filepath}")
    
    return audio


def load_audio(filepath, target_sr=SAMPLE_RATE):
    """
    Load audio file and resample if needed.
    
    Args:
        filepath (str): Path to audio file
        target_sr (int): Target sample rate
        
    Returns:
        tuple: (audio_samples, sample_rate)
    """
    audio, sr = librosa.load(filepath, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def remove_silence(audio, sample_rate=SAMPLE_RATE, frame_duration_ms=FRAME_DURATION_MS, aggressive=3):
    """
    Remove silence from audio using energy-based VAD.
    Optimized for Python 3.11 compatibility.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        frame_duration_ms (int): Frame duration in milliseconds
        aggressive (int): VAD aggressiveness (0-3)
        
    Returns:
        numpy.ndarray: Audio with silence removed
    """
    return energy_based_vad(audio, sample_rate, frame_duration_ms, aggressive)


def energy_based_vad(audio, sample_rate=SAMPLE_RATE, frame_duration_ms=FRAME_DURATION_MS, aggressive=3):
    """
    Python 3.11 compatible energy-based voice activity detection.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        frame_duration_ms (int): Frame duration in milliseconds
        aggressive (int): Controls the threshold (0-3, higher = more aggressive)
        
    Returns:
        numpy.ndarray: Audio with silence removed
    """
    # Ensure audio is in the right format
    if len(audio) == 0:
        return np.array([])
    
    # Convert audio to float if not already
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    
    # Calculate frame parameters
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    hop_length = frame_length  # Non-overlapping frames
    
    # Calculate the number of frames
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    if num_frames <= 0:
        return audio  # Audio too short, return as is
    
    # Initialize arrays for frames and their energies
    energies = np.zeros(num_frames)
    
    # Calculate energy for each frame
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]
        # RMS energy
        energies[i] = np.sqrt(np.mean(frame**2))
    
    # Adjust threshold based on aggressive level (0-3)
    # Higher aggressive value = more frames considered silence
    percentile_threshold = 85 - 15 * (aggressive / 3)  # 85% to 70% based on aggressive level
    threshold = np.percentile(energies, percentile_threshold) if len(energies) > 0 else 0
    
    # List of voice frames
    voiced_frames = []
    
    for i in range(num_frames):
        if energies[i] > threshold:
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            voiced_frames.append(audio[start:end])
    
    # If no voice frames detected, return a small portion of the original audio
    if not voiced_frames:
        if len(audio) > frame_length:
            # Return the frame with highest energy if no frames passed the threshold
            max_energy_frame = np.argmax(energies)
            start = max_energy_frame * hop_length
            end = min(start + frame_length, len(audio))
            return audio[start:end]
        return audio
    
    # Concatenate voiced frames
    return np.concatenate(voiced_frames)


def calculate_speech_rate(audio, sample_rate=SAMPLE_RATE):
    """
    Estimate speech rate in syllables per second.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        
    Returns:
        float: Estimated syllables per second
    """
    # Use energy peaks to estimate syllables
    energy = np.square(audio)
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    threshold = energy_mean + 0.5 * energy_std
    
    # Find peaks above threshold
    peaks = np.where(energy > threshold)[0]
    if len(peaks) == 0:
        return 0
    
    # Count syllables from peaks with reasonable spacing
    min_spacing_samples = int(0.1 * sample_rate)  # Minimum 100ms between syllables
    syllable_count = 1
    last_peak = peaks[0]
    
    for peak in peaks[1:]:
        if peak - last_peak > min_spacing_samples:
            syllable_count += 1
            last_peak = peak
    
    duration = len(audio) / sample_rate
    syllables_per_second = syllable_count / duration
    
    return syllables_per_second


def plot_waveform(audio, sample_rate=SAMPLE_RATE, title="Audio Waveform"):
    """
    Plot audio waveform.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        title (str): Plot title
    """
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_audio_features(audio, sample_rate=SAMPLE_RATE, n_mfcc=13, n_mels=40, n_fft=512, hop_length=256):
    """
    Extract audio features for emotion recognition.
    
    Args:
        audio (numpy.ndarray): Audio samples
        sample_rate (int): Sample rate in Hz
        n_mfcc (int): Number of MFCCs to extract
        n_mels (int): Number of Mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        
    Returns:
        dict: Dictionary of audio features
    """
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Pitch (F0) estimation using more Python 3.11 compatible approach
    try:
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), 
                                       fmax=librosa.note_to_hz('C7'), sr=sample_rate)
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            features['f0_mean'] = np.mean(f0)
            features['f0_std'] = np.std(f0)
            features['f0_min'] = np.min(f0)
            features['f0_max'] = np.max(f0)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
    except Exception as e:
        print(f"Warning: Failed to extract pitch features: {e}")
        features['f0_mean'] = 0
        features['f0_std'] = 0
        features['f0_min'] = 0
        features['f0_max'] = 0
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Speech rate
    features['speech_rate'] = calculate_speech_rate(audio, sample_rate)
    
    return features


if __name__ == "__main__":
    # Example usage
    print("Testing audio processing module (Python 3.11 compatible)...")
    
    # Record a short audio sample
    test_audio = record_audio(duration=3, filename="test_recording")
    
    # Plot the waveform
    plot_waveform(test_audio, title="Test Recording")
    
    # Remove silence
    print("Removing silence...")
    no_silence = remove_silence(test_audio)
    
    # Extract features
    print("Extracting features...")
    features = extract_audio_features(no_silence)
    
    # Print features
    print("\nExtracted Features:")
    for feature, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"{feature}: shape={value.shape}")
        else:
            print(f"{feature}: {value:.4f}") 