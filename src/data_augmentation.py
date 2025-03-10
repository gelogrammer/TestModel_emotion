"""
Audio Data Augmentation Module for Speech Emotion Recognition

This module provides functions for augmenting audio data to improve model training.
Augmentation techniques help create more diverse training samples and improve model robustness.
"""

import numpy as np
import librosa
import random
from scipy import signal


def time_stretch(audio, rate_range=(0.8, 1.2)):
    """
    Time stretch the audio signal without changing the pitch.
    
    Args:
        audio (numpy.ndarray): Audio signal
        rate_range (tuple): Range of stretching rates (min_rate, max_rate)
        
    Returns:
        numpy.ndarray: Time-stretched audio
    """
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sample_rate, semitones_range=(-2, 2)):
    """
    Shift the pitch of the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sample_rate (int): Sample rate of the audio
        semitones_range (tuple): Range of semitones to shift (min_semitones, max_semitones)
        
    Returns:
        numpy.ndarray: Pitch-shifted audio
    """
    n_semitones = random.uniform(semitones_range[0], semitones_range[1])
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_semitones)


def add_noise(audio, noise_level_range=(0.001, 0.01)):
    """
    Add random Gaussian noise to the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        noise_level_range (tuple): Range of noise levels (min_level, max_level)
        
    Returns:
        numpy.ndarray: Noisy audio
    """
    noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def apply_filtering(audio, sample_rate, filter_type='lowpass', cutoff_freq=None):
    """
    Apply a filter to the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sample_rate (int): Sample rate of the audio
        filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff_freq (int or tuple): Cutoff frequency or frequencies
        
    Returns:
        numpy.ndarray: Filtered audio
    """
    nyquist = sample_rate / 2.0
    
    if cutoff_freq is None:
        if filter_type == 'lowpass':
            cutoff_freq = random.uniform(0.5, 0.8) * nyquist
        elif filter_type == 'highpass':
            cutoff_freq = random.uniform(0.05, 0.3) * nyquist
        elif filter_type == 'bandpass':
            low = random.uniform(0.1, 0.3) * nyquist
            high = random.uniform(0.6, 0.9) * nyquist
            cutoff_freq = (low, high)
    
    if filter_type == 'lowpass':
        b, a = signal.butter(4, cutoff_freq / nyquist, btype='lowpass')
    elif filter_type == 'highpass':
        b, a = signal.butter(4, cutoff_freq / nyquist, btype='highpass')
    elif filter_type == 'bandpass':
        b, a = signal.butter(4, [f / nyquist for f in cutoff_freq], btype='bandpass')
    else:
        return audio
    
    return signal.filtfilt(b, a, audio)


def time_shift(audio, shift_range=(-0.2, 0.2)):
    """
    Shift the audio in time by rolling.
    
    Args:
        audio (numpy.ndarray): Audio signal
        shift_range (tuple): Range of shift as a fraction of total length
        
    Returns:
        numpy.ndarray: Time-shifted audio
    """
    shift_factor = random.uniform(shift_range[0], shift_range[1])
    shift_amount = int(len(audio) * shift_factor)
    return np.roll(audio, shift_amount)


def augment_audio(audio, sample_rate, augmentation_types=None, num_augmentations=1):
    """
    Apply multiple augmentations to an audio sample.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sample_rate (int): Sample rate of the audio
        augmentation_types (list): List of augmentation types to apply
        num_augmentations (int): Number of augmented samples to generate
        
    Returns:
        list: List of augmented audio samples
    """
    if augmentation_types is None:
        augmentation_types = ['time_stretch', 'pitch_shift', 'add_noise', 'apply_filtering', 'time_shift']
    
    augmented_samples = []
    
    for _ in range(num_augmentations):
        # Apply 1-3 random augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentation_types, num_augs)
        
        # Start with a copy of the original audio
        augmented_audio = audio.copy()
        
        for aug_type in selected_augs:
            if aug_type == 'time_stretch':
                augmented_audio = time_stretch(augmented_audio)
            elif aug_type == 'pitch_shift':
                augmented_audio = pitch_shift(augmented_audio, sample_rate)
            elif aug_type == 'add_noise':
                augmented_audio = add_noise(augmented_audio)
            elif aug_type == 'apply_filtering':
                filter_type = random.choice(['lowpass', 'highpass', 'bandpass'])
                augmented_audio = apply_filtering(augmented_audio, sample_rate, filter_type)
            elif aug_type == 'time_shift':
                augmented_audio = time_shift(augmented_audio)
        
        # Ensure same length as original
        if len(augmented_audio) > len(audio):
            augmented_audio = augmented_audio[:len(audio)]
        elif len(augmented_audio) < len(audio):
            augmented_audio = np.pad(augmented_audio, (0, len(audio) - len(augmented_audio)))
        
        augmented_samples.append(augmented_audio)
    
    return augmented_samples 