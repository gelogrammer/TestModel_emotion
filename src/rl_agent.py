"""
Reinforcement Learning Agent for Speech Emotion Recognition

This module implements a reinforcement learning agent that learns to
predict emotions from speech features using feedback-based learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque, namedtuple
import json
from datetime import datetime

# Define emotion classes
EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a named tuple for experience replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class EmotionQNetwork(nn.Module):
    """
    Q-Network for emotion recognition using speech features.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=len(EMOTIONS)):
        """
        Initialize the Q-Network.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Number of output actions (emotions)
        """
        super(EmotionQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Q-values for each action (emotion)
        """
        return self.network(x)


class EmotionDQNAgent:
    """
    Deep Q-Network agent for emotion recognition.
    """
    
    def __init__(self, input_dim, hidden_dim=128, learning_rate=0.001, 
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update_freq=10):
        """
        Initialize the DQN agent.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            epsilon (float): Exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            memory_size (int): Size of replay memory
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network update
        """
        self.input_dim = input_dim
        self.output_dim = len(EMOTIONS)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q-Networks
        self.q_network = EmotionQNetwork(input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_network = EmotionQNetwork(input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize training parameters
        self.updates = 0
        self.losses = []
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))
        
        # Emotion history for tracking improvement
        self.emotion_history = {emotion: {'correct': 0, 'total': 0} for emotion in EMOTIONS}
    
    def add_to_memory(self, state, action, reward, next_state, done):
        """
        Add experience to replay memory.
        
        Args:
            state: Current state (feature vector)
            action (int): Action taken (emotion index)
            reward (float): Reward received
            next_state: Next state (feature vector)
            done (bool): Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state (feature vector)
            training (bool): Whether to use exploration or not
            
        Returns:
            int: Selected action (emotion index)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randrange(self.output_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()
    
    def train_step(self):
        """
        Perform a single training step using experience replay.
        
        Returns:
            float: Loss value
        """
        # Check if enough samples in memory
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random minibatch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch for training
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in batch]).to(self.device)
        
        # Compute Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Log loss
        self.losses.append(loss.item())
        self.writer.add_scalar('Loss/train', loss.item(), self.updates)
        self.writer.add_scalar('Epsilon', self.epsilon, self.updates)
        
        return loss.item()
    
    def update_emotion_history(self, predicted_emotion, actual_emotion):
        """
        Update emotion prediction history.
        
        Args:
            predicted_emotion (int): Predicted emotion index
            actual_emotion (int): Actual emotion index
        """
        predicted_label = EMOTIONS[predicted_emotion]
        actual_label = EMOTIONS[actual_emotion]
        
        self.emotion_history[actual_label]['total'] += 1
        if predicted_emotion == actual_emotion:
            self.emotion_history[actual_label]['correct'] += 1
        
        # Log accuracy
        for emotion in EMOTIONS:
            stats = self.emotion_history[emotion]
            accuracy = stats['correct'] / max(1, stats['total'])
            self.writer.add_scalar(f'Accuracy/{emotion}', accuracy, self.updates)
    
    def get_emotion_accuracies(self):
        """
        Get emotion prediction accuracies.
        
        Returns:
            dict: Emotion accuracies
        """
        accuracies = {}
        for emotion in EMOTIONS:
            stats = self.emotion_history[emotion]
            accuracies[emotion] = stats['correct'] / max(1, stats['total'])
        return accuracies
    
    def save_model(self, filename=None):
        """
        Save the model to disk.
        
        Args:
            filename (str, optional): Filename to save model to
        """
        if filename is None:
            filename = f"emotion_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        filepath = os.path.join(MODELS_DIR, filename)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'updates': self.updates,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'emotion_history': self.emotion_history
        }, filepath)
        
        print(f"Model saved to {filepath}")
        
        # Save configuration
        config_filepath = os.path.join(MODELS_DIR, f"{os.path.splitext(filename)[0]}_config.json")
        config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'emotions': EMOTIONS
        }
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_model(self, filepath):
        """
        Load the model from disk.
        
        Args:
            filepath (str): Path to model file
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if model architecture matches
        if 'input_dim' in checkpoint and checkpoint['input_dim'] != self.input_dim:
            print(f"Model input dimension mismatch: expected {self.input_dim}, got {checkpoint['input_dim']}")
            return False
        
        # Load model parameters
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.updates = checkpoint['updates']
        
        if 'emotion_history' in checkpoint:
            self.emotion_history = checkpoint['emotion_history']
        
        print(f"Model loaded from {filepath}")
        return True


# Example usage
if __name__ == "__main__":
    # Simple test
    input_dim = 40  # Example: 40 features from audio processing
    agent = EmotionDQNAgent(input_dim=input_dim)
    
    # Test action selection
    state = np.random.rand(input_dim)
    action = agent.select_action(state)
    print(f"Selected action (emotion): {EMOTIONS[action]}")
    
    # Test training step
    for i in range(10):
        state = np.random.rand(input_dim)
        action = agent.select_action(state)
        reward = random.uniform(-1, 1)
        next_state = np.random.rand(input_dim)
        done = random.random() > 0.8
        
        agent.add_to_memory(state, action, reward, next_state, done)
    
    if len(agent.memory) >= agent.batch_size:
        loss = agent.train_step()
        print(f"Training loss: {loss:.4f}")
    
    # Save the model
    agent.save_model("test_model.pt") 