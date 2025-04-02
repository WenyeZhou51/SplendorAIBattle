import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import deque
import random

# Simple Actor-Critic network for PPO
class ActorCritic(nn.Module):
    def __init__(self, input_dim=2000, output_dim=52, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        shared_features = self.shared(x)
        
        # Get action probabilities
        action_logits = self.actor(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get state value
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
        
    def get_action_and_value(self, x, action_mask=None):
        """Get action, log probability, probability, and value"""
        shared_features = self.shared(x)
        
        # Get action logits
        action_logits = self.actor(shared_features)
        
        # Apply mask if provided
        if action_mask is not None:
            # Set logits of invalid actions to large negative value
            masked_action_logits = action_logits.clone()
            masked_action_logits[~action_mask] = -1e10
            action_probs = F.softmax(masked_action_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
        
        # Create distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        
        # Get action log probability
        action_log_prob = dist.log_prob(action)
        
        # Get state value
        state_value = self.critic(shared_features)
        
        return action, action_log_prob, action_probs, state_value

# Simple PPO Agent
class PPOAgent:
    def __init__(self, state_dim=2000, action_dim=52, hidden_dim=256):
        # Initialize policy network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        
    def select_action(self, state, valid_actions=None):
        """Select an action based on the policy and valid actions"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = self.policy(state_tensor)
            
            # Convert to numpy
            action_probs = action_probs.squeeze().cpu().numpy()
            
            # Filter for valid actions if provided
            if valid_actions is not None:
                # Create mask for valid actions
                valid_mask = np.zeros_like(action_probs)
                valid_mask[valid_actions] = 1
                
                # Apply mask
                valid_action_probs = action_probs * valid_mask
                
                # Normalize probabilities
                if valid_action_probs.sum() > 0:
                    valid_action_probs = valid_action_probs / valid_action_probs.sum()
                else:
                    # Uniform distribution over valid actions if all have zero probability
                    valid_action_probs = np.zeros_like(action_probs)
                    valid_action_probs[valid_actions] = 1.0 / len(valid_actions)
                
                # Choose action based on probabilities
                action = np.random.choice(len(valid_action_probs), p=valid_action_probs)
                log_prob = np.log(valid_action_probs[action] + 1e-10)
            else:
                # Choose the action with highest probability
                action = np.argmax(action_probs)
                log_prob = np.log(action_probs[action] + 1e-10)
            
            return action, log_prob, state_value.item()

class SplendorPPOAgent:
    """
    PPO agent for playing Splendor
    """
    def __init__(self, 
                 input_dim=2300, 
                 hidden_dim=256, 
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 save_dir='models'):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(input_dim, 100, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create save directory
        self.save_dir = save_dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, save_dir)
        os.makedirs(save_path, exist_ok=True)
        
    def select_action(self, state, valid_moves_mask):
        """
        Select an action based on the current state
        
        Args:
            state: Current observation of the environment
            valid_moves_mask: Boolean mask of valid moves
            
        Returns:
            action index, log probability, value
        """
        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        valid_moves_mask_tensor = torch.zeros(1, 100, dtype=torch.bool).to(self.device)
        
        # Set valid moves to True in the mask
        valid_indices = []
        for i in range(len(valid_moves_mask)):
            if valid_moves_mask[i]:
                valid_indices.append(i)
                if i < 100:  # Ensure we don't exceed the action space
                    valid_moves_mask_tensor[0, i] = True
        
        if not valid_indices:
            raise ValueError("No valid moves available")
        
        with torch.no_grad():
            # Get action probabilities and value
            action_probs, value = self.network(state_tensor)
            
            # Apply mask to action probabilities
            masked_probs = action_probs.clone()
            masked_probs = masked_probs * valid_moves_mask_tensor.float()
            
            # Renormalize probabilities if any valid moves have non-zero probability
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # If all valid actions have zero probability, use uniform distribution
                for i in range(len(valid_indices)):
                    if i < 100:  # Ensure we don't exceed the action space
                        masked_probs[0, valid_indices[i]] = 1.0 / len(valid_indices)
        
        # Get only the valid action probabilities
        valid_action_probs = masked_probs[0][valid_moves_mask_tensor[0]].cpu().numpy()
        
        # Choose action based on probabilities
        action_idx = np.random.choice(len(valid_action_probs), p=valid_action_probs)
        log_prob = np.log(valid_action_probs[action_idx] + 1e-10)
        
        return valid_indices[action_idx], log_prob, value.item()
    
    def update(self, rollout_data):
        """
        Update the policy using PPO algorithm
        
        Args:
            rollout_data: Dictionary containing batch data for training
            
        Returns:
            Dictionary of loss metrics
        """
        # Extract data
        states = torch.FloatTensor(rollout_data['states']).to(self.device)
        actions = torch.LongTensor(rollout_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout_data['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device)
        action_masks = torch.BoolTensor(rollout_data['action_masks']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # Calculate new log probs, values, and entropy
        for i in range(len(states)):
            # Get state and action mask for this sample
            state = states[i].unsqueeze(0)
            action_mask = action_masks[i].unsqueeze(0)
            
            # Get action probabilities and value
            action_probs, state_value = self.network(state)
            # Apply mask to valid actions
            if action_mask.shape[0] > 0:  # Ensure mask isn't empty
                masked_probs = action_probs.clone()
                masked_probs = masked_probs * action_mask
                # Re-normalize
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                action_probs = masked_probs
            
            # Create distribution
            dist = Categorical(action_probs)
            
            # Get log prob of the action that was taken
            action_log_prob = dist.log_prob(actions[i])
            
            # Calculate policy loss (PPO clipped objective)
            ratio = torch.exp(action_log_prob - old_log_probs[i])
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages[i]
            policy_loss -= torch.min(ratio * advantages[i], clip_adv)
            
            # Value loss
            value_loss += 0.5 * (state_value - returns[i]) ** 2
            
            # Entropy loss
            entropy = dist.entropy()
            entropy_loss -= entropy
        
        policy_loss /= len(states)
        value_loss /= len(states)
        entropy_loss /= len(states)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the state after the last one
            
        Returns:
            returns and advantages
        """
        returns = []
        advantages = []
        
        gae = 0
        
        # If the last state is terminal, set next_value to 0
        if len(dones) > 0 and dones[-1]:
            next_value = 0
            
        # Append next_value to values for easier calculation
        values = values + [next_value]
        
        # Calculate returns and advantages in reverse order
        for i in reversed(range(len(rewards))):
            # Calculate TD error: r + gamma * V(s') - V(s)
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            
            # Calculate advantage using GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            
            # Insert at the beginning to maintain the right order
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        return returns, advantages
    
    def save(self, episode):
        """Save model checkpoint"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, self.save_dir, f'ppo_checkpoint_{episode}.pt')
        
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path)
        
    def load(self, checkpoint_path):
        """Load model from checkpoint"""
        # If path is relative, make it relative to the current script directory
        if not os.path.isabs(checkpoint_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, checkpoint_path)
            
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 