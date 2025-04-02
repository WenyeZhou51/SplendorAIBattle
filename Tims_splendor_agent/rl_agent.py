import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from splendor_env import SplendorEnv

# Define the Actor-Critic network model with card-specific processing
class ActorCritic(nn.Module):
    def __init__(self, input_dim=2300, output_dim=100):
        super(ActorCritic, self).__init__()
        
        # Define indices for different parts of the state vector
        # These need to be adjusted based on your actual state representation
        self.card_feature_start = 50     # Estimated start index for card features
        self.card_feature_end = 600      # Estimated end index for card features
        self.card_feature_dim = self.card_feature_end - self.card_feature_start
        
        # Card-specific processing pathway
        self.card_encoder = nn.Sequential(
            nn.Linear(self.card_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Main feature pathway for non-card features
        self.main_encoder = nn.Sequential(
            nn.Linear(input_dim - self.card_feature_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Merge pathways and continue with shared layers
        self.shared = nn.Sequential(
            nn.Linear(128 + 256, 384),  # Combined dimensions from both pathways
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Linear(128, output_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Handle single state case (during action selection)
        original_dim = x.dim()
        if original_dim == 1:
            x = x.unsqueeze(0)  # Add batch dimension for BatchNorm
        
        # Extract card features and other features
        card_features = x[:, self.card_feature_start:self.card_feature_end]
        
        # Create a mask to select non-card features (all indices except card features)
        feature_indices = list(range(0, self.card_feature_start)) + list(range(self.card_feature_end, x.shape[1]))
        other_features = x[:, feature_indices]
        
        # Handling training vs evaluation mode for BatchNorm
        if x.size(0) == 1:  # If batch size is 1 (inference/evaluation mode)
            # Set eval mode for BatchNorm
            self.card_encoder.eval()
            self.main_encoder.eval()
            self.shared.eval()
            
            with torch.no_grad():
                # Process card features through card-specific pathway
                card_encoded = self.card_encoder(card_features)
                
                # Process other features through main pathway
                other_encoded = self.main_encoder(other_features)
                
                # Concatenate the outputs from both pathways
                combined = torch.cat([card_encoded, other_encoded], dim=1)
                
                # Continue with shared layers
                features = self.shared(combined)
            
            # Set back to training mode if necessary
            if self.training:
                self.card_encoder.train()
                self.main_encoder.train()
                self.shared.train()
        else:
            # Normal forward pass with batch size > 1 (training mode)
            # Process card features through card-specific pathway
            card_encoded = self.card_encoder(card_features)
            
            # Process other features through main pathway
            other_encoded = self.main_encoder(other_features)
            
            # Concatenate the outputs from both pathways
            combined = torch.cat([card_encoded, other_encoded], dim=1)
            
            # Continue with shared layers
            features = self.shared(combined)
        
        # Get action probabilities from actor
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Get state value from critic
        value = self.critic(features)
        
        # Remove batch dimension if it was added
        if original_dim == 1:
            value = value.squeeze(0)
            action_probs = action_probs.squeeze(0)
        
        return action_probs, value

class PPOAgent:
    def __init__(
        self,
        input_dim=2300,
        output_dim=100,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        epochs=4,
        device=None
    ):
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize network
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        
        # Training storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.masks = []
    
    def get_action(self, state, valid_moves_mask=None):
        """
        Choose an action based on the current state using the policy network.
        
        Args:
            state: Current environment state
            valid_moves_mask: Mask of valid moves (1 for valid, 0 for invalid)
            
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        
        # Apply mask for valid moves
        if valid_moves_mask is not None:
            mask = torch.zeros_like(action_probs)
            for i in range(min(len(valid_moves_mask), len(mask))):
                if i < len(valid_moves_mask) and valid_moves_mask[i]:
                    mask[i] = 1.0
            
            # Zero out invalid actions
            masked_probs = action_probs * mask
            
            # Renormalize probabilities (if any valid moves have non-zero probability)
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # If all valid moves have zero probability, use uniform distribution over valid moves
                masked_probs = mask / mask.sum() if mask.sum() > 0 else action_probs
        else:
            masked_probs = action_probs
        
        # Create a distribution and sample
        dist = Categorical(masked_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)
    
    def remember(self, state, action, reward, done, log_prob, mask):
        """Store episode data for learning"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
    
    def clear_memory(self):
        """Clear episode data after learning"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.masks = []
    
    def calculate_returns(self):
        """Calculate cumulative discounted rewards"""
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns
    
    def update(self):
        """Update policy using PPO algorithm"""
        # Convert episode data to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        masks = [torch.FloatTensor(m).to(self.device) if m is not None else None for m in self.masks]
        
        # Calculate returns
        returns = self.calculate_returns()
        
        # Learning loop
        for _ in range(self.epochs):
            for i in range(len(states)):
                # Get current action probabilities and state value
                action_probs, state_value = self.policy(states[i])
                
                # Apply mask if available
                if masks[i] is not None:
                    mask = masks[i]
                    masked_probs = action_probs * mask
                    if masked_probs.sum() > 0:
                        masked_probs = masked_probs / masked_probs.sum()
                    else:
                        masked_probs = mask / mask.sum() if mask.sum() > 0 else action_probs
                else:
                    masked_probs = action_probs
                
                # Calculate entropy (for exploration)
                dist = Categorical(masked_probs)
                entropy = dist.entropy()
                
                # Get log probability of the taken action
                new_log_prob = dist.log_prob(actions[i])
                
                # Calculate the ratio between new and old policies
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                
                # Calculate surrogate losses
                advantage = returns[i] - state_value.detach()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                
                # Calculate final loss (negative because we're minimizing)
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = 0.5 * (returns[i] - state_value) ** 2
                entropy_loss = -0.01 * entropy  # Encourage exploration
                
                loss = actor_loss + critic_loss + entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear memory after update
        self.clear_memory()
    
    def save_model(self, path="models/splendor_agent.pt"):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        print(f"Model loaded from {path}")

def train(num_episodes=1000, save_every=100, model_path="models/splendor_agent.pt", verbose=True, track_metrics=False):
    """
    Train the PPO agent to play Splendor
    
    Args:
        num_episodes: Number of episodes to train for
        save_every: How often to save the model
        model_path: Path to save the model
        verbose: Whether to print detailed training information
        track_metrics: Whether to track and save additional metrics like win rates
    """
    # Initialize environment and agent
    env = SplendorEnv(num_players=2)
    agent = PPOAgent()
    
    # Training metrics
    all_rewards = []
    avg_rewards = []
    episode_lengths = []
    state_errors = 0
    
    # Zero-sum game specific metrics
    if track_metrics:
        player_wins = [0, 0]  # Track wins per player [player 0, player 1]
        win_rates = []        # Track win rate of player 0 over time
        tie_games = 0         # Track games that end in a tie
    
    for episode in range(1, num_episodes+1):
        # Reset environment
        state, info = env.reset()
        
        # Get mask of valid moves
        valid_moves = list(env.valid_moves_mapping.keys())
        valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
        
        total_reward = 0
        done = False
        steps = 0
        
        # Episode loop
        while not done:
            # Select action
            action, log_prob = agent.get_action(state, valid_moves_mask)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Check if there was a state verification error but don't log it
            if 'state_verification_error' in info and info['state_verification_error']:
                state_errors += 1
            
            # Store in memory
            agent.remember(state, action, reward, done, log_prob.item(), valid_moves_mask)
            
            # Update state and valid moves mask
            state = next_state
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            total_reward += reward
            steps += 1
            
            # Update policy if episode is done
            if done:
                agent.update()
                
                # Track win/loss for zero-sum analysis
                if track_metrics and terminated:
                    scores = info['scores']
                    max_score = max(scores)
                    winners = [i for i, score in enumerate(scores) if score == max_score]
                    
                    if len(winners) > 1:
                        # It's a tie
                        tie_games += 1
                    else:
                        # Record the winner
                        player_wins[winners[0]] += 1
        
        # Record metrics
        all_rewards.append(total_reward)
        episode_lengths.append(steps)
        avg_reward = sum(all_rewards[-100:]) / min(len(all_rewards), 100)
        avg_rewards.append(avg_reward)
        
        # Track win rate
        if track_metrics:
            total_games = player_wins[0] + player_wins[1] + tie_games
            win_rate = player_wins[0] / total_games if total_games > 0 else 0.5
            win_rates.append(win_rate)
        
        # Print progress - modified to print every 5 episodes with concise format
        if episode % 5 == 0:
            if track_metrics:
                print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Win Rate P0: {win_rate:.2f}")
            else:
                print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        # Save model
        if episode % save_every == 0:
            agent.save_model(model_path)
            
            # Plot learning curves
            plt.figure(figsize=(15, 10))
            
            # Plot rewards
            plt.subplot(2, 1, 1)
            plt.plot(all_rewards, alpha=0.3, label='Rewards')
            plt.plot(avg_rewards, label='Avg Rewards (100 ep)')
            plt.title(f'PPO Training Progress - Episode {episode}')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            
            # Plot win rate (for zero-sum game analysis)
            if track_metrics:
                plt.subplot(2, 1, 2)
                plt.plot(win_rates, label='Player 0 Win Rate', color='green')
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Policy (50%)')
                plt.title('Win Rate Analysis')
                plt.xlabel('Episode')
                plt.ylabel('Win Rate (Player 0)')
                plt.ylim(0, 1)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"models/training_curve_{episode}.png")
            plt.close()
    
    # Save final model
    agent.save_model(model_path)
    
    # Print final statistics
    if track_metrics:
        total_games = player_wins[0] + player_wins[1] + tie_games
        win_rate = player_wins[0] / total_games if total_games > 0 else 0.5
        print(f"Training complete - {num_episodes} episodes")
        print(f"Final Avg Reward: {avg_rewards[-1]:.2f}, Total State Errors: {state_errors}")
        print(f"Win Rate: Player 0: {win_rate:.2f} ({player_wins[0]}/{total_games}), " +
              f"Player 1: {1-win_rate:.2f} ({player_wins[1]}/{total_games}), " +
              f"Ties: {tie_games}/{total_games}")
    else:
        print(f"Training complete - {num_episodes} episodes, Final Avg Reward: {avg_rewards[-1]:.2f}, Total State Errors: {state_errors}")
    
    # Return the trained agent and metrics
    return agent, all_rewards, avg_rewards, episode_lengths

def evaluate(model_path="models/splendor_agent.pt", num_episodes=100):
    """
    Evaluate the trained agent
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        avg_reward: Average reward across all evaluation episodes
        win_rate: Percentage of games won
    """
    # Initialize environment and agent
    env = SplendorEnv(num_players=2)
    agent = PPOAgent()
    agent.load_model(model_path)
    
    # Evaluation metrics
    total_rewards = []
    wins = 0
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        
        total_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get valid moves mask
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            # Select action
            action, _ = agent.get_action(state, valid_moves_mask)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            # Check if game ended with a win
            if terminated and env.game_state.current_player_index == 0 and env.game_state.players[0].score >= 15:
                wins += 1
        
        total_rewards.append(total_reward)
        
        if episode % 5 == 0:
            print(f"Evaluation episode {episode}/{num_episodes}, Reward: {total_reward:.2f}")
    
    avg_reward = sum(total_rewards) / num_episodes
    win_rate = wins / num_episodes * 100
    
    print(f"Evaluation complete over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    return avg_reward, win_rate

if __name__ == "__main__":
    # Train the agent
    train(num_episodes=1000, save_every=100)
    
    # Evaluate the agent
    evaluate() 