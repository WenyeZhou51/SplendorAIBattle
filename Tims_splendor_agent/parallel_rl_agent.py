import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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

class ParallelPPOAgent:
    def __init__(
        self,
        input_dim=2300,
        output_dim=100,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        epochs=4,
        num_workers=4,
        device=None
    ):
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize network
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        # Make sure the model is shared across processes
        self.policy.share_memory()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.num_workers = num_workers
        
        # Training storage for each worker
        self.shared_memory = {i: {"states": [], "actions": [], "rewards": [], 
                              "dones": [], "log_probs": [], "masks": []} 
                              for i in range(num_workers)}
    
    def get_action(self, state, valid_moves_mask=None):
        """Choose an action based on the current state using the policy network."""
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
    
    def remember(self, worker_id, state, action, reward, done, log_prob, mask):
        """Store episode data for learning"""
        self.shared_memory[worker_id]["states"].append(state)
        self.shared_memory[worker_id]["actions"].append(action)
        self.shared_memory[worker_id]["rewards"].append(reward)
        self.shared_memory[worker_id]["dones"].append(done)
        self.shared_memory[worker_id]["log_probs"].append(log_prob)
        self.shared_memory[worker_id]["masks"].append(mask)
    
    def clear_memory(self, worker_id=None):
        """Clear episode data after learning"""
        if worker_id is not None:
            self.shared_memory[worker_id] = {"states": [], "actions": [], "rewards": [], 
                                           "dones": [], "log_probs": [], "masks": []}
        else:
            # Clear all worker memories
            for worker_id in range(self.num_workers):
                self.shared_memory[worker_id] = {"states": [], "actions": [], "rewards": [], 
                                               "dones": [], "log_probs": [], "masks": []}
    
    def calculate_returns(self, rewards, dones):
        """Calculate cumulative discounted rewards"""
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
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
        """Update policy using PPO algorithm with data from all workers"""
        # Combine data from all workers
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_masks = []
        
        for worker_id in range(self.num_workers):
            memory = self.shared_memory[worker_id]
            
            if len(memory["states"]) > 0:  # Only process if worker has collected data
                # Calculate returns for this worker's data
                returns = self.calculate_returns(memory["rewards"], memory["dones"])
                
                # Append this worker's data
                all_states.extend(memory["states"])
                all_actions.extend(memory["actions"])
                all_log_probs.extend(memory["log_probs"])
                all_returns.extend(returns.tolist())
                all_masks.extend(memory["masks"])
        
        # Convert combined data to tensors
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        masks = [torch.FloatTensor(m).to(self.device) if m is not None else None for m in all_masks]
        
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
        
        # Clear memory after update for all workers
        self.clear_memory()
    
    def save_model(self, path="models/parallel_splendor_agent.pt"):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        print(f"Model loaded from {path}")

def worker_process(worker_id, agent, episodes_per_worker, update_interval, barrier, train_stats, 
                   model_path=None, save_interval=None):
    """
    Worker process function for parallel training
    
    Args:
        worker_id: ID of this worker
        agent: Shared PPO agent
        episodes_per_worker: Number of episodes for this worker to run
        update_interval: How often to update the policy (in terms of episodes)
        barrier: Multiprocessing barrier for synchronization
        train_stats: Shared dict to track training progress
        model_path: Path to save intermediate models
        save_interval: How often to save the model (in terms of total episodes across all workers)
    """
    # Initialize environment
    env = SplendorEnv(num_players=2)
    
    # Track worker statistics
    total_rewards = []
    episode_lengths = []
    player_wins = [0, 0]
    tie_games = 0
    
    # Training loop
    for episode in range(episodes_per_worker):
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
            
            # Store in memory
            agent.remember(worker_id, state, action, reward, done, log_prob.item(), valid_moves_mask)
            
            # Update state and valid moves mask
            state = next_state
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            total_reward += reward
            steps += 1
        
        # Track statistics for this worker
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Track win/loss for zero-sum analysis
        if terminated:
            scores = info['scores']
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
            
            if len(winners) > 1:
                # It's a tie
                tie_games += 1
            else:
                # Record the winner
                player_wins[winners[0]] += 1
        
        # Update global training stats
        with train_stats["lock"]:
            train_stats["episodes"] += 1
            train_stats["rewards"].append(total_reward)
            
            # Update win statistics
            train_stats["player0_wins"] += (1 if winners[0] == 0 else 0) if len(winners) == 1 else 0
            train_stats["player1_wins"] += (1 if winners[0] == 1 else 0) if len(winners) == 1 else 0
            train_stats["ties"] += (1 if len(winners) > 1 else 0)
            
            # Calculate win rate for tracking
            total_games = train_stats["player0_wins"] + train_stats["player1_wins"] + train_stats["ties"]
            if total_games > 0:
                train_stats["win_rates"].append(train_stats["player0_wins"] / total_games)
            
            # Calculate moving average of rewards
            window_size = min(100, len(train_stats["rewards"]))
            train_stats["avg_rewards"].append(
                sum(train_stats["rewards"][-window_size:]) / window_size
            )
            
            global_episode = train_stats["episodes"]
            
            # Save model at specified intervals
            if save_interval and model_path and global_episode % save_interval == 0:
                # Only one worker should save
                if worker_id == 0:
                    agent.save_model(f"{model_path}_ep{global_episode}.pt")
                    
                    # Generate learning curve plot
                    plt.figure(figsize=(15, 10))
                    
                    # Plot rewards
                    plt.subplot(2, 1, 1)
                    plt.plot(train_stats["rewards"], alpha=0.3, label='Rewards')
                    plt.plot(train_stats["avg_rewards"], label='Avg Rewards (100 ep)')
                    plt.title(f'Parallel PPO Training Progress - Episode {global_episode}')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.legend()
                    
                    # Plot win rate
                    plt.subplot(2, 1, 2)
                    plt.plot(train_stats["win_rates"], label='Player 0 Win Rate', color='green')
                    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Policy (50%)')
                    plt.title('Win Rate Analysis')
                    plt.xlabel('Episode')
                    plt.ylabel('Win Rate (Player 0)')
                    plt.ylim(0, 1)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(f"models/parallel_training_curve_{global_episode}.png")
                    plt.close()
        
        # Update policy at regular intervals
        if (episode + 1) % update_interval == 0:
            # Wait for all workers to reach this point
            barrier.wait()
            
            # Only one worker should perform the update
            if worker_id == 0:
                agent.update()
            
            # Wait for update to complete before continuing
            barrier.wait()

def train_parallel(
    num_episodes=1000, 
    num_workers=4,
    update_interval=4,
    save_every=100, 
    model_path="models/parallel_splendor_agent.pt"
):
    """
    Train the PPO agent in parallel across multiple processes
    
    Args:
        num_episodes: Total number of episodes to train for
        num_workers: Number of parallel worker processes
        update_interval: How often to update the policy (in episodes per worker)
        save_every: How often to save the model (in total episodes)
        model_path: Path to save the model
    """
    # Use all available CPU cores if not specified
    if num_workers <= 0:
        num_workers = mp.cpu_count()
    
    print(f"Training with {num_workers} parallel workers")
    
    # Initialize shared agent
    agent = ParallelPPOAgent(num_workers=num_workers)
    
    # Create multiprocessing barrier for synchronization
    barrier = mp.Barrier(num_workers)
    
    # Calculate episodes per worker
    base_episodes = num_episodes // num_workers
    extra = num_episodes % num_workers
    episodes_per_worker = [base_episodes + (1 if i < extra else 0) for i in range(num_workers)]
    
    # Create manager for shared training statistics
    manager = mp.Manager()
    train_stats = manager.dict({
        "episodes": 0,
        "rewards": manager.list(),
        "avg_rewards": manager.list(),
        "player0_wins": 0,
        "player1_wins": 0, 
        "ties": 0,
        "win_rates": manager.list(),
        "lock": manager.Lock()
    })
    
    # Create and start worker processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, agent, episodes_per_worker[i], update_interval, barrier, 
                  train_stats, model_path, save_every)
        )
        processes.append(p)
        p.start()
    
    # Create progress bar for monitoring
    with tqdm(total=num_episodes) as pbar:
        previous_count = 0
        
        # Update progress bar until all episodes complete
        while any(p.is_alive() for p in processes):
            current_count = train_stats["episodes"]
            if current_count > previous_count:
                pbar.update(current_count - previous_count)
                previous_count = current_count
            time.sleep(0.1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Save final model
    agent.save_model(model_path)
    
    # Print final statistics
    total_games = train_stats["player0_wins"] + train_stats["player1_wins"] + train_stats["ties"]
    win_rate = train_stats["player0_wins"] / total_games if total_games > 0 else 0.5
    
    print(f"Training complete - {train_stats['episodes']} episodes")
    print(f"Final Avg Reward: {train_stats['avg_rewards'][-1]:.2f}")
    print(f"Win Rate: Player 0: {win_rate:.2f} ({train_stats['player0_wins']}/{total_games}), " +
          f"Player 1: {1-win_rate:.2f} ({train_stats['player1_wins']}/{total_games}), " +
          f"Ties: {train_stats['ties']}/{total_games}")
    
    return agent

def evaluate(model_path="models/parallel_splendor_agent.pt", num_episodes=100):
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
    agent = ParallelPPOAgent(num_workers=1)  # Only need 1 worker for evaluation
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate a parallel PPO agent for Splendor')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'both'],
                        help='Operation mode: train, evaluate, or both')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train or evaluate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for training')
    parser.add_argument('--update_interval', type=int, default=4,
                        help='Update policy every N episodes per worker')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save frequency during training (total episodes)')
    parser.add_argument('--model_path', type=str, default='models/parallel_splendor_agent.pt',
                        help='Path to save/load the model')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        train_parallel(
            num_episodes=args.episodes,
            num_workers=args.workers,
            update_interval=args.update_interval,
            save_every=args.save_every,
            model_path=args.model_path
        )
    
    if args.mode in ['evaluate', 'both']:
        evaluate(model_path=args.model_path, num_episodes=100) 