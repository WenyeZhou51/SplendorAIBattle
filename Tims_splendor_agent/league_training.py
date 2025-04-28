import os
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import random
import pickle
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from parallel_rl_agent import ParallelPPOAgent
from enhanced_splendor_env import EnhancedSplendorEnv

class SplendorLeague:
    """
    A league of Splendor agent policies for training with varied opponents.
    Implements a simplified ELO rating system and opponent selection strategies.
    """
    def __init__(self, save_dir="models/league"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # League members (agent_id -> model_path)
        self.members = {}
        
        # ELO ratings (agent_id -> rating)
        self.ratings = {}
        
        # Match history (list of tuples: (agent1_id, agent2_id, winner_id, timestamp))
        self.match_history = []
        
        # Default rating for new members
        self.default_rating = 1200
        
        # ELO K factor (determines how much ratings change after each match)
        self.k_factor = 32
        
        # Load league history if it exists
        self.league_info_path = os.path.join(save_dir, "league_info.pkl")
        self._load_league_info()
    
    def _load_league_info(self):
        """Load league information from disk if available"""
        if os.path.exists(self.league_info_path):
            try:
                with open(self.league_info_path, 'rb') as f:
                    league_info = pickle.load(f)
                    self.members = league_info.get('members', {})
                    self.ratings = league_info.get('ratings', {})
                    self.match_history = league_info.get('match_history', [])
                    
                    # Verify model paths exist
                    valid_members = {}
                    for agent_id, model_path in self.members.items():
                        # If it's a special path like fixed: or training:, keep it
                        if model_path.startswith("fixed:") or model_path.startswith("training:"):
                            valid_members[agent_id] = model_path
                        # Otherwise check if file exists
                        elif os.path.exists(model_path):
                            valid_members[agent_id] = model_path
                        else:
                            print(f"Removing league member {agent_id}: model path {model_path} not found")
                    
                    # Update members dictionary
                    self.members = valid_members
                    
                    # Remove ratings for deleted members
                    valid_ratings = {}
                    for agent_id, rating in self.ratings.items():
                        if agent_id in valid_members:
                            valid_ratings[agent_id] = rating
                    self.ratings = valid_ratings
                    
                    # Filter match history to include only valid members
                    valid_history = []
                    for match in self.match_history:
                        agent1_id, agent2_id = match[0], match[1]
                        if agent1_id in valid_members and agent2_id in valid_members:
                            valid_history.append(match)
                    self.match_history = valid_history
                    
                    print(f"Loaded league with {len(self.members)} members")
            except Exception as e:
                print(f"Error loading league info: {e}")
                self._create_default_league()
        else:
            self._create_default_league()
    
    def _create_default_league(self):
        """Create a default league with rule-based 'fixed' opponents"""
        self.members = {}
        self.ratings = {}
        self.match_history = []
        
        # Add fixed opponents (rule-based)
        fixed_opponents = {
            "random_agent": "fixed:random",
            "heuristic_agent": "fixed:heuristic",
        }
        
        for agent_id, model_path in fixed_opponents.items():
            self.add_member(agent_id, model_path)
    
    def save_league_info(self):
        """Save league information to disk"""
        league_info = {
            'members': self.members,
            'ratings': self.ratings,
            'match_history': self.match_history
        }
        
        with open(self.league_info_path, 'wb') as f:
            pickle.dump(league_info, f)
    
    def add_member(self, agent_id, model_path):
        """Add a new member to the league"""
        self.members[agent_id] = model_path
        self.ratings[agent_id] = self.default_rating
        print(f"Added {agent_id} to league with rating {self.default_rating}")
        self.save_league_info()
    
    def update_rating(self, agent1_id, agent2_id, winner_id):
        """Update ELO ratings based on match result"""
        if agent1_id not in self.ratings or agent2_id not in self.ratings:
            print(f"Error: Both agents must be in the league to update ratings")
            return
        
        r1 = self.ratings[agent1_id]
        r2 = self.ratings[agent2_id]
        
        # Expected scores (probability of winning)
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        s1 = 1 if winner_id == agent1_id else 0 if winner_id == agent2_id else 0.5
        s2 = 1 if winner_id == agent2_id else 0 if winner_id == agent1_id else 0.5
        
        # Update ratings
        self.ratings[agent1_id] = r1 + self.k_factor * (s1 - e1)
        self.ratings[agent2_id] = r2 + self.k_factor * (s2 - e2)
        
        # Record match in history
        self.match_history.append((agent1_id, agent2_id, winner_id, time.time()))
        self.save_league_info()
    
    def select_opponent(self, current_id, selection_strategy="mix"):
        """
        Select an opponent from the league based on specified strategy
        
        Strategies:
        - random: Select a random opponent
        - best: Select the highest-rated opponent
        - similar: Select an opponent with similar rating
        - mix: Mix of strategies (default)
        """
        if len(self.members) <= 1:
            print("No opponents available in the league")
            return None
        
        potential_opponents = [aid for aid in self.members.keys() if aid != current_id]
        
        if selection_strategy == "random" or not potential_opponents:
            return random.choice(potential_opponents)
        
        elif selection_strategy == "best":
            # Return the highest-rated opponent
            return max(potential_opponents, key=lambda x: self.ratings.get(x, 0))
        
        elif selection_strategy == "similar":
            # Return opponent with closest rating
            current_rating = self.ratings.get(current_id, self.default_rating)
            return min(potential_opponents, 
                       key=lambda x: abs(self.ratings.get(x, self.default_rating) - current_rating))
        
        elif selection_strategy == "mix":
            # 40% time random, 30% time best, 30% time similar
            strategy = np.random.choice(
                ["random", "best", "similar"], 
                p=[0.4, 0.3, 0.3]
            )
            return self.select_opponent(current_id, strategy)
        
        else:
            print(f"Unknown selection strategy: {selection_strategy}")
            return random.choice(potential_opponents)


class LeagueTrainer:
    """
    Trainer for Splendor agents using a league-based approach
    with varied opponents and ELO ratings.
    """
    def __init__(
        self,
        base_model_path="models/splendor_league_agent.pt",
        league_dir="models/league",
        save_frequency=200,
        num_workers=4,
        update_interval=4
    ):
        self.base_model_path = base_model_path
        self.save_frequency = save_frequency
        self.num_workers = num_workers
        self.update_interval = update_interval
        
        # Initialize the league
        self.league = SplendorLeague(league_dir)
        
        # Current agent ID format: agent_{timestamp}
        self.current_agent_id = f"agent_{int(time.time())}"
        
        # Initialize the agent
        self.agent = ParallelPPOAgent(num_workers=num_workers)
        
        # Make base_model_path absolute
        self.base_model_path = self._ensure_absolute_path(base_model_path)
        
        # Add current agent to the league to avoid rating errors
        # Use a special marker for the current training agent
        self.league.add_member(self.current_agent_id, "training:current")
        
        # Fixed opponents handlers
        self.fixed_opponents = {
            "random": self._random_policy,
            "heuristic": self._heuristic_policy
        }
    
    def _random_policy(self, state, valid_moves_mask):
        """A random policy that selects a random valid move"""
        valid_indices = [i for i, v in enumerate(valid_moves_mask) if v]
        action = random.choice(valid_indices)
        # Return action and a dummy log probability 
        return action, 0.0
    
    def _heuristic_policy(self, state, valid_moves_mask):
        """
        A simple heuristic policy for Splendor:
        1. Prefer claiming nobles if possible
        2. Prefer buying cards over reserving
        3. Prefer cards with points
        4. Prefer taking diverse gems
        """
        # This is a placeholder for a more sophisticated heuristic
        # In a full implementation, this would analyze the game state and make
        # strategic decisions based on predefined rules
        
        # For now, just a slightly better than random policy:
        # 70% random, 30% bias toward buying cards with points
        if random.random() < 0.3:
            return self._random_policy(state, valid_moves_mask)
        
        # Simplified heuristic: just pick the first valid move
        valid_indices = [i for i, v in enumerate(valid_moves_mask) if v]
        if valid_indices:
            action = valid_indices[0]
        else:
            action = 0  # Fallback
        
        return action, 0.0
    
    def _load_opponent_policy(self, opponent_id):
        """Load an opponent policy from the league"""
        model_path = self.league.members.get(opponent_id)
        
        # Fixed policies (not neural network-based)
        if model_path and model_path.startswith("fixed:"):
            fixed_policy_name = model_path.split(":", 1)[1]
            if fixed_policy_name in self.fixed_opponents:
                return self.fixed_opponents[fixed_policy_name]
            else:
                print(f"Unknown fixed policy: {fixed_policy_name}")
                return self._random_policy
        
        # Special training marker for the current training agent
        elif model_path and model_path.startswith("training:"):
            # Use the current agent's policy
            return self.agent.get_action
        
        # Neural network policies
        elif model_path and os.path.exists(model_path):
            try:
                opponent_agent = ParallelPPOAgent(num_workers=1)
                opponent_agent.load_model(model_path)
                return opponent_agent.get_action
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                return self._random_policy
        
        # Fallback to random policy
        else:
            print(f"Could not load policy for {opponent_id}, path '{model_path}' not found")
            return self._random_policy
    
    def _ensure_absolute_path(self, path):
        """Convert relative paths to absolute paths"""
        if not os.path.isabs(path):
            return os.path.abspath(path)
        return path
    
    def train(self, num_episodes=1000):
        """
        Train the agent using league-based opponent selection
        
        Args:
            num_episodes: Total number of episodes to train for
        """
        # Use all available CPU cores if workers is set to 0
        if self.num_workers <= 0:
            self.num_workers = mp.cpu_count()
            # Reinitialize the agent with the correct number of workers
            self.agent = ParallelPPOAgent(num_workers=self.num_workers)
            print(f"Using all available CPU cores: {self.num_workers}")
        
        print(f"Starting league-based training for {num_episodes} episodes")
        print(f"Current agent: {self.current_agent_id}")
        print(f"Using {self.num_workers} workers")
        
        # Create multiprocessing manager for shared stats
        manager = mp.Manager()
        train_stats = manager.dict({
            "episodes": 0,
            "rewards": manager.list(),
            "avg_rewards": manager.list(),
            "win_rates": manager.list(),
            "opponent_ids": manager.list(),
            "lock": manager.Lock()
        })
        
        # Create barrier for synchronization
        barrier = mp.Barrier(self.num_workers)
        
        # Calculate episodes per worker
        base_episodes = num_episodes // self.num_workers
        extra = num_episodes % self.num_workers
        episodes_per_worker = [base_episodes + (1 if i < extra else 0) for i in range(self.num_workers)]
        
        # Start worker processes
        processes = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(i, episodes_per_worker[i], barrier, train_stats)
            )
            processes.append(p)
            p.start()
        
        # Monitor progress
        with tqdm(total=num_episodes) as pbar:
            previous_count = 0
            
            while any(p.is_alive() for p in processes):
                current_count = train_stats["episodes"]
                if current_count > previous_count:
                    pbar.update(current_count - previous_count)
                    previous_count = current_count
                time.sleep(0.1)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Final save
        final_model_path = self.base_model_path
        self.agent.save_model(final_model_path)
        
        # Add final agent to league with updated ID
        final_agent_id = f"{self.current_agent_id}_final"
        
        # Verify the file exists before adding to league
        if os.path.exists(final_model_path):
            self.league.add_member(final_agent_id, final_model_path)
        else:
            print(f"Warning: Could not create final model file at {final_model_path}")
        
        # Generate and save training plots
        self._generate_training_plots(train_stats)
        
        print(f"Training complete - final model saved as {final_agent_id}")
        return self.agent
    
    def _worker_process(self, worker_id, num_episodes, barrier, train_stats):
        """Worker process for league-based training"""
        # Initialize environment
        env = EnhancedSplendorEnv(num_players=2)
        
        # Training loop
        for episode in range(num_episodes):
            # Every N episodes, save intermediate model and add to league
            global_episode = 0
            with train_stats["lock"]:
                global_episode = train_stats["episodes"]
            
            if worker_id == 0 and global_episode > 0 and global_episode % self.save_frequency == 0:
                # Save intermediate model
                intermediate_id = f"{self.current_agent_id}_{global_episode}"
                intermediate_path = self.base_model_path.replace(".pt", f"_{global_episode}.pt")
                
                # Ensure absolute path
                intermediate_path = self._ensure_absolute_path(intermediate_path)
                
                # Save the model
                self.agent.save_model(intermediate_path)
                
                # Verify the file was created before adding to league
                if os.path.exists(intermediate_path):
                    # Add to league
                    self.league.add_member(intermediate_id, intermediate_path)
                else:
                    print(f"Warning: Could not create model file at {intermediate_path}")
            
            # Reset environment
            state, info = env.reset()
            
            # Select opponent for this episode
            opponent_id = self.league.select_opponent(self.current_agent_id, "mix")
            opponent_policy = self._load_opponent_policy(opponent_id)
            
            # Track per-episode stats
            total_reward = 0
            steps = 0
            done = False
            current_player_idx = 0
            
            # Episode loop
            while not done:
                valid_moves = list(env.valid_moves_mapping.keys())
                valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
                
                # Determine which policy to use based on current player
                if env.game_state.current_player_index == 0:
                    # Current agent's turn
                    action, log_prob = self.agent.get_action(state, valid_moves_mask)
                    # Remember experience for learning
                    current_player_idx = 0
                else:
                    # Opponent's turn
                    action, _ = opponent_policy(state, valid_moves_mask)
                    # Don't need to remember opponent's experiences
                    log_prob = 0
                    current_player_idx = 1
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Only store experiences for the learning agent (player 0)
                if env.game_state.current_player_index == 0:
                    self.agent.remember(worker_id, state, action, reward, done, log_prob, valid_moves_mask)
                    total_reward += reward
                
                # Update state
                state = next_state
                steps += 1
            
            # Determine winner
            winner = None
            if terminated:
                scores = info['scores']
                max_score = max(scores)
                winners = [i for i, score in enumerate(scores) if score == max_score]
                if len(winners) == 1:  # Clear winner
                    winner = winners[0]
            
            # Update league ratings
            if winner is not None:
                winner_id = self.current_agent_id if winner == 0 else opponent_id
                try:
                    if opponent_id in self.league.members and self.current_agent_id in self.league.members:
                        self.league.update_rating(self.current_agent_id, opponent_id, winner_id)
                except Exception as e:
                    print(f"Error updating ratings: {e}")
            
            # Update global stats
            with train_stats["lock"]:
                train_stats["episodes"] += 1
                train_stats["rewards"].append(total_reward)
                train_stats["opponent_ids"].append(opponent_id)
                
                # Win tracking
                if winner == 0:
                    win = 1
                elif winner == 1:
                    win = 0
                else:
                    win = 0.5  # Tie
                    
                train_stats["win_rates"].append(win)
                
                # Calculate moving average of rewards
                window_size = min(100, len(train_stats["rewards"]))
                train_stats["avg_rewards"].append(
                    sum(train_stats["rewards"][-window_size:]) / window_size
                )
            
            # Update policy at regular intervals
            if (episode + 1) % self.update_interval == 0:
                # Wait for all workers to reach this point
                barrier.wait()
                
                # Only one worker should perform the update
                if worker_id == 0:
                    self.agent.update()
                
                # Wait for update to complete before continuing
                barrier.wait()
    
    def _generate_training_plots(self, train_stats):
        """Generate and save training plots"""
        plt.figure(figsize=(15, 15))
        
        # Plot rewards
        plt.subplot(3, 1, 1)
        plt.plot(train_stats["rewards"], alpha=0.3, label='Rewards')
        plt.plot(train_stats["avg_rewards"], label='Avg Rewards (100 ep)')
        plt.title('League Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # Plot win rates (moving average over 100 episodes)
        plt.subplot(3, 1, 2)
        win_rates = []
        window = 100
        for i in range(len(train_stats["win_rates"])):
            start_idx = max(0, i - window + 1)
            win_rates.append(sum(train_stats["win_rates"][start_idx:i+1]) / (i - start_idx + 1))
        
        plt.plot(win_rates, label='Win Rate (moving avg)', color='green')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Win Rate')
        plt.title('Win Rate Against League Opponents')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.legend()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(self.base_model_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"league_training_{self.current_agent_id}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Splendor agent using league-based approach')
    parser.add_argument('--episodes', type=int, default=5000,
                      help='Number of episodes to train for')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel workers')
    parser.add_argument('--save_frequency', type=int, default=500,
                      help='Frequency to save intermediate models and add to league')
    parser.add_argument('--update_interval', type=int, default=4,
                      help='Policy update frequency (in episodes per worker)')
    parser.add_argument('--model_path', type=str, default='models/splendor_league_agent.pt',
                      help='Path to save the model')
    parser.add_argument('--league_dir', type=str, default='models/league',
                      help='Directory for league model storage')
    
    args = parser.parse_args()
    
    # Create league trainer and start training
    trainer = LeagueTrainer(
        base_model_path=args.model_path,
        league_dir=args.league_dir,
        save_frequency=args.save_frequency,
        num_workers=args.workers,
        update_interval=args.update_interval
    )
    
    trainer.train(num_episodes=args.episodes) 