import os
import sys
import time
import random
import numpy as np
import torch
from collections import deque

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from lapidary.game import GameState
from ppo_agent import SplendorPPOAgent
from splendor_env import SplendorEnv

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    num_episodes=10000,
    num_steps_per_update=128,
    num_updates=4,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    hidden_dim=256,
    seed=42,
    log_interval=10,
    save_interval=100
):
    """
    Train a PPO agent using self-play
    
    Args:
        num_episodes: Number of training episodes
        num_steps_per_update: Number of steps to collect before an update
        num_updates: Number of optimization iterations per update
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_ratio: PPO clip parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy loss coefficient
        hidden_dim: Hidden layer dimension
        seed: Random seed
        log_interval: How often to log training stats
        save_interval: How often to save model
    """
    seed_everything(seed)
    
    # Create environment
    env = SplendorEnv(num_players=2)
    
    # Create agent
    agent = SplendorPPOAgent(
        input_dim=2300,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rate = deque(maxlen=100)
    
    # Action type tracking
    action_types = {
        'gems': 0,
        'buy_available': 0,
        'buy_reserved': 0,
        'reserve': 0,
        'pass': 0,
        'other': 0
    }
    
    total_steps = 0
    start_time = time.time()
    
    print("Starting training...")
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        obs, info = env.reset(seed=seed + episode)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Storage for trajectory data
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        action_masks = []
        
        # Episode action type tracking
        episode_action_types = {
            'gems': 0,
            'buy_available': 0,
            'buy_reserved': 0,
            'reserve': 0,
            'pass': 0,
            'other': 0
        }
        
        # Play one episode
        while not done:
            # Create a mask of valid moves
            valid_moves_mask = [False] * 100
            for i in range(min(len(env.valid_moves), 100)):
                valid_moves_mask[i] = True
            
            # Self-play: agent plays against itself
            action, log_prob, value = agent.select_action(obs, valid_moves_mask)
            
            # Store trajectory data
            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            action_masks.append(valid_moves_mask)
            
            # Track action type
            if action < len(env.valid_moves) and action in env.valid_moves_mapping:
                move = env.valid_moves_mapping[action]
                if isinstance(move, tuple) and len(move) > 0:
                    move_type = move[0]
                    
                    # Check for pass actions
                    if move_type == 'gems' and isinstance(move[1], (list, tuple)) and len(move[1]) > 0 and sum(move[1]) == 0:
                        move_type = 'pass'
                    
                    if move_type in action_types:
                        action_types[move_type] += 1
                        episode_action_types[move_type] += 1
                    else:
                        action_types['other'] += 1
                        episode_action_types['other'] += 1
                else:
                    action_types['other'] += 1
                    episode_action_types['other'] += 1
            else:
                action_types['other'] += 1
                episode_action_types['other'] += 1
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            # Update
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            done = terminated or truncated
            
            # Update policy if we've collected enough steps or if episode is done
            if len(states) >= num_steps_per_update or done:
                # If the episode is not done, bootstrap value
                if not done:
                    with torch.no_grad():
                        # Convert to tensor
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                        
                        # Create valid moves mask
                        valid_moves_mask_tensor = torch.zeros(1, 100, dtype=torch.bool).to(agent.device)
                        for i in range(min(len(env.valid_moves), 100)):
                            valid_moves_mask_tensor[0, i] = True
                            
                        # Get value of last state using forward method
                        _, next_value = agent.network(state_tensor)
                        next_value = next_value.item()
                else:
                    next_value = 0
                
                # Compute returns and advantages
                returns, advantages = agent.compute_gae(
                    rewards, values, dones, next_value
                )
                
                # Prepare rollout data
                rollout_data = {
                    'states': states,
                    'actions': actions,
                    'log_probs': log_probs,
                    'returns': returns,
                    'advantages': advantages,
                    'action_masks': action_masks
                }
                
                # Update policy multiple times
                for _ in range(num_updates):
                    loss_info = agent.update(rollout_data)
                
                # Clear trajectory storage
                states = []
                actions = []
                log_probs = []
                values = []
                rewards = []
                dones = []
                action_masks = []
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine if agent won
        if terminated:
            scores = info['scores']
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
            
            # In case of a tie, prioritize by fewer cards
            if len(winners) > 1:
                min_cards = min(len(env.game_state.players[i].cards_played) for i in winners)
                winners = [i for i in winners if len(env.game_state.players[i].cards_played) == min_cards]
            
            # Track if player 0 won (in self-play, tracking either player is fine)
            # This should approach 50% in a balanced self-play scenario
            win = 0 in winners
            win_rate.append(float(win))
        
        # Log progress
        if episode % log_interval == 0:
            mean_reward = np.mean(episode_rewards[-log_interval:])
            mean_length = np.mean(episode_lengths[-log_interval:])
            current_win_rate = np.mean(win_rate) if win_rate else 0
            elapsed_time = time.time() - start_time
            
            # Calculate action type percentages
            total_actions = sum(action_types.values())
            if total_actions > 0:
                action_percentages = {k: (v / total_actions) * 100 for k, v in action_types.items()}
            else:
                action_percentages = {k: 0 for k in action_types}
            
            print(f"Episode {episode}/{num_episodes} | " +
                  f"Avg Reward: {mean_reward:.2f} | " +
                  f"Avg Length: {mean_length:.2f} | " +
                  f"Win Rate: {current_win_rate:.2f} | " +
                  f"Elapsed: {elapsed_time:.2f}s")
            
            print(f"Action Types: Gems: {action_percentages['gems']:.1f}%, " +
                  f"Buy: {action_percentages['buy_available']:.1f}%, " +
                  f"Buy Reserved: {action_percentages['buy_reserved']:.1f}%, " +
                  f"Reserve: {action_percentages['reserve']:.1f}%, " +
                  f"Pass: {action_percentages['pass']:.1f}%, " +
                  f"Other: {action_percentages['other']:.1f}%")
            
        # Save model
        if episode % save_interval == 0:
            agent.save(episode)
            print(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save("final")
    print("Training complete!")
    
    return agent

if __name__ == "__main__":
    # Train the agent
    agent = train(
        num_episodes=200,
        num_steps_per_update=128,
        num_updates=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.05,
        hidden_dim=256,
        seed=42,
        log_interval=10,
        save_interval=100
    ) 