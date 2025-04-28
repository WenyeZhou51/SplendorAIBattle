import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from parallel_rl_agent import ParallelPPOAgent
from enhanced_splendor_env import EnhancedSplendorEnv
from splendor_env import SplendorEnv

def evaluate_agent(
    model_path,
    num_episodes=100,
    opponent="random",
    enhanced_rewards=True,
    verbose=True
):
    """
    Evaluate a trained Splendor agent against different opponents
    
    Args:
        model_path: Path to the agent model to evaluate
        num_episodes: Number of episodes to evaluate
        opponent: Type of opponent ('random', 'heuristic')
        enhanced_rewards: Whether to use the enhanced reward environment
        verbose: Whether to print detailed evaluation information
        
    Returns:
        win_rate: Percentage of games won by the agent
        avg_reward: Average reward per episode
        avg_length: Average game length in moves
    """
    # Initialize environment based on the enhanced_rewards flag
    if enhanced_rewards:
        env = EnhancedSplendorEnv(num_players=2)
    else:
        env = SplendorEnv(num_players=2)
    
    # Load agent
    agent = ParallelPPOAgent(num_workers=1)
    try:
        agent.load_model(model_path)
        if verbose:
            print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0, 0
    
    # Opponent policies
    def random_policy(state, valid_moves_mask):
        # Sample a random valid move
        valid_indices = [i for i, v in enumerate(valid_moves_mask) if v]
        if not valid_indices:
            raise ValueError("No valid moves available")
        action = np.random.choice(valid_indices)
        return action, 0.0  # Action and dummy log probability
    
    def heuristic_policy(state, valid_moves_mask):
        # A simple heuristic policy that prioritizes:
        # 1. Buy cards if possible
        # 2. Get gems efficiently
        # 3. Reserve cards with gold
        valid_indices = [i for i, v in enumerate(valid_moves_mask) if v]
        if not valid_indices:
            raise ValueError("No valid moves available")
            
        # For the demo version, just randomly select with weights
        # In a real implementation, this would analyze the actual moves
        weights = np.ones(len(valid_indices))
        action_idx = np.random.choice(range(len(valid_indices)), p=weights/weights.sum())
        action = valid_indices[action_idx]
        return action, 0.0
    
    # Select opponent policy
    if opponent == "random":
        opponent_policy = random_policy
    elif opponent == "heuristic":
        opponent_policy = heuristic_policy
    else:
        print(f"Unknown opponent '{opponent}', using random")
        opponent_policy = random_policy
    
    # Evaluation metrics
    wins = 0
    ties = 0
    losses = 0
    total_rewards = []
    game_lengths = []
    points_scored = []
    
    # Run evaluation episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get valid moves
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            # Determine which policy to use based on current player
            if env.game_state.current_player_index == 0:
                # Agent's turn (player 0)
                action, _ = agent.get_action(state, valid_moves_mask)
            else:
                # Opponent's turn (player 1)
                action, _ = opponent_policy(state, valid_moves_mask)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Only count rewards for the agent (player 0)
            if env.game_state.current_player_index == 0:
                total_reward += reward
            
            # Update state and counters
            state = next_state
            steps += 1
        
        # Determine outcome
        if terminated:
            scores = info['scores']
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
            
            if len(winners) > 1:
                # Tie
                ties += 1
            elif winners[0] == 0:
                # Agent won
                wins += 1
            else:
                # Opponent won
                losses += 1
            
            # Record points scored by agent
            points_scored.append(scores[0])
        
        # Record episode stats
        total_rewards.append(total_reward)
        game_lengths.append(steps)
    
    # Calculate summary statistics
    win_rate = wins / num_episodes * 100
    tie_rate = ties / num_episodes * 100
    loss_rate = losses / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(game_lengths)
    avg_points = np.mean(points_scored) if points_scored else 0
    
    # Print results
    if verbose:
        print(f"\nEvaluation Results over {num_episodes} games against {opponent} opponent:")
        print(f"Win Rate: {win_rate:.1f}% ({wins}/{num_episodes})")
        print(f"Tie Rate: {tie_rate:.1f}% ({ties}/{num_episodes})")
        print(f"Loss Rate: {loss_rate:.1f}% ({losses}/{num_episodes})")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Game Length: {avg_length:.1f} moves")
        print(f"Average Points: {avg_points:.1f}")
    
    return win_rate, avg_reward, avg_length

def compare_agents(
    baseline_model_path,
    enhanced_model_path,
    opponents=["random", "heuristic"],
    num_episodes=50,
    save_chart=True,
    chart_path="agent_comparison.png"
):
    """
    Compare performance of baseline and enhanced agents against different opponents
    
    Args:
        baseline_model_path: Path to baseline agent model
        enhanced_model_path: Path to enhanced agent model
        opponents: List of opponents to test against
        num_episodes: Episodes to run per evaluation
        save_chart: Whether to save comparison chart
        chart_path: Path to save comparison chart
    """
    results = {
        "baseline": {},
        "enhanced": {}
    }
    
    # Evaluate baseline agent
    print("\nEvaluating baseline agent...")
    for opponent in opponents:
        print(f"\nTesting against {opponent} opponent:")
        win_rate, avg_reward, avg_length = evaluate_agent(
            baseline_model_path, 
            num_episodes=num_episodes,
            opponent=opponent,
            enhanced_rewards=False,
            verbose=True
        )
        results["baseline"][opponent] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length
        }
    
    # Evaluate enhanced agent
    print("\nEvaluating enhanced agent...")
    for opponent in opponents:
        print(f"\nTesting against {opponent} opponent:")
        win_rate, avg_reward, avg_length = evaluate_agent(
            enhanced_model_path, 
            num_episodes=num_episodes,
            opponent=opponent,
            enhanced_rewards=True,
            verbose=True
        )
        results["enhanced"][opponent] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length
        }
    
    # Generate comparison chart
    if save_chart:
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        bar_width = 0.35
        index = np.arange(len(opponents))
        
        # Win rate comparison
        plt.subplot(2, 1, 1)
        plt.bar(index, [results["baseline"][opp]["win_rate"] for opp in opponents], 
                bar_width, label='Baseline Agent')
        plt.bar(index + bar_width, [results["enhanced"][opp]["win_rate"] for opp in opponents],
                bar_width, label='Enhanced Agent')
        
        plt.ylabel('Win Rate (%)')
        plt.title('Agent Performance Comparison')
        plt.xticks(index + bar_width/2, opponents)
        plt.legend()
        
        # Reward comparison
        plt.subplot(2, 1, 2)
        plt.bar(index, [results["baseline"][opp]["avg_reward"] for opp in opponents], 
                bar_width, label='Baseline Agent')
        plt.bar(index + bar_width, [results["enhanced"][opp]["avg_reward"] for opp in opponents],
                bar_width, label='Enhanced Agent')
        
        plt.ylabel('Average Reward')
        plt.xlabel('Opponent Type')
        plt.xticks(index + bar_width/2, opponents)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"Comparison chart saved to {chart_path}")
    
    # Print summary
    print("\nPerformance Summary:")
    print("=" * 60)
    print(f"{'Agent':<15} {'Opponent':<15} {'Win Rate':<10} {'Avg Reward':<15} {'Avg Length':<10}")
    print("-" * 60)
    
    for agent_type in ["baseline", "enhanced"]:
        for opponent in opponents:
            result = results[agent_type][opponent]
            print(f"{agent_type:<15} {opponent:<15} {result['win_rate']:<10.1f}% {result['avg_reward']:<15.2f} {result['avg_length']:<10.1f}")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Splendor agents')
    parser.add_argument('--baseline', type=str, default='models/parallel_splendor_agent.pt',
                        help='Path to baseline agent model')
    parser.add_argument('--enhanced', type=str, default='models/splendor_league_agent.pt',
                        help='Path to enhanced agent model')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to evaluate per opponent')
    parser.add_argument('--mode', type=str, choices=['single', 'compare'], default='compare',
                        help='Evaluation mode - single agent or comparison')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for single agent evaluation')
    parser.add_argument('--opponent', type=str, default='random',
                        help='Opponent type for single evaluation')
    parser.add_argument('--chart', type=str, default='agent_comparison.png',
                        help='Path to save comparison chart')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        model_path = args.model if args.model else args.enhanced
        print(f"Evaluating single agent: {model_path}")
        evaluate_agent(
            model_path,
            num_episodes=args.episodes,
            opponent=args.opponent,
            enhanced_rewards=True,
            verbose=True
        )
    else:
        print(f"Comparing agents:\n  Baseline: {args.baseline}\n  Enhanced: {args.enhanced}")
        compare_agents(
            args.baseline,
            args.enhanced,
            opponents=["random", "heuristic"],
            num_episodes=args.episodes,
            save_chart=True,
            chart_path=args.chart
        ) 