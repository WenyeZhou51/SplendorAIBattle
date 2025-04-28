import os
import sys
import random
import numpy as np
import torch
import argparse
from collections import defaultdict, Counter
import copy

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from splendor_env import SplendorEnv
from parallel_rl_agent import ParallelPPOAgent
from lapidary.game import GameState, colours

class GreedyBot:
    """
    Implementation of a greedy bot for Splendor, similar to the one in the WebGUI.
    The strategy prioritizes:
    1. Win immediately if possible
    2. Buy cards with highest points/cost ratio
    3. Take gems that help obtain the cheapest available card
    """
    def __init__(self, name="GreedyBot"):
        self.name = name
    
    def get_action(self, state, valid_moves_mapping, game_state=None):
        """
        Choose the best action according to greedy heuristics
        
        Args:
            state: State vector of the Splendor environment (numpy array)
            valid_moves_mapping: Dictionary mapping action indices to game moves
            game_state: GameState object, passed separately
            
        Returns:
            action: The selected action index
        """
        # Need the game_state for proper evaluation
        if game_state is None:
            # If game_state not provided, just pick the first valid move as fallback
            valid_moves = list(valid_moves_mapping.keys())
            return valid_moves[0]
        
        valid_moves = list(valid_moves_mapping.keys())
        current_player = game_state.current_player_index
        
        # Group moves by type for easier processing
        buy_moves = []
        reserve_moves = []
        take_gems_moves = []
        
        for move_idx in valid_moves:
            move_type = valid_moves_mapping[move_idx][0]
            if move_type == "buy":
                buy_moves.append(move_idx)
            elif move_type == "reserve":
                reserve_moves.append(move_idx)
            elif move_type == "take_gems":
                take_gems_moves.append(move_idx)
                
        # STRATEGY 1: Check if any move leads to immediate win (15+ points)
        player_score = game_state.players[current_player].score
        for move_idx in buy_moves:
            move_details = valid_moves_mapping[move_idx]
            # Extract card points from the move details
            # Move format is typically ('buy', card_details)
            card_points = 0
            if len(move_details) > 1 and isinstance(move_details[1], dict) and 'points' in move_details[1]:
                card_points = move_details[1]['points']
            elif len(move_details) > 1 and hasattr(move_details[1], 'points'):
                card_points = move_details[1].points
                
            # If this card gives enough points to win, buy it immediately
            if player_score + card_points >= 15:
                return move_idx
        
        # STRATEGY 2: Buy cards with highest points/cost ratio
        best_card_move = None
        best_value_ratio = -1
        
        for move_idx in buy_moves:
            move_details = valid_moves_mapping[move_idx]
            card = None
            
            # Extract card details from move
            if len(move_details) > 1:
                if isinstance(move_details[1], dict):
                    card = move_details[1]
                else:
                    card = move_details[1]
            
            if card:
                points = 0
                total_cost = 0
                
                # Extract points
                if hasattr(card, 'points'):
                    points = card.points
                elif isinstance(card, dict) and 'points' in card:
                    points = card['points']
                
                # Extract costs and calculate total
                costs = {}
                if hasattr(card, 'costs'):
                    costs = card.costs
                elif isinstance(card, dict) and 'costs' in card:
                    costs = card['costs']
                
                # Sum up all costs
                for color, cost in costs.items():
                    total_cost += cost
                
                # Calculate value ratio (adding 1 to points to value cards with prestige)
                # Avoid division by zero
                if total_cost > 0:
                    value_ratio = (points + 1) / total_cost
                    if value_ratio > best_value_ratio:
                        best_value_ratio = value_ratio
                        best_card_move = move_idx
                elif points > 0:  # Free card with points
                    return move_idx  # Get it immediately
        
        # If we found a good card to buy, do it
        if best_card_move is not None:
            return best_card_move
        
        # STRATEGY 3: Take gems that help obtain the cheapest available card
        # First, identify the cheapest card we might want
        cheapest_card = None
        cheapest_cost = float('inf')
        
        # Check cards in the tableau and potentially noble cards
        available_cards = []
        
        # Extract cards from the game state - assuming the implementation details
        for tier in range(3):  # Usually 3 tiers in Splendor
            if hasattr(game_state, 'visible_cards') and tier < len(game_state.visible_cards):
                available_cards.extend(game_state.visible_cards[tier])
            elif hasattr(game_state, 'cards') and tier < len(game_state.cards):
                available_cards.extend(game_state.cards[tier])
        
        player = game_state.players[current_player]
        player_gems = {}
        
        # Get player's current gems
        if hasattr(player, 'gems'):
            player_gems = player.gems
        
        # Find the card that requires the fewest additional gems
        for card in available_cards:
            if card is None:
                continue
                
            # Skip if we don't have proper card format
            if not hasattr(card, 'costs') and (not isinstance(card, dict) or 'costs' not in card):
                continue
                
            # Extract card costs
            costs = card.costs if hasattr(card, 'costs') else card['costs']
            
            # Calculate additional gems needed
            additional_gems_needed = 0
            for color, cost in costs.items():
                player_has = player_gems.get(color, 0)
                if cost > player_has:
                    additional_gems_needed += (cost - player_has)
            
            if additional_gems_needed < cheapest_cost:
                cheapest_cost = additional_gems_needed
                cheapest_card = card
        
        # If we identified a desired card, take gems that help get it
        if cheapest_card is not None and take_gems_moves:
            # Extract the costs of the cheapest card
            costs = cheapest_card.costs if hasattr(cheapest_card, 'costs') else cheapest_card['costs']
            
            # Find gems we need
            gems_needed = {}
            for color, cost in costs.items():
                player_has = player_gems.get(color, 0)
                if cost > player_has:
                    gems_needed[color] = cost - player_has
            
            # Score each take_gems move based on how many needed gems it provides
            best_take_move = None
            best_take_score = -1
            
            for move_idx in take_gems_moves:
                move_details = valid_moves_mapping[move_idx]
                # Assuming take_gems move format: ('take_gems', {color: count, ...})
                gems_to_take = move_details[1] if len(move_details) > 1 else {}
                
                take_score = 0
                for color, count in gems_to_take.items():
                    if color in gems_needed:
                        take_score += min(count, gems_needed[color])
                
                if take_score > best_take_score:
                    best_take_score = take_score
                    best_take_move = move_idx
            
            if best_take_move is not None:
                return best_take_move
        
        # STRATEGY 4: If we can't buy a good card or take useful gems, reserve a high-point card
        if reserve_moves:
            best_reserve_move = None
            highest_points = -1
            
            for move_idx in reserve_moves:
                move_details = valid_moves_mapping[move_idx]
                # Reserve move format: ('reserve', card_details)
                card = move_details[1] if len(move_details) > 1 else None
                
                if card:
                    points = 0
                    if hasattr(card, 'points'):
                        points = card.points
                    elif isinstance(card, dict) and 'points' in card:
                        points = card['points']
                    
                    if points > highest_points:
                        highest_points = points
                        best_reserve_move = move_idx
            
            if best_reserve_move is not None:
                return best_reserve_move
        
        # If all else fails, just take any gems or random action
        if take_gems_moves:
            return take_gems_moves[0]
        elif buy_moves:
            return buy_moves[0]
        elif reserve_moves:
            return reserve_moves[0]
        
        # Fallback to first valid move if no clear strategy
        return valid_moves[0]


def evaluate_vs_greedy(model_path, num_games=10, swap_positions=False, verbose=True):
    """
    Evaluate the trained agent against a greedy bot
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
        swap_positions: Whether to swap positions (agent as player 0 or 1)
        verbose: Whether to print detailed game information
    
    Returns:
        stats: Dictionary containing evaluation statistics
    """
    # Initialize environment
    env = SplendorEnv(num_players=2)
    
    # Load trained agent
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    
    # Initialize the greedy bot
    greedy_bot = GreedyBot(name="GreedyBot")
    
    # Statistics to track
    stats = {
        "agent_wins": 0,
        "greedy_wins": 0,
        "ties": 0,
        "invalid_actions_attempted": 0,
        "invalid_actions_detail": defaultdict(int),
        "agent_action_pattern": defaultdict(int),
        "agent_scores": [],
        "greedy_scores": [],
        "game_lengths": []
    }
    
    for game_num in range(num_games):
        if verbose:
            print(f"\nGame {game_num + 1}/{num_games}")
        
        # Reset the environment
        state, info = env.reset()
        
        # Determine player assignments
        if swap_positions and game_num % 2 == 1:
            # Odd games: Greedy is player 0, Agent is player 1
            player_names = ["GreedyBot", "Agent"]
            agent_player_idx = 1
        else:
            # Even games: Agent is player 0, Greedy is player 1
            player_names = ["Agent", "GreedyBot"]
            agent_player_idx = 0
        
        done = False
        turn_count = 0
        invalid_attempts = 0
        action_pattern = defaultdict(int)
        
        while not done:
            # Get valid moves
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            # Determine whose turn it is
            current_player_idx = env.game_state.current_player_index
            
            # Select action based on player
            try:
                if current_player_idx == agent_player_idx:
                    # Agent's turn
                    action, _ = agent.get_action(state, valid_moves_mask)
                    
                    # Check the type of move for pattern analysis
                    move_type = env.valid_moves_mapping[action][0]
                    action_pattern[move_type] += 1
                    
                    if verbose:
                        print(f"Turn {turn_count}: Agent selects action: {env.valid_moves_mapping[action]}")
                else:
                    # Greedy bot's turn
                    action = greedy_bot.get_action(state, env.valid_moves_mapping, game_state=env.game_state)
                    
                    if verbose:
                        print(f"Turn {turn_count}: GreedyBot selects action: {env.valid_moves_mapping[action]}")
            
            except Exception as e:
                # Log invalid action attempt
                invalid_attempts += 1
                if isinstance(e, ValueError) and "Invalid action" in str(e):
                    error_detail = str(e).split('. ')[0]  # Extract first part of error message
                    stats["invalid_actions_detail"][error_detail] += 1
                
                # Break the game if there's an error
                print(f"Error during action selection: {e}")
                break
            
            # Take action in environment
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                done = terminated or truncated
                turn_count += 1
                
                if verbose:
                    scores = info['scores']
                    print(f"  Scores: {player_names[0]}: {scores[0]}, {player_names[1]}: {scores[1]}")
            
            except Exception as e:
                # Log environment execution error
                print(f"Error during environment step: {e}")
                break
        
        # Record game statistics
        stats["invalid_actions_attempted"] += invalid_attempts
        stats["game_lengths"].append(turn_count)
        
        # Add action pattern to overall stats
        for move_type, count in action_pattern.items():
            stats["agent_action_pattern"][move_type] += count
        
        # Determine winner
        if done:
            scores = info['scores']
            stats["agent_scores"].append(scores[agent_player_idx])
            stats["greedy_scores"].append(scores[1 - agent_player_idx])
            
            if scores[0] > scores[1]:
                winner = player_names[0]
                if agent_player_idx == 0:
                    stats["agent_wins"] += 1
                else:
                    stats["greedy_wins"] += 1
            elif scores[1] > scores[0]:
                winner = player_names[1]
                if agent_player_idx == 1:
                    stats["agent_wins"] += 1
                else:
                    stats["greedy_wins"] += 1
            else:
                winner = "Tie"
                stats["ties"] += 1
            
            if verbose:
                print(f"Game {game_num + 1} ended. Winner: {winner}")
                print(f"Final scores: {player_names[0]}: {scores[0]}, {player_names[1]}: {scores[1]}")
        else:
            print(f"Game {game_num + 1} terminated early due to errors")
    
    # Calculate win rates
    num_completed_games = stats["agent_wins"] + stats["greedy_wins"] + stats["ties"]
    if num_completed_games > 0:
        stats["agent_win_rate"] = stats["agent_wins"] / num_completed_games
        stats["greedy_win_rate"] = stats["greedy_wins"] / num_completed_games
        stats["tie_rate"] = stats["ties"] / num_completed_games
    
    # Calculate average scores
    if stats["agent_scores"]:
        stats["avg_agent_score"] = sum(stats["agent_scores"]) / len(stats["agent_scores"])
        stats["avg_greedy_score"] = sum(stats["greedy_scores"]) / len(stats["greedy_scores"])
    
    return stats


def print_detailed_analysis(stats):
    """Print a detailed analysis of the evaluation results"""
    print("\n======= EVALUATION RESULTS =======")
    print(f"Games played: {sum([stats['agent_wins'], stats['greedy_wins'], stats['ties']])}")
    print(f"Agent wins: {stats['agent_wins']} ({stats.get('agent_win_rate', 0)*100:.1f}%)")
    print(f"Greedy bot wins: {stats['greedy_wins']} ({stats.get('greedy_win_rate', 0)*100:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats.get('tie_rate', 0)*100:.1f}%)")
    
    if 'avg_agent_score' in stats:
        print(f"\nAverage scores:")
        print(f"Agent: {stats['avg_agent_score']:.2f}")
        print(f"Greedy bot: {stats['avg_greedy_score']:.2f}")
    
    print(f"\nAverage game length: {sum(stats['game_lengths'])/len(stats['game_lengths']):.1f} turns")
    
    print("\nAgent action patterns:")
    total_actions = sum(stats["agent_action_pattern"].values())
    sorted_patterns = sorted(stats["agent_action_pattern"].items(), key=lambda x: x[1], reverse=True)
    for move_type, count in sorted_patterns:
        print(f"  {move_type}: {count} ({count/total_actions*100:.1f}%)")
    
    if stats["invalid_actions_attempted"] > 0:
        print("\nInvalid actions attempted by agent:")
        print(f"  Total: {stats['invalid_actions_attempted']}")
        
        print("\nDetails of invalid actions:")
        sorted_errors = sorted(stats["invalid_actions_detail"].items(), key=lambda x: x[1], reverse=True)
        for error, count in sorted_errors:
            print(f"  {error}: {count}")
    else:
        print("\nNo invalid actions attempted by agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained Splendor agent against a greedy bot')
    parser.add_argument('--model', type=str, default='models/parallel_splendor_agent.pt',
                        help='Path to the trained model')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play')
    parser.add_argument('--swap', action='store_true',
                        help='Swap positions between games')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game information')
    
    args = parser.parse_args()
    
    print(f"Evaluating model {args.model} against greedy bot for {args.games} games...")
    stats = evaluate_vs_greedy(
        model_path=args.model,
        num_games=args.games,
        swap_positions=args.swap,
        verbose=args.verbose
    )
    
    print_detailed_analysis(stats) 