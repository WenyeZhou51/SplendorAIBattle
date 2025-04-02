import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from parallel_rl_agent import ParallelPPOAgent
from splendor_env import SplendorEnv

def analyze_model_strategy(model_path="models/card_specialized_splendor_agent.pt", 
                         output_dir="models/strategy_analysis", 
                         num_games=5):
    """
    Analyze the gameplay strategy of a trained Splendor model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and environment
    print(f"Loading model from {model_path}...")
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    env = SplendorEnv(num_players=2)
    
    # Statistics to track
    action_frequencies = defaultdict(int)
    gem_preferences = defaultdict(int)
    card_preferences = []  # Will store (tier, color, points, cost) tuples
    game_lengths = []
    points_per_game = []
    gems_collected = defaultdict(int)
    cards_purchased = defaultdict(int)
    nobles_acquired = 0
    
    # Action value estimates
    action_values = defaultdict(list)
    
    # Track actions over time (early, mid, late game)
    early_actions = defaultdict(int)  # First 33% of moves
    mid_actions = defaultdict(int)    # Middle 33% of moves
    late_actions = defaultdict(int)   # Last 33% of moves
    
    print(f"Analyzing {num_games} games...")
    for game_idx in range(num_games):
        state, info = env.reset()
        game_actions = []
        done = False
        step = 0
        total_points = 0
        
        print(f"Starting game {game_idx+1}...")
        
        while not done and step < 200:  # Prevent infinite loops
            # Get valid moves
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            # Get action probabilities and values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action_probs, value = agent.policy(state_tensor)
                
                # Apply mask for valid moves
                masked_probs = action_probs * torch.FloatTensor(valid_moves_mask)
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
            
            # Get top actions
            top_actions = torch.topk(masked_probs, min(5, len(valid_moves)))
            
            # Record action preferences and corresponding move details
            for i, idx in enumerate(top_actions.indices):
                action_idx = idx.item()
                prob = top_actions.values[i].item()
                
                if action_idx in env.valid_moves_mapping:
                    move = env.valid_moves_mapping[action_idx]
                    move_type = move[0]  # 'gems', 'card', 'reserve'
                    
                    # Track general action preference
                    action_frequencies[move_type] += prob
                    action_values[move_type].append(value.item())
                    
                    # Record the move
                    game_actions.append((step, move_type, prob, value.item()))
                    
                    # Track specific preferences
                    if move_type == 'gems' and len(move) > 1:
                        for color, count in move[1].items():
                            if count > 0:
                                gem_preferences[color] += prob * count
                    
                    elif move_type == 'card' and len(move) > 1:
                        card = move[1]
                        if hasattr(card, 'tier') and hasattr(card, 'colour') and hasattr(card, 'points'):
                            # Calculate total cost
                            total_cost = sum(card.cost.values()) if hasattr(card, 'cost') else 0
                            # Store card details with the selection probability
                            card_preferences.append((card.tier, card.colour, card.points, 
                                                  total_cost, prob))
            
            # Take actual action
            action, _ = agent.get_action(state, valid_moves_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Record actual move
            if action in env.valid_moves_mapping:
                actual_move = env.valid_moves_mapping[action]
                actual_move_type = actual_move[0]
                
                # Count gems collected
                if actual_move_type == 'gems' and len(actual_move) > 1:
                    for color, count in actual_move[1].items():
                        if count > 0:
                            gems_collected[color] += count
                
                # Count cards purchased
                if actual_move_type == 'card' and len(actual_move) > 1:
                    card = actual_move[1]
                    if hasattr(card, 'colour'):
                        cards_purchased[card.colour] += 1
            else:
                print(f"Warning: Model selected invalid action {action} not in valid_moves_mapping")
            
            # Store state transitions
            state = next_state
            step += 1
            done = terminated or truncated
            
            # Check if game ended due to victory
            if terminated and hasattr(env.game_state, 'players'):
                player_scores = [player.score for player in env.game_state.players]
                total_points = player_scores[0]  # Assuming we're tracking player 0
                
                # Count nobles
                if hasattr(env.game_state.players[0], 'nobles'):
                    nobles_acquired += len(env.game_state.players[0].nobles)
        
        game_lengths.append(step)
        points_per_game.append(total_points)
        
        # Categorize actions into early, mid, late game
        if game_actions:
            total_steps = len(game_actions)
            early_end = total_steps // 3
            mid_end = 2 * total_steps // 3
            
            for i, (step, move_type, prob, _) in enumerate(game_actions):
                if i < early_end:
                    early_actions[move_type] += prob
                elif i < mid_end:
                    mid_actions[move_type] += prob
                else:
                    late_actions[move_type] += prob
        
        print(f"Game {game_idx+1} completed in {step} steps with {total_points} points")
    
    print("\nGenerating analysis visualizations...")
    
    # 1. Overall action preferences
    plt.figure(figsize=(10, 6))
    total = sum(action_frequencies.values())
    labels = list(action_frequencies.keys())
    values = [freq/total*100 for freq in action_frequencies.values()]
    
    colors = {'gems': 'gold', 'card': 'green', 'reserve': 'blue', 'noble': 'purple'}
    bar_colors = [colors.get(label, 'gray') for label in labels]
    
    plt.bar(labels, values, color=bar_colors)
    plt.title('Action Type Preferences')
    plt.xlabel('Action Type')
    plt.ylabel('Preference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_preferences.png'))
    plt.close()
    
    # 2. Gem color preferences
    plt.figure(figsize=(10, 6))
    total_gems = sum(gem_preferences.values())
    if total_gems > 0:
        labels = list(gem_preferences.keys())
        values = [count/total_gems*100 for count in gem_preferences.values()]
        
        # Use actual gem colors
        gem_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 
                    'black': 'black', 'white': 'lightgray', 'gold': 'gold'}
        colors = [gem_colors.get(label, 'gray') for label in labels]
        
        plt.bar(labels, values, color=colors)
        plt.title('Gem Color Preferences')
        plt.xlabel('Gem Color')
        plt.ylabel('Preference (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gem_preferences.png'))
    plt.close()
    
    # 3. Card tier and point preferences
    if card_preferences:
        # Group by tier
        tiers = defaultdict(list)
        for tier, color, points, cost, prob in card_preferences:
            tiers[tier].append((points, cost, prob))
            
        # Plot points distribution by tier
        plt.figure(figsize=(12, 6))
        for i, tier in enumerate(sorted(tiers.keys())):
            tier_data = tiers[tier]
            points = [p for p, _, _ in tier_data]
            weights = [prob for _, _, prob in tier_data]
            
            plt.subplot(1, len(tiers), i+1)
            if weights and sum(weights) > 0:
                # Normalize weights
                weights = [w/sum(weights) for w in weights]
                plt.hist(points, bins=range(6), weights=weights, alpha=0.7)
                plt.title(f'Tier {tier} Points')
                plt.xlabel('Points')
                plt.ylabel('Preference')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'card_point_preferences.png'))
        plt.close()
        
        # Plot cost efficiency (points/cost) by tier
        plt.figure(figsize=(12, 6))
        for i, tier in enumerate(sorted(tiers.keys())):
            tier_data = tiers[tier]
            # Calculate points per cost (avoid division by zero)
            efficiency = [p/max(c, 1) for p, c, _ in tier_data]
            weights = [prob for _, _, prob in tier_data]
            
            plt.subplot(1, len(tiers), i+1)
            if weights and sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
                plt.hist(efficiency, bins=5, weights=weights, alpha=0.7)
                plt.title(f'Tier {tier} Efficiency')
                plt.xlabel('Points/Cost')
                plt.ylabel('Preference')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'card_efficiency.png'))
        plt.close()
    
    # 4. Action evolution through game (early, mid, late)
    plt.figure(figsize=(12, 6))
    action_types = sorted(set(list(early_actions.keys()) + list(mid_actions.keys()) + list(late_actions.keys())))
    
    # Normalize each phase
    early_total = sum(early_actions.values())
    mid_total = sum(mid_actions.values())
    late_total = sum(late_actions.values())
    
    early_values = [early_actions[action]/early_total*100 if early_total > 0 else 0 for action in action_types]
    mid_values = [mid_actions[action]/mid_total*100 if mid_total > 0 else 0 for action in action_types]
    late_values = [late_actions[action]/late_total*100 if late_total > 0 else 0 for action in action_types]
    
    x = np.arange(len(action_types))
    width = 0.25
    
    plt.bar(x - width, early_values, width, label='Early Game', color='lightblue')
    plt.bar(x, mid_values, width, label='Mid Game', color='royalblue')
    plt.bar(x + width, late_values, width, label='Late Game', color='darkblue')
    
    plt.title('Action Evolution Throughout Game')
    plt.xlabel('Action Type')
    plt.ylabel('Preference (%)')
    plt.xticks(x, action_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_evolution.png'))
    plt.close()
    
    # 5. Game statistics
    plt.figure(figsize=(12, 10))
    
    # Game lengths
    plt.subplot(2, 2, 1)
    plt.hist(game_lengths, bins=5, color='blue', alpha=0.7)
    plt.title('Game Length Distribution')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    
    # Points per game
    plt.subplot(2, 2, 2)
    plt.hist(points_per_game, bins=5, color='green', alpha=0.7)
    plt.title('Points Distribution')
    plt.xlabel('Final Points')
    plt.ylabel('Frequency')
    
    # Gems collected
    plt.subplot(2, 2, 3)
    if gems_collected:
        labels = list(gems_collected.keys())
        values = list(gems_collected.values())
        gem_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 
                     'black': 'black', 'white': 'lightgray', 'gold': 'gold'}
        colors = [gem_colors.get(label, 'gray') for label in labels]
        plt.bar(labels, values, color=colors)
        plt.title('Gems Collected')
        plt.xlabel('Gem Color')
        plt.ylabel('Count')
    
    # Cards purchased
    plt.subplot(2, 2, 4)
    if cards_purchased:
        labels = list(cards_purchased.keys())
        values = list(cards_purchased.values())
        card_colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 
                      'black': 'black', 'white': 'lightgray'}
        colors = [card_colors.get(label, 'gray') for label in labels]
        plt.bar(labels, values, color=colors)
        plt.title('Cards Purchased')
        plt.xlabel('Card Color')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'game_statistics.png'))
    plt.close()
    
    # Print summary statistics
    print("\nStrategy Analysis Summary:")
    print(f"Average game length: {sum(game_lengths)/len(game_lengths):.2f} steps")
    print(f"Average final points: {sum(points_per_game)/len(points_per_game):.2f}")
    print(f"Total nobles acquired: {nobles_acquired}")
    
    print("\nAction Type Preferences:")
    for action_type, freq in sorted(action_frequencies.items(), key=lambda x: -x[1]):
        print(f"  {action_type}: {freq/total*100:.1f}%")
    
    print("\nGem Color Preferences:")
    for color, count in sorted(gem_preferences.items(), key=lambda x: -x[1]):
        print(f"  {color}: {count/total_gems*100:.1f}%" if total_gems > 0 else f"  {color}: 0.0%")
    
    print("\nAction Type Evolution (Early → Mid → Late):")
    for action in action_types:
        early_pct = early_actions[action]/early_total*100 if early_total > 0 else 0
        mid_pct = mid_actions[action]/mid_total*100 if mid_total > 0 else 0
        late_pct = late_actions[action]/late_total*100 if late_total > 0 else 0
        print(f"  {action}: {early_pct:.1f}% → {mid_pct:.1f}% → {late_pct:.1f}%")
    
    print(f"\nAnalysis visualizations saved to {output_dir}")
    
    return {
        'action_frequencies': action_frequencies,
        'gem_preferences': gem_preferences,
        'game_lengths': game_lengths,
        'points_per_game': points_per_game,
        'avg_game_length': sum(game_lengths)/len(game_lengths) if game_lengths else 0,
        'avg_points': sum(points_per_game)/len(points_per_game) if points_per_game else 0,
        'nobles_acquired': nobles_acquired
    }

if __name__ == "__main__":
    analyze_model_strategy() 