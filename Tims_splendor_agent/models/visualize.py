import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
import argparse

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from parallel_rl_agent import ParallelPPOAgent
from splendor_env import SplendorEnv

def visualize_network_weights(model_path, output_dir="./visualizations"):
    """
    Visualize and save heatmaps of the model's weights for different components
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    
    # Create a custom colormap: blue for negative, white for zero, red for positive
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list("bwr", colors, N=256)
    
    # List of network components to visualize
    components = [
        ("Card Encoder First Layer", agent.policy.card_encoder[0].weight.data),
        ("Card Encoder Last Layer", agent.policy.card_encoder[3].weight.data),
        ("Main Encoder First Layer", agent.policy.main_encoder[0].weight.data),
        ("Shared Layer", agent.policy.shared[0].weight.data),
        ("Actor (Policy) Layer", agent.policy.actor.weight.data),
        ("Critic (Value) Layer", agent.policy.critic.weight.data)
    ]
    
    # Create weight visualizations
    print("Generating weight visualizations...")
    for name, weights in components:
        plt.figure(figsize=(12, 8))
        weights_np = weights.detach().cpu().numpy()
        
        # Handle large matrices by sampling or using a different visualization
        if weights_np.shape[0] > 100 or weights_np.shape[1] > 100:
            # Downsample by taking every nth element
            step_x = max(1, weights_np.shape[0] // 100)
            step_y = max(1, weights_np.shape[1] // 100)
            weights_np = weights_np[::step_x, ::step_y]
        
        # Scale the image to use full colormap
        vmax = max(abs(weights_np.min()), abs(weights_np.max()))
        vmin = -vmax
        
        plt.imshow(weights_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f'{name} Weights ({weights_np.shape[0]}x{weights_np.shape[1]})')
        plt.xlabel('Input Features')
        plt.ylabel('Neurons')
        plt.tight_layout()
        
        # Save the figure
        filename = name.lower().replace(" ", "_") + "_weights.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    # Calculate and visualize weight statistics
    all_weights = []
    weight_stats = {}
    
    for name, weights in components:
        weights_np = weights.detach().cpu().numpy().flatten()
        all_weights.extend(weights_np)
        
        weight_stats[name] = {
            'mean': np.mean(weights_np),
            'std': np.std(weights_np),
            'min': np.min(weights_np),
            'max': np.max(weights_np),
            'zeros': np.sum(weights_np == 0) / len(weights_np)
        }
    
    # Overall distribution of weights
    plt.figure(figsize=(10, 6))
    sns.histplot(all_weights, kde=True, bins=50)
    plt.title('Overall Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_weight_distribution.png'))
    plt.close()
    
    # Weight statistics table
    print("\nWeight Statistics:")
    print(f"{'Component':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Zeros %':>10}")
    print("-" * 80)
    for name, stats in weight_stats.items():
        print(f"{name:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f} {stats['zeros']*100:>10.2f}%")
    
    print(f"\nVisualizations saved to {output_dir}")

def analyze_activations(model_path, num_samples=5, output_dir="./visualizations"):
    """
    Analyze and visualize activation patterns using random game states
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and environment
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    env = SplendorEnv(num_players=2)
    
    # Collect sample states
    print(f"Collecting {num_samples} sample states for activation analysis...")
    states = []
    for _ in range(num_samples):
        state, _ = env.reset()
        states.append(state)
    
    # Register hooks to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # For 1D activation output, reshape to 2D for better visualization
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks for the different components
    hooks = [
        agent.policy.card_encoder[2].register_forward_hook(get_activation('card_encoder_relu1')),
        agent.policy.card_encoder[5].register_forward_hook(get_activation('card_encoder_relu2')),
        agent.policy.main_encoder[2].register_forward_hook(get_activation('main_encoder_relu1')),
        agent.policy.shared[2].register_forward_hook(get_activation('shared_relu1')),
        agent.policy.shared[5].register_forward_hook(get_activation('shared_relu2')),
        agent.policy.shared[8].register_forward_hook(get_activation('shared_relu3')),
    ]
    
    # Process each state and record activations
    for i, state in enumerate(states):
        # Forward pass to trigger hooks
        with torch.no_grad():
            action_probs, value = agent.policy(torch.FloatTensor(state))
        
        # Visualize activations for this state
        for name, activation in activations.items():
            plt.figure(figsize=(12, 6))
            
            # Get activation values
            act_np = activation.numpy()[0]  # Get first batch item
            
            # Handle large activations by sampling
            if act_np.shape[0] > 100:
                step = max(1, act_np.shape[0] // 100)
                act_np = act_np[::step]
            
            # Visualize as bar chart for 1D or heatmap for 2D
            if len(act_np.shape) == 1 or act_np.shape[1] == 1:
                plt.bar(range(len(act_np)), act_np)
                plt.title(f'Activation Pattern: {name} (Sample {i+1})')
                plt.xlabel('Neuron Index')
                plt.ylabel('Activation Value')
            else:
                plt.imshow(act_np, aspect='auto', cmap='viridis')
                plt.colorbar(label='Activation Value')
                plt.title(f'Activation Pattern: {name} (Sample {i+1})')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Neuron Index')
            
            plt.tight_layout()
            filename = f"{name}_sample_{i+1}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Activation visualizations saved to {output_dir}")

def analyze_decision_making(model_path, num_games=3, output_dir="./visualizations"):
    """
    Analyze the agent's decision-making process by tracking its choices during gameplay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and environment
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    env = SplendorEnv(num_players=2)
    
    # Tracking data
    action_frequencies = defaultdict(int)
    action_values = defaultdict(list)
    gem_color_preferences = {"red": 0, "blue": 0, "green": 0, "black": 0, "white": 0, "gold": 0}
    card_tier_preferences = {1: 0, 2: 0, 3: 0}
    card_color_preferences = {"red": 0, "blue": 0, "green": 0, "black": 0, "white": 0}
    game_lengths = []
    
    print(f"Analyzing decision making over {num_games} games...")
    for game in range(num_games):
        state, _ = env.reset()
        game_length = 0
        done = False
        
        while not done and game_length < 200:  # Prevent infinite loops
            # Get valid moves
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            
            # Get action probabilities directly
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action_probs, value = agent.policy(state_tensor)
                masked_probs = action_probs * torch.FloatTensor(valid_moves_mask)
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
            
            # Track top action preferences
            top_actions = torch.topk(masked_probs, min(5, len(valid_moves)))
            for i, idx in enumerate(top_actions.indices):
                action_idx = idx.item()
                if action_idx in env.valid_moves_mapping:
                    move = env.valid_moves_mapping[action_idx]
                    move_type = move[0]  # First element is move type ('gems', 'card', 'reserve')
                    
                    # Record action type frequency
                    action_frequencies[move_type] += top_actions.values[i].item()
                    
                    # Record action value estimate
                    action_values[move_type].append(value.item())
                    
                    # Record specific preferences
                    if move_type == 'gems' and len(move) > 1:
                        for color, count in move[1].items():
                            if color in gem_color_preferences and count > 0:
                                gem_color_preferences[color] += count
                    
                    elif move_type == 'card' and len(move) > 1:
                        card = move[1]
                        if hasattr(card, 'tier'):
                            card_tier_preferences[card.tier] += 1
                        if hasattr(card, 'colour'):
                            card_color_preferences[card.colour] += 1
            
            # Take actual action
            action, _ = agent.get_action(state, valid_moves_mask)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            game_length += 1
            done = terminated or truncated
        
        game_lengths.append(game_length)
        print(f"Game {game+1} completed in {game_length} steps")
    
    # Generate visualizations
    
    # 1. Action type preferences
    plt.figure(figsize=(10, 6))
    total = sum(action_frequencies.values())
    labels = list(action_frequencies.keys())
    values = [freq/total*100 for freq in action_frequencies.values()]
    plt.bar(labels, values, color=['blue', 'green', 'orange'])
    plt.title('Action Type Preferences')
    plt.xlabel('Action Type')
    plt.ylabel('Preference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_type_preferences.png'))
    plt.close()
    
    # 2. Gem color preferences
    plt.figure(figsize=(10, 6))
    total_gems = sum(gem_color_preferences.values())
    labels = list(gem_color_preferences.keys())
    values = [count/total_gems*100 if total_gems > 0 else 0 for count in gem_color_preferences.values()]
    colors = ['red', 'blue', 'green', 'black', 'gray', 'gold']
    plt.bar(labels, values, color=colors)
    plt.title('Gem Color Preferences')
    plt.xlabel('Gem Color')
    plt.ylabel('Preference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gem_color_preferences.png'))
    plt.close()
    
    # 3. Card tier preferences
    plt.figure(figsize=(10, 6))
    total_cards = sum(card_tier_preferences.values())
    labels = list(card_tier_preferences.keys())
    values = [count/total_cards*100 if total_cards > 0 else 0 for count in card_tier_preferences.values()]
    plt.bar(labels, values, color=['lightblue', 'royalblue', 'darkblue'])
    plt.title('Card Tier Preferences')
    plt.xlabel('Card Tier')
    plt.ylabel('Preference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'card_tier_preferences.png'))
    plt.close()
    
    # 4. Card color preferences
    plt.figure(figsize=(10, 6))
    total_card_colors = sum(card_color_preferences.values())
    labels = list(card_color_preferences.keys())
    values = [count/total_card_colors*100 if total_card_colors > 0 else 0 for count in card_color_preferences.values()]
    colors = ['red', 'blue', 'green', 'black', 'gray']
    plt.bar(labels, values, color=colors)
    plt.title('Card Color Preferences')
    plt.xlabel('Card Color')
    plt.ylabel('Preference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'card_color_preferences.png'))
    plt.close()
    
    # 5. Estimated action values
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for action_type, values in action_values.items():
        if values:
            data.append(values)
            labels.append(action_type)
    
    plt.boxplot(data, labels=labels)
    plt.title('Estimated Value by Action Type')
    plt.xlabel('Action Type')
    plt.ylabel('Estimated Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_value_estimates.png'))
    plt.close()
    
    # 6. Game length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(game_lengths, bins=10)
    plt.title('Game Length Distribution')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'game_length_distribution.png'))
    plt.close()
    
    # Print summary statistics
    print("\nDecision Analysis Summary:")
    print(f"Average game length: {sum(game_lengths)/len(game_lengths):.2f} steps")
    
    print("\nAction Type Preferences:")
    for action_type, freq in action_frequencies.items():
        print(f"  {action_type}: {freq/total*100:.1f}%")
    
    print("\nGem Color Preferences:")
    for color, count in gem_color_preferences.items():
        if total_gems > 0:
            print(f"  {color}: {count/total_gems*100:.1f}%")
        else:
            print(f"  {color}: 0.0%")
    
    print("\nDecision analysis visualizations saved to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and analyze a trained Splendor PPO model')
    parser.add_argument('--model_path', type=str, default='parallel_splendor_agent.pt', 
                      help='Path to the model file (.pt)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of samples for activation analysis')
    parser.add_argument('--games', type=int, default=3,
                      help='Number of games for decision analysis')
    parser.add_argument('--all', action='store_true',
                      help='Run all analyses')
    parser.add_argument('--weights', action='store_true',
                      help='Visualize network weights')
    parser.add_argument('--activations', action='store_true',
                      help='Analyze activation patterns')
    parser.add_argument('--decisions', action='store_true',
                      help='Analyze decision making')
    
    args = parser.parse_args()
    
    # Make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If no specific analyses are requested, run weights visualization by default
    if not (args.all or args.weights or args.activations or args.decisions):
        args.weights = True
    
    # Run requested analyses
    if args.all or args.weights:
        print("\n=== Visualizing Network Weights ===")
        visualize_network_weights(args.model_path, args.output_dir)
    
    if args.all or args.activations:
        print("\n=== Analyzing Activation Patterns ===")
        analyze_activations(args.model_path, args.samples, args.output_dir)
    
    if args.all or args.decisions:
        print("\n=== Analyzing Decision Making ===")
        analyze_decision_making(args.model_path, args.games, args.output_dir)
    
    print("\nAll analyses complete!") 