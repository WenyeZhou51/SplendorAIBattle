import sys
import os
import random
import numpy as np
import copy
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lapidary.game import GameState, colours
from lapidary.aibase import MoveInfo
from splendor_env import SplendorEnv

class EnhancedSplendorEnv(SplendorEnv):
    """
    Enhanced Splendor Environment with improved reward function
    featuring strategic rewards for important game concepts.
    """
    
    def __init__(self, num_players=2, render_mode=None, terminal_reward_scale=3.0):
        super().__init__(num_players, render_mode)
        # Scale for terminal rewards (win/loss) - reduced from 10.0 to balance with strategic rewards
        self.terminal_reward_scale = terminal_reward_scale
        
    def step(self, action):
        """
        Execute an action in the environment with enhanced reward calculation.
        
        Args:
            action (int): Index of the move to make from valid_moves_mapping
        
        Returns:
            observation: The new state observation
            reward: Enhanced reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Capture pre-move state for reward calculation
        pre_move_state = self._get_strategic_state()
        
        # Execute the standard environment step
        observation, basic_reward, terminated, truncated, info = super().step(action)
        
        # Capture post-move state for reward calculation
        post_move_state = self._get_strategic_state()
        
        # Calculate enhanced reward
        enhanced_reward = self._calculate_enhanced_reward(
            pre_move_state, 
            post_move_state, 
            basic_reward,
            terminated,
            info
        )
        
        # Update info with reward components for debugging
        info['reward_components'] = post_move_state['reward_components']
        
        return observation, enhanced_reward, terminated, truncated, info
    
    def _get_strategic_state(self):
        """Extract strategic information from the current game state"""
        state = {}
        current_player = self.game_state.current_player_index
        player = self.game_state.players[current_player]
        
        # Card color counts
        color_counts = {}
        for color in colours:
            color_counts[color] = player.num_cards_of_colour(color)
        state['color_counts'] = color_counts
        
        # Gem counts
        state['gems'] = {color: player.num_gems(color) for color in colours + ['gold']}
        
        # Score
        state['score'] = player.score
        
        # Card count
        state['cards_played'] = len(player.cards_played)
        state['cards_reserved'] = len(player.cards_in_hand)
        
        # Nobles progress - calculate distance to each noble
        nobles_distance = {}
        for i, noble in enumerate(self.game_state.nobles):
            distance = 0
            for color in colours:
                color_requirement = noble.num_required(color)
                current_cards = color_counts[color]
                if color_requirement > current_cards:
                    distance += color_requirement - current_cards
            nobles_distance[i] = distance
        state['nobles_distance'] = nobles_distance
        
        # Track minimum distance to any noble
        state['min_noble_distance'] = min(nobles_distance.values()) if nobles_distance else float('inf')
        
        # Store components to explain reward
        state['reward_components'] = {}
        
        return state
    
    def _calculate_enhanced_reward(self, pre_state, post_state, basic_reward, terminated, info):
        """
        Calculate enhanced reward based on strategic game concepts.
        
        This reward function emphasizes:
        1. Card color concentration (combos)
        2. Progress toward nobles
        3. Efficient gem-to-card conversion
        4. Balanced terminal rewards
        """
        reward = 0.0
        reward_components = post_state['reward_components']
        
        # 1. Base reward component (from score difference) - keep this
        score_diff = post_state['score'] - pre_state['score']
        reward += score_diff * 0.5  # Increased from 0.1
        reward_components['score'] = score_diff * 0.5
        
        # 2. Reward for color concentration (collecting sets of same color)
        color_concentration_reward = 0.0
        for color in colours:
            pre_count = pre_state['color_counts'][color]
            post_count = post_state['color_counts'][color]
            
            if post_count > pre_count:
                # More reward for higher concentrations (quadratic scaling)
                # e.g., going from 2->3 cards is better than 0->1
                concentration_bonus = post_count * 0.1
                color_concentration_reward += concentration_bonus
        
        reward += color_concentration_reward
        reward_components['color_concentration'] = color_concentration_reward
        
        # 3. Reward for approaching nobles
        noble_progress_reward = 0.0
        if pre_state['min_noble_distance'] > post_state['min_noble_distance']:
            # Reward for getting closer to a noble
            distance_reduction = pre_state['min_noble_distance'] - post_state['min_noble_distance']
            noble_progress_reward = distance_reduction * 0.3
        reward += noble_progress_reward
        reward_components['noble_progress'] = noble_progress_reward
        
        # 4. Reward for card acquisition efficiency
        efficiency_reward = 0.0
        if post_state['cards_played'] > pre_state['cards_played']:
            # Calculate how many gems were spent
            gems_spent = 0
            for color in colours + ['gold']:
                diff = pre_state['gems'][color] - post_state['gems'][color]
                if diff > 0:  # Only count gems that were spent
                    gems_spent += diff
            
            # An efficient purchase uses fewer gems
            # Base efficiency is 0.4, reduced if more gems were spent
            if gems_spent > 0:
                efficiency_reward = 0.4 * max(0, (8 - gems_spent) / 8)
            else:
                # Free card (likely from reserving with enough discounts)
                efficiency_reward = 0.6  # Bonus for getting a "free" card
                
        reward += efficiency_reward
        reward_components['efficiency'] = efficiency_reward
        
        # 5. Smaller reward for reserving cards 
        reservation_reward = 0.0
        if post_state['cards_reserved'] > pre_state['cards_reserved']:
            # Check if gold was gained (strategic reservation)
            if post_state['gems']['gold'] > pre_state['gems']['gold']:
                reservation_reward = 0.2  # Better reward for getting gold
            else:
                reservation_reward = 0.1  # Basic reservation
                
        reward += reservation_reward
        reward_components['reservation'] = reservation_reward
        
        # 6. Penalty for non-productive turns
        if (reward == 0 and 
            post_state['gems'] == pre_state['gems'] and
            post_state['cards_played'] == pre_state['cards_played'] and
            post_state['cards_reserved'] == pre_state['cards_reserved']):
            reward -= 0.3  # Reduced from -1.0
            reward_components['non_productive'] = -0.3
        
        # 7. Terminal rewards (reduced from original)
        if terminated:
            scores = info['scores']
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
            current_player = self.game_state.current_player_index
            
            # Equal magnitude for win/loss (but scaled down)
            if current_player in winners:
                reward += self.terminal_reward_scale  # Win reward (3.0 default)
                reward_components['terminal_win'] = self.terminal_reward_scale
            else:
                reward -= self.terminal_reward_scale  # Loss penalty (3.0 default)
                reward_components['terminal_loss'] = -self.terminal_reward_scale
        
        # Store total reward for debugging
        reward_components['total'] = reward
        post_state['reward_components'] = reward_components
        
        return reward 