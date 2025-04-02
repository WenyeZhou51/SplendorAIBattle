import sys
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add lapidary to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lapidary.game import GameState, colours
from lapidary.aibase import MoveInfo

class SplendorEnv(gym.Env):
    """
    Gymnasium Environment for Splendor board game.
    This environment allows training reinforcement learning agents through self-play.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, num_players=2, render_mode=None):
        self.num_players = num_players
        self.render_mode = render_mode
        
        # Observation space is the state vector of the game
        # The size of the state vector is 2300 as specified in the nn.py file
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2300,), dtype=np.float32)
        
        # Action space - maximum number of possible actions in Splendor
        # Based on examining the game logic, a safe upper bound would be around 100 possible moves
        self.action_space = spaces.Discrete(100)
        
        # Initialize the game state
        self.game_state = None
        self.valid_moves = []
        self.valid_moves_mapping = {}
        
    def _get_obs(self):
        """Get the observation of the current state from the active player's perspective."""
        return self.game_state.get_state_vector(self.game_state.current_player_index)
    
    def _get_info(self):
        """Get additional information about the current state."""
        return {
            'scores': [player.score for player in self.game_state.players],
            'current_player': self.game_state.current_player_index,
            'num_moves': len(self.valid_moves)
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize a new game
        self.game_state = GameState(players=self.num_players, init_game=True, validate=True)
        
        # Add compatibility methods to handle inconsistent naming
        self._add_compatibility_methods()
        
        # Get valid moves for the current player
        self.valid_moves = self.game_state.get_valid_moves(self.game_state.current_player_index)
        self._update_valid_moves_mapping()
        
        return self._get_obs(), self._get_info()
    
    def _update_valid_moves_mapping(self):
        """Create a mapping from action indices to valid moves."""
        self.valid_moves_mapping = {}
        for i, move in enumerate(self.valid_moves):
            self.valid_moves_mapping[i] = move
    
    def _add_compatibility_methods(self):
        """Add compatibility methods to handle inconsistent naming in the codebase."""
        # Add num_cards_of_color method to Player class that calls num_cards_of_colour
        for player in self.game_state.players:
            if not hasattr(player, 'num_cards_of_color'):
                player.num_cards_of_color = player.num_cards_of_colour

    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action (int): Index of the move to make from valid_moves_mapping
        
        Returns:
            observation: The new state observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Ensure the action is valid
        if action not in self.valid_moves_mapping:
            # Instead of choosing a random action, use a penalty and return the current state
            # This way the agent learns not to choose invalid actions
            print(f"Warning: Invalid action {action} provided. Valid actions: {list(self.valid_moves_mapping.keys())}")
            return self._get_obs(), -1.0, False, False, {
                'scores': [player.score for player in self.game_state.players],
                'current_player': self.game_state.current_player_index,
                'num_moves': len(self.valid_moves),
                'invalid_action': True
            }
        
        # Get the actual move
        move = self.valid_moves_mapping[action]
        
        # Check if move would cause player to exceed gem limit and fix it if needed
        # In Splendor, no player can have more than 10 gems total
        if move[0] == 'gems' or move[0] == 'reserve':
            current_player = self.game_state.players[self.game_state.current_player_index]
            
            # Calculate how many gems the player would have after this move
            gems_gained = {}
            if move[0] == 'gems':
                gems_gained = move[1]
            elif move[0] == 'reserve' and len(move) > 3:
                gems_gained = move[3]
            
            # Count total gems after move
            total_gems = current_player.total_num_gems
            for color, count in gems_gained.items():
                total_gems += count
            
            # If exceeds 10, modify the move to discard gems
            if total_gems > 10:
                print(f"Warning: Move would cause player to exceed 10 gems limit. Modifying move.")
                # This will get handled in the game's move validation
        
        # Execute the move
        old_scores = [player.score for player in self.game_state.players]
        old_gems = [sum(player.gems.values()) for player in self.game_state.players]
        old_cards = [len(player.cards_played) for player in self.game_state.players]
        old_reserved = [len(player.cards_in_hand) for player in self.game_state.players]
        current_player = self.game_state.current_player_index
        
        # Apply move and check for state verification errors
        state_verification_success = True
        try:
            # Temporarily store current verify_state setting
            old_verify = self.game_state._verify_state if hasattr(self.game_state, '_verify_state') else True
            # Temporarily disable automatic verification
            self.game_state._verify_state = False
            # Apply the move without verification
            self.game_state.make_move(move, refill_market=True, increment_player=True)
            
            # Fix player gem counts if they exceed 10
            for player in self.game_state.players:
                if player.total_num_gems > 10:
                    # In a real game, the player would choose which gems to discard
                    # For simplicity in the environment, we'll discard gems automatically
                    # starting with the most common ones
                    gems_to_discard = player.total_num_gems - 10
                    
                    # Sort colors by amount (descending)
                    colors_by_amount = sorted(
                        [(c, player.num_gems(c)) for c in colours + ['gold'] if player.num_gems(c) > 0],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Discard gems
                    for color, amount in colors_by_amount:
                        # Skip gold if possible (it's more valuable)
                        if color == 'gold' and gems_to_discard < player.total_num_gems - player.num_gems('gold'):
                            continue
                            
                        discard = min(amount, gems_to_discard)
                        if discard > 0:
                            # Return gems to supply
                            self.game_state.set_gems_available(color, 
                                self.game_state.num_gems_available(color) + discard)
                            player.set_gems(color, player.num_gems(color) - discard)
                            gems_to_discard -= discard
                            
                        if gems_to_discard == 0:
                            break
            
            # Explicitly run verification
            state_verification_success = self.game_state.verify_state()
            # Restore original verification setting
            self.game_state._verify_state = old_verify
        except Exception as e:
            state_verification_success = False
        
        # Ensure compatibility methods are available after state changes
        self._add_compatibility_methods()
        
        # Get the new valid moves for the current player
        self.valid_moves = self.game_state.get_valid_moves(self.game_state.current_player_index)
        self._update_valid_moves_mapping()
        
        # Check if game is over
        terminated = any(player.score >= 15 for player in self.game_state.players)
        
        # Calculate base reward: difference in score for the player who just made the move
        new_scores = [player.score for player in self.game_state.players]
        new_gems = [sum(player.gems.values()) for player in self.game_state.players]
        new_cards = [len(player.cards_played) for player in self.game_state.players]
        new_reserved = [len(player.cards_in_hand) for player in self.game_state.players]
        
        # Make rewards more zero-sum and reduce impact of small rewards
        
        # Base reward is score improvement (reduced by factor of 10)
        reward = (new_scores[current_player] - old_scores[current_player]) * 0.1
        
        # Add smaller rewards for progress toward acquiring cards (reduced by factor of 10)
        # Reward for collecting gems
        if new_gems[current_player] > old_gems[current_player]:
            reward += 0.01  # Was 0.1
        
        # Reward for buying cards
        if new_cards[current_player] > old_cards[current_player]:
            reward += 0.03  # Was 0.3
        
        # Reward for reserving cards
        if new_reserved[current_player] > old_reserved[current_player]:
            reward += 0.01  # Was 0.1
            
        # Small penalty for passing (reduced by factor of 10)
        if (reward == 0 and 
            new_gems[current_player] == old_gems[current_player] and
            new_cards[current_player] == old_cards[current_player] and
            new_reserved[current_player] == old_reserved[current_player]):
            reward -= 0.005  # Was 0.05
        
        # If game is over, give equal reward to winner and penalty to loser (zero-sum)
        if terminated:
            max_score = max(new_scores)
            winners = [i for i, score in enumerate(new_scores) if score == max_score]
            
            # In case of a tie, prioritize by fewer cards
            if len(winners) > 1:
                min_cards = min(len(self.game_state.players[i].cards_played) for i in winners)
                winners = [i for i in winners if len(self.game_state.players[i].cards_played) == min_cards]
            
            # Equal magnitude for win/loss (zero-sum): +10 for winners, -10 for losers
            for i in range(self.num_players):
                if i == current_player:
                    if i in winners:
                        reward = 10.0  # Fixed value for winning
                    else:
                        reward = -10.0  # Equal magnitude penalty for losing
        
        # Create info dictionary including verification status
        info = self._get_info()
        info['state_verification_error'] = not state_verification_success
        
        return self._get_obs(), reward, terminated, False, info
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            print("\n==== Splendor Game State ====")
            print(f"Current Player: {self.game_state.current_player_index}")
            print(f"Scores: {[player.score for player in self.game_state.players]}")
            print(f"Available moves: {len(self.valid_moves)}")
            # Add more detailed rendering if needed
            
    def close(self):
        """Close the environment."""
        pass 