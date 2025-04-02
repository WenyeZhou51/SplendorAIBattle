import sys
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add lapidary to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lapidary.game import GameState
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
        
        # Get valid moves for the current player
        self.valid_moves = self.game_state.get_valid_moves(self.game_state.current_player_index)
        self._update_valid_moves_mapping()
        
        return self._get_obs(), self._get_info()
    
    def _update_valid_moves_mapping(self):
        """Create a mapping from action indices to valid moves."""
        self.valid_moves_mapping = {}
        for i, move in enumerate(self.valid_moves):
            self.valid_moves_mapping[i] = move
    
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
        
        # Execute the move
        old_scores = [player.score for player in self.game_state.players]
        old_gems = [sum(player.gems.values()) for player in self.game_state.players]
        old_cards = [len(player.cards_played) for player in self.game_state.players]
        old_reserved = [len(player.cards_in_hand) for player in self.game_state.players]
        current_player = self.game_state.current_player_index
        
        # Apply move
        self.game_state.make_move(move)
        
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
        
        # Base reward is score improvement
        reward = new_scores[current_player] - old_scores[current_player]
        
        # Add smaller rewards for progress toward acquiring cards
        # Reward for collecting gems
        if new_gems[current_player] > old_gems[current_player]:
            reward += 0.1
        
        # Reward for buying cards
        if new_cards[current_player] > old_cards[current_player]:
            reward += 0.3
        
        # Reward for reserving cards
        if new_reserved[current_player] > old_reserved[current_player]:
            reward += 0.1
            
        # Small penalty for passing (which is often represented as a move that doesn't change state)
        if (reward == 0 and 
            new_gems[current_player] == old_gems[current_player] and
            new_cards[current_player] == old_cards[current_player] and
            new_reserved[current_player] == old_reserved[current_player]):
            reward -= 0.05
        
        # If game is over, give a larger reward to the winner
        if terminated:
            max_score = max(new_scores)
            winners = [i for i, score in enumerate(new_scores) if score == max_score]
            
            # In case of a tie, prioritize by fewer cards
            if len(winners) > 1:
                min_cards = min(len(self.game_state.players[i].cards_played) for i in winners)
                winners = [i for i in winners if len(self.game_state.players[i].cards_played) == min_cards]
            
            # Give +10 reward to winner(s), -5 to losers
            for i in range(self.num_players):
                if i == current_player:
                    if i in winners:
                        reward = 10
                    else:
                        reward = -5
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
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