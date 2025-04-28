import os
import sys
import torch
import numpy as np
import time

# Add lapidary to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lapidary.game import GameState

class GenericModelWrapper(torch.nn.Module):
    """
    A wrapper for any model to provide a consistent interface 
    regardless of the original training method
    """
    def __init__(self, model):
        super(GenericModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        """
        Forward pass through the network, handles different model output formats
        Returns: (action_probs, state_value)
        """
        # Try different calling conventions
        try:
            # Try standard forward call and unpack the result
            result = self.model(x)
            
            # If result is a tuple, assume (action_probs, value)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            # If it's just action_probs, return with fake value
            else:
                return result, torch.tensor([[0.0]])
        except Exception as e:
            print(f"Error in forward call: {e}")
            # Return a default result if model fails
            return torch.zeros((1, 100)), torch.tensor([[0.0]])
    
    def get_action_and_value(self, x, action_mask=None):
        """
        Get action, log_prob, probs, and value
        """
        # Get action probabilities using forward
        action_probs, value = self(x)
        
        # Apply mask if provided
        if action_mask is not None:
            masked_probs = action_probs.clone()
            masked_probs = masked_probs * action_mask.float()
            # Renormalize if any valid moves have non-zero probability
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = action_probs
        
        # Return dummy action and log_prob along with the actual probs and value
        return torch.tensor([0]), torch.tensor([0.0]), masked_probs, value

# Simple network for cases where we just need a model with the right dimensions
class SimpleNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(SimpleNetwork, self).__init__()
        # Create a simple feed-forward network
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        # Value head
        self.value_head = torch.nn.Linear(hidden_dim, 1)
        
        print(f"Initialized SimpleNetwork with input_dim: {input_dim}, output_dim: {output_dim}, hidden_dim: {hidden_dim}")
    
    def forward(self, x):
        # Extract features from shared layers
        features = self.network[:-1](x)
        # Get logits from last layer and convert to probabilities
        logits = self.network[-1](features)
        action_probs = torch.nn.functional.softmax(logits, dim=-1)
        # Get state value
        value = self.value_head(features)
        return action_probs, value

class ModelBotAI:
    """A model-based AI for Splendor that can use any trained model"""
    
    def __init__(self, model_path=None, input_dim=2300, output_dim=100):
        self.name = "ModelBot"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Load the model
        if model_path:
            print(f"Loading model from {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            try:
                # Try to load the model
                model_dict = torch.load(model_path, map_location=self.device)
                
                # Check what type of file we have
                if isinstance(model_dict, torch.nn.Module):
                    # It's a full model
                    self.model = GenericModelWrapper(model_dict).to(self.device)
                    print("Loaded full model directly")
                    self.model.eval()
                    return
                
                # Special case for files with network and optimizer keys
                if isinstance(model_dict, dict) and 'network' in model_dict and 'optimizer' in model_dict:
                    raise ValueError("Model file contains network and optimizer keys but not a state dict that can be loaded directly")
                
                # It's some kind of state dict
                if isinstance(model_dict, dict) and 'state_dict' in model_dict:
                    # It's a checkpoint with state_dict
                    model_state_dict = model_dict['state_dict']
                else:
                    # Assume it's a raw state dict
                    model_state_dict = model_dict
                
                # Handle special case where the state dict has 'network' key
                if isinstance(model_state_dict, dict) and 'network' in model_state_dict and isinstance(model_state_dict['network'], dict):
                    print("Found network key in state_dict, using it directly")
                    model_state_dict = model_state_dict['network']
                
                # Try to load with ActorCritic structure
                try:
                    # Try to import ActorCritic for backward compatibility
                    from Tims_splendor_agent.ppo_agent import ActorCritic
                    base_model = ActorCritic(input_dim=input_dim, output_dim=output_dim).to(self.device)
                    
                    # Try to load the state dict
                    try:
                        base_model.load_state_dict(model_state_dict)
                        print("Successfully loaded state dict with ActorCritic model")
                    except RuntimeError as e:
                        # Try adjusted state dict keys
                        adjusted_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                        try:
                            base_model.load_state_dict(adjusted_state_dict)
                            print("Successfully loaded adjusted state dict with ActorCritic model")
                        except RuntimeError:
                            raise RuntimeError(f"Could not load state dict with standard adjustments: {e}")
                except (ImportError, RuntimeError) as e:
                    print(f"Could not load with ActorCritic: {e}")
                    
                    # If the state dict has keys that match typical actor-critic structure
                    if all(k in model_state_dict for k in ['shared.0.weight', 'actor.weight']):
                        print("Detected actor-critic structure in state dict")
                        # Create a compatible model structure
                        class ActorCriticModel(torch.nn.Module):
                            def __init__(self, input_dim, output_dim):
                                super(ActorCriticModel, self).__init__()
                                self.shared = torch.nn.Sequential(
                                    torch.nn.Linear(input_dim, 256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 256),
                                    torch.nn.ReLU()
                                )
                                self.actor = torch.nn.Linear(256, output_dim)
                                self.critic = torch.nn.Linear(256, 1)
                            
                            def forward(self, x):
                                shared_features = self.shared(x)
                                action_probs = torch.nn.functional.softmax(self.actor(shared_features), dim=-1)
                                value = self.critic(shared_features)
                                return action_probs, value
                        
                        base_model = ActorCriticModel(input_dim, output_dim).to(self.device)
                        
                        # Try to load the state dict, adjusting keys if needed
                        try:
                            base_model.load_state_dict(model_state_dict)
                            print("Successfully loaded state dict with ActorCriticModel")
                        except RuntimeError:
                            # Try again with adjusted state_dict keys
                            adjusted_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
                            try:
                                base_model.load_state_dict(adjusted_state_dict)
                                print("Successfully loaded adjusted state dict with ActorCriticModel")
                            except RuntimeError as e:
                                raise RuntimeError(f"Could not load with ActorCriticModel: {e}")
                    else:
                        # Unable to load state dict - raise error
                        raise ValueError("State dict format is incompatible with known model architectures")
                        
                # Wrap the model
                self.model = GenericModelWrapper(base_model).to(self.device)
                self.model.eval()
                print("Model initialized")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        else:
            # No model path, so we can't load anything
            raise ValueError("No model path provided. A valid model path is required.")
    
    def make_move(self, state):
        """Make a move using the neural network model"""
        start_time = time.time()
        
        # Get valid moves
        valid_moves = state.get_valid_moves(state.current_player_index)
        if not valid_moves or len(valid_moves) == 0:
            print("No valid moves available!")
            return None, 0
        
        # Get state vector for the current player
        state_vector = state.get_state_vector(state.current_player_index)
        
        # Convert state vector to tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        # Create valid moves mask for network
        valid_moves_mask = torch.zeros(1, self.output_dim, dtype=torch.bool).to(self.device)
        for i in range(min(len(valid_moves), self.output_dim)):
            valid_moves_mask[0, i] = True
        
        # Use the model to get action probabilities
        with torch.no_grad():
            # Try to use get_action_and_value if the model has it
            try:
                _, _, action_probs, value = self.model.get_action_and_value(
                    state_tensor, action_mask=valid_moves_mask
                )
                # Extract numpy array
                action_probs = action_probs[0].cpu().numpy()
            except (AttributeError, TypeError):
                # Fall back to forward method
                action_probs, value = self.model(state_tensor)
                # Get probabilities as numpy array
                action_probs = action_probs[0].cpu().numpy()
            
            # Create a valid moves mask - assign zero probability to invalid moves
            valid_mask = np.zeros(action_probs.shape)
            for i in range(min(len(valid_moves), len(action_probs))):
                valid_mask[i] = 1.0
            
            # Apply mask
            masked_probs = action_probs * valid_mask
            
            # Normalize probabilities
            if np.sum(masked_probs) > 0:
                masked_probs = masked_probs / np.sum(masked_probs)
            else:
                # If model gives zero probability to all valid moves, use uniform distribution
                masked_probs = np.zeros_like(action_probs)
                masked_probs[:len(valid_moves)] = 1.0 / len(valid_moves)
            
            # Choose action based on probabilities
            move_index = np.argmax(masked_probs[:len(valid_moves)])
            
            # Get the selected move
            selected_move = valid_moves[move_index]
            
            print(f"Model selected move: {selected_move} (index {move_index})")
            print(f"Move selection took {time.time() - start_time:.2f} seconds")
            
            return selected_move, float(value.item())
    
    def choose_noble(self, state, nobles):
        """Choose a noble if multiple are available"""
        # If we have a valid model, use it to evaluate nobles
        if self.model:
            best_noble = nobles[0]
            best_score = float('-inf')
            
            try:
                for noble in nobles:
                    # Create a copy of the state with this noble
                    test_state = state.copy()
                    test_state.players[state.current_player_index].nobles.append(noble)
                    
                    # Get state vector
                    state_vector = test_state.get_state_vector(state.current_player_index)
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                    
                    # Evaluate state with this noble
                    with torch.no_grad():
                        try:
                            # Try to use get_action_and_value
                            valid_moves = test_state.get_valid_moves(test_state.current_player_index)
                            valid_moves_mask = torch.zeros(1, self.output_dim, dtype=torch.bool).to(self.device)
                            for i in range(min(len(valid_moves), self.output_dim)):
                                valid_moves_mask[0, i] = True
                                
                            _, _, _, value = self.model.get_action_and_value(
                                state_tensor, action_mask=valid_moves_mask
                            )
                        except (AttributeError, TypeError):
                            # Fall back to forward
                            _, value = self.model(state_tensor)
                            
                        score = float(value.item())
                    
                    if score > best_score:
                        best_score = score
                        best_noble = noble
                
                return best_noble
            except Exception as e:
                print(f"Error evaluating nobles: {e}")
                return nobles[0]
        
        # Fallback to first noble if model evaluation fails
        return nobles[0]

# For backward compatibility
PPOBotAI = ModelBotAI

def main():
    """
    Example of how to use the model agent as a bot
    """
    # Import required modules
    from lapidary.ais import GameManager
    from lapidary.ais import RandomAI
    
    # Create model bot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'ppo_checkpoint_final.pt')
    model_bot = ModelBotAI(model_path=model_path)
    
    # Play against RandomAI
    game_manager = GameManager(players=2, ais=[model_bot, RandomAI()], end_score=15)
    game_info = game_manager.run_game(verbose=True)
    
    # Print game results
    print("\nGame finished!")
    print(f"Winner: Player {game_info.winner_index}")
    print(f"Game length: {game_info.length} rounds")
    print(f"Winner cards: {game_info.winner_num_bought} cards bought")
    
if __name__ == "__main__":
    main() 