import os
import sys
import torch
import numpy as np
import importlib
import importlib.util
from abc import ABC, abstractmethod

# Add paths to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'Tims_splendor_agent'))

# Abstract base class for model adapters
class ModelAdapter(ABC):
    """Base class that all model adapters must implement"""
    
    def __init__(self, model_path, input_dim=2300, output_dim=100):
        self.model_path = model_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self):
        """Load the model from the specified path"""
        pass
    
    @abstractmethod
    def predict_move(self, state, valid_moves):
        """
        Predict the best move for the current state
        
        Args:
            state: GameState object representing the current game state
            valid_moves: List of valid moves to choose from
            
        Returns:
            selected_move: The selected move
            confidence: Confidence value for the selected move (optional)
        """
        pass
    
    def get_info(self):
        """Return information about the loaded model"""
        return {
            "model_type": self.__class__.__name__,
            "model_path": self.model_path,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "loaded": self.model is not None
        }


class PPOModelAdapter(ModelAdapter):
    """Adapter for the original PPO models"""
    
    def load_model(self):
        """Load the model using the ModelBotAI implementation"""
        try:
            # Import here to avoid circular imports
            from Tims_splendor_agent.model_bot import ModelBotAI
            
            # Use the existing ModelBotAI to load the model
            model_bot = ModelBotAI(
                model_path=self.model_path,
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )
            
            self.model = model_bot
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load PPO model: {str(e)}")
    
    def predict_move(self, state, valid_moves):
        """Use the ModelBotAI to predict a move"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use the existing make_move method
        selected_move, value = self.model.make_move(state)
        return selected_move, float(value)


class ParallelPPOAdapter(ModelAdapter):
    """Adapter for the parallel PPO agent models"""
    
    def load_model(self):
        """Load a model trained with the ParallelPPOAgent"""
        try:
            # Dynamically import the ParallelPPOAgent class
            try:
                from Tims_splendor_agent.parallel_rl_agent import ParallelPPOAgent, ActorCritic
            except ImportError:
                print("Error importing ParallelPPOAgent, trying dynamic import")
                # Alternative dynamic import if the normal import fails
                spec = importlib.util.spec_from_file_location(
                    "parallel_rl_agent",
                    os.path.join(parent_dir, "Tims_splendor_agent", "parallel_rl_agent.py")
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ParallelPPOAgent = module.ParallelPPOAgent
                ActorCritic = module.ActorCritic
            
            # Create a ParallelPPOAgent with 1 worker (for inference only)
            agent = ParallelPPOAgent(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                num_workers=1
            )
            
            # Load the model
            agent.load_model(self.model_path)
            self.model = agent
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Parallel PPO agent: {str(e)}")
    
    def predict_move(self, state, valid_moves):
        """Predict a move using the ParallelPPOAgent"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create a valid moves mask for the agent
        valid_moves_mask = [0] * self.output_dim
        for i, move in enumerate(valid_moves):
            if i < self.output_dim:
                valid_moves_mask[i] = 1
        
        # Get state vector for the current player
        state_vector = state.get_state_vector(state.current_player_index)
        
        try:
            # Get action using the ParallelPPOAgent's get_action method
            action, log_prob = self.model.get_action(state_vector, valid_moves_mask)
            
            # Return the selected move and a dummy value (ParallelPPOAgent doesn't return state value)
            return valid_moves[action], 0.0
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to predict move with Parallel PPO agent: {str(e)}")


class GenericModelAdapter(ModelAdapter):
    """Adapter for standard PyTorch models with a simple interface"""
    
    def load_model(self):
        """Load any generic PyTorch model"""
        try:
            model_dict = torch.load(self.model_path, map_location=self.device)
            
            # If it's a full model (nn.Module instance)
            if isinstance(model_dict, torch.nn.Module):
                self.model = model_dict
            else:
                # Handle state dict loading
                from Tims_splendor_agent.model_bot import GenericModelWrapper, SimpleNetwork
                
                # Create a simple network with the right dimensions
                base_model = SimpleNetwork(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim
                ).to(self.device)
                
                # Try to load weights from state dict
                try:
                    if isinstance(model_dict, dict) and 'state_dict' in model_dict:
                        base_model.load_state_dict(model_dict['state_dict'])
                    else:
                        base_model.load_state_dict(model_dict)
                except Exception as e:
                    print(f"Warning: Could not load state dict directly: {e}")
                    # Try with module prefix adjustments
                    adjusted_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
                    base_model.load_state_dict(adjusted_dict)
                
                # Wrap the model
                self.model = GenericModelWrapper(base_model).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load generic model: {str(e)}")
    
    def predict_move(self, state, valid_moves):
        """Make a prediction with a generic model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get state vector for the current player
        state_vector = state.get_state_vector(state.current_player_index)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        # Create valid moves mask
        valid_moves_mask = torch.zeros(1, self.output_dim, dtype=torch.bool).to(self.device)
        for i in range(min(len(valid_moves), self.output_dim)):
            valid_moves_mask[0, i] = True
        
        # Get prediction
        with torch.no_grad():
            try:
                # Try to use get_action_and_value if available
                _, _, action_probs, value = self.model.get_action_and_value(
                    state_tensor, action_mask=valid_moves_mask
                )
                action_probs = action_probs[0].cpu().numpy()
            except (AttributeError, TypeError):
                # Fall back to forward method
                action_probs, value = self.model(state_tensor)
                action_probs = action_probs[0].cpu().numpy()
            
            # Apply valid moves mask
            masked_probs = np.zeros_like(action_probs)
            for i in range(min(len(valid_moves), len(action_probs))):
                masked_probs[i] = action_probs[i]
            
            # Normalize probabilities
            if np.sum(masked_probs) > 0:
                masked_probs = masked_probs / np.sum(masked_probs)
            else:
                # If zero probabilities, use uniform distribution
                masked_probs[:len(valid_moves)] = 1.0 / len(valid_moves)
            
            # Choose action with highest probability
            move_index = np.argmax(masked_probs[:len(valid_moves)])
            
            # Return selected move
            return valid_moves[move_index], float(value.item())


# Create a registry of available adapters
MODEL_ADAPTERS = {
    "ppo": PPOModelAdapter,
    "parallel_ppo": ParallelPPOAdapter,
    "generic": GenericModelAdapter
}

def get_adapter(model_type, model_path, input_dim=2300, output_dim=100):
    """
    Get an appropriate adapter for the specified model type
    
    Args:
        model_type: String identifier for the model type
        model_path: Path to the model file
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model
        
    Returns:
        adapter: Instance of a ModelAdapter subclass
    """
    if model_type not in MODEL_ADAPTERS:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_ADAPTERS.keys())}")
    
    adapter_class = MODEL_ADAPTERS[model_type]
    return adapter_class(model_path, input_dim, output_dim) 