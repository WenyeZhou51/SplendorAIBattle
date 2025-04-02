import os
import numpy as np
import torch
from parallel_rl_agent import ParallelPPOAgent
from splendor_env import SplendorEnv

def test_model_operation():
    """
    Test basic model operations to verify the architecture works correctly.
    """
    print("Testing model architecture...")
    # Initialize environment and agent
    env = SplendorEnv(num_players=2)
    agent = ParallelPPOAgent(num_workers=1)
    
    # Reset environment
    state, info = env.reset()
    
    # Get valid moves mask
    valid_moves = list(env.valid_moves_mapping.keys())
    valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
    
    # Test action selection
    try:
        print("Testing action selection...")
        action, log_prob = agent.get_action(state, valid_moves_mask)
        print(f"Selected action: {action}, Log prob: {log_prob.item():.4f}")
        print("Action selection successful!")
    except Exception as e:
        print(f"Error in action selection: {e}")
        return False
    
    # Test taking an action in the environment
    try:
        print("\nTesting environment step...")
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Terminated: {terminated}")
        print("Environment step successful!")
    except Exception as e:
        print(f"Error in environment step: {e}")
        return False
    
    # Test memory storage
    try:
        print("\nTesting memory storage...")
        agent.remember(0, state, action, reward, terminated, log_prob.item(), valid_moves_mask)
        print("Memory storage successful!")
    except Exception as e:
        print(f"Error in memory storage: {e}")
        return False
    
    # Simulate a small batch for policy update
    try:
        print("\nTesting policy update with minimal batch...")
        # Add a few more experiences
        for _ in range(5):
            state = next_state
            valid_moves = list(env.valid_moves_mapping.keys())
            valid_moves_mask = [1 if i in valid_moves else 0 for i in range(100)]
            action, log_prob = agent.get_action(state, valid_moves_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(0, state, action, reward, terminated or truncated, log_prob.item(), valid_moves_mask)
            if terminated or truncated:
                state, info = env.reset()
        
        # Update policy
        agent.update()
        print("Policy update successful!")
    except Exception as e:
        print(f"Error in policy update: {e}")
        return False
    
    # Test model saving and loading
    try:
        print("\nTesting model saving...")
        test_path = "models/test_model.pt"
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        agent.save_model(test_path)
        print("Model saving successful!")
        
        print("\nTesting model loading...")
        new_agent = ParallelPPOAgent(num_workers=1)
        new_agent.load_model(test_path)
        print("Model loading successful!")
    except Exception as e:
        print(f"Error in model saving/loading: {e}")
        return False
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_model_operation()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILURE'}") 