import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from parallel_rl_agent import ParallelPPOAgent

def visualize_model_weights(model_path="models/card_specialized_splendor_agent.pt", output_dir="models/visualizations"):
    """
    Create and save visualizations of model weights
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    agent = ParallelPPOAgent(num_workers=1)
    agent.load_model(model_path)
    
    # Visualize card encoder weights
    print("Visualizing card encoder weights...")
    card_weights = agent.policy.card_encoder[0].weight.data.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(card_weights, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f'Card Encoder Weights ({card_weights.shape[0]}x{card_weights.shape[1]})')
    plt.xlabel('Input Features')
    plt.ylabel('Neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'card_encoder_weights.png'))
    plt.close()
    
    # Visualize main encoder weights
    print("Visualizing main encoder weights...")
    main_weights = agent.policy.main_encoder[0].weight.data.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(main_weights, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f'Main Encoder Weights ({main_weights.shape[0]}x{main_weights.shape[1]})')
    plt.xlabel('Input Features')
    plt.ylabel('Neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_encoder_weights.png'))
    plt.close()
    
    # Visualize actor (policy) weights
    print("Visualizing actor weights...")
    actor_weights = agent.policy.actor.weight.data.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(actor_weights, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f'Actor Weights ({actor_weights.shape[0]}x{actor_weights.shape[1]})')
    plt.xlabel('Input Features')
    plt.ylabel('Actions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actor_weights.png'))
    plt.close()
    
    # Visualize critic (value) weights
    print("Visualizing critic weights...")
    critic_weights = agent.policy.critic.weight.data.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(critic_weights, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f'Critic Weights ({critic_weights.shape[0]}x{critic_weights.shape[1]})')
    plt.xlabel('Input Features')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'critic_weights.png'))
    plt.close()
    
    # Visualize weight distributions
    print("Visualizing weight distributions...")
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each component
    plt.subplot(2, 2, 1)
    plt.hist(card_weights.flatten(), bins=50, alpha=0.7)
    plt.title('Card Encoder Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.hist(main_weights.flatten(), bins=50, alpha=0.7)
    plt.title('Main Encoder Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.hist(actor_weights.flatten(), bins=50, alpha=0.7)
    plt.title('Actor Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 4)
    plt.hist(critic_weights.flatten(), bins=50, alpha=0.7)
    plt.title('Critic Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_distributions.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_model_weights() 