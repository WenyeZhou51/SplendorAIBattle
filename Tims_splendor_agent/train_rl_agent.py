import os
import argparse
from rl_agent import train, evaluate

def main():
    """Main function to parse arguments and train/evaluate the agent"""
    parser = argparse.ArgumentParser(description='Train or evaluate an RL agent for Splendor')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'both'],
                        help='Mode to run: train, evaluate, or both')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train/evaluate')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save model every N episodes during training')
    parser.add_argument('--model_path', type=str, default='models/splendor_agent.pt',
                        help='Path to save/load the model')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed training information including state verification errors')
    parser.add_argument('--track_metrics', action='store_true',
                        help='Track and save win rates, episode lengths and other metrics for analysis')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    if args.mode in ['train', 'both']:
        print(f"Training agent for {args.episodes} episodes...")
        train(num_episodes=args.episodes, save_every=args.save_every, 
              model_path=args.model_path, verbose=args.verbose, 
              track_metrics=args.track_metrics)
    
    if args.mode in ['evaluate', 'both']:
        print(f"Evaluating agent for {min(args.episodes, 100)} episodes...")
        evaluate(model_path=args.model_path, num_episodes=min(args.episodes, 100))

if __name__ == "__main__":
    main() 