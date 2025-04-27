#!/usr/bin/env python3
"""
Script to train a Splendor PPO agent using parallel processing.
Offers significant speedup over the sequential version by utilizing multiple CPU cores.
"""

from parallel_rl_agent import train_parallel, evaluate
import argparse
import time
import os

if __name__ == "__main__":
    # Default path to Splendor/models directory (one level up from script location)
    default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(default_model_dir, exist_ok=True)
    default_model_path = os.path.join(default_model_dir, "parallel_splendor_agent.pt")
    
    parser = argparse.ArgumentParser(description='Train or evaluate a parallel PPO agent for Splendor')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'both'],
                        help='Operation mode: train, evaluate, or both')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train or evaluate')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0 = use all CPU cores)')
    parser.add_argument('--update_interval', type=int, default=4,
                        help='Update policy every N episodes per worker')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save frequency during training (total episodes)')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help='Path to save/load the model')
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    start_time = time.time()
    
    if args.mode in ['train', 'both']:
        print(f"Starting parallel training for {args.episodes} episodes...")
        print(f"Model will be saved to: {args.model_path}")
        train_parallel(
            num_episodes=args.episodes,
            num_workers=args.workers,
            update_interval=args.update_interval,
            save_every=args.save_every,
            model_path=args.model_path
        )
    
    if args.mode in ['evaluate', 'both']:
        print(f"Evaluating model from {args.model_path}...")
        evaluate(model_path=args.model_path, num_episodes=100)
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    if args.mode == 'train':
        print("\nTo evaluate this model later, run:")
        print(f"python train_parallel.py --mode evaluate --model_path {args.model_path}") 