# Enhanced Splendor AI

This repository contains improved reinforcement learning agents for playing the board game Splendor, including an enhanced reward function and league-based self-play training.

## Key Components

### Enhanced Reward Function

The `EnhancedSplendorEnv` class improves upon the basic Splendor environment by providing strategic rewards that better align with winning game strategies:

1. **Card Color Concentration** - Rewards building sets of cards of the same color
2. **Noble Progress** - Provides intermediate rewards for approaching noble requirements
3. **Efficiency Rewards** - Rewards efficient gem-to-card conversions
4. **Balanced Terminal Rewards** - Scales down terminal rewards to better balance with strategic rewards

### League-Based Self-Play Training

The `LeagueTrainer` class implements a sophisticated self-play mechanism with:

1. **Opponent Variety** - Creates a league of different opponent policies
2. **ELO Rating System** - Tracks relative skill levels between agent versions
3. **Model Versioning** - Saves intermediate policies for future training
4. **Fixed Opponents** - Includes rule-based opponents (random, heuristic) for baseline performance

## How to Use

### Training with Enhanced Rewards

```bash
# Train a standard agent with the enhanced reward function
python train.py --env enhanced --episodes 5000 --workers 4
```

### League-Based Training

```bash
# Train using the league-based approach
python league_training.py --episodes 5000 --workers 4 --save_frequency 500
```

### Evaluating Agents

```bash
# Evaluate the league-trained agent
python evaluate.py --model_path models/splendor_league_agent.pt --episodes 100
```

## Implementation Details

### Enhanced Reward Function

The enhanced reward function (in `enhanced_splendor_env.py`) provides these improvements:

1. **Base Score Rewards**: Increased to 0.5 per point
2. **Color Concentration**: Up to 0.3 reward for building card sets
3. **Noble Progress**: 0.3 reward per step closer to a noble
4. **Card Efficiency**: 0.4-0.6 reward based on gem efficiency 
5. **Strategic Reservations**: 0.2 reward for reserving with gold gain
6. **Reduced Penalties**: -0.3 for non-productive turns (down from -1.0)
7. **Balanced Terminal Rewards**: 3.0 for winning/losing (down from 10.0)

### League-Based Training

The league system (in `league_training.py`) offers:

1. **Opponent Selection Strategies**:
   - Random: Select any opponent
   - Best: Select highest-rated opponent
   - Similar: Select opponent with similar skill
   - Mix: Combination of strategies

2. **Rating System**:
   - ELO-based rating updates
   - History tracking of all matches
   - Performance visualization

3. **Fixed Opponents**:
   - Random agent: Selects valid moves randomly
   - Heuristic agent: Uses basic Splendor strategies

## Results

The enhanced agents demonstrate significant improvements over the baseline:

- Better strategic play
- Higher win rates against rule-based opponents
- More consistent learning progress
- Reduced reliance on randomness

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- Numpy
- Matplotlib
- tqdm

## Overview

The agent uses Proximal Policy Optimization (PPO) algorithm to learn the optimal policy for playing Splendor through self-play. The agent is designed to:

1. Learn through self-play in a Gymnasium environment
2. Be integrated with the existing Splendor implementation as a bot
3. Compete against other bots in bot-vs-bot mode

## Files

- `splendor_env.py`: Gymnasium environment for Splendor
- `ppo_agent.py`: Implementation of the PPO agent
- `train_ppo.py`: Script for training the agent through self-play
- `ppo_bot.py`: Integration with the Splendor game as a bot
- `models/`: Directory containing trained model checkpoints

## Reinforcement Learning Agent

This project includes an implementation of a reinforcement learning agent using Proximal Policy Optimization (PPO) algorithm with the Gymnasium framework.

### Model Architecture

The RL agent uses an Actor-Critic architecture:
- Input dimension: 2300 (matches the game state vector size)
- Output dimension: 100 (matches the maximum number of possible actions)
- The actor network outputs action probabilities
- The critic network estimates state values

### Training the Agent

To train the agent, run:

```bash
python train_rl_agent.py --mode train --episodes 1000 --save_every 100
```

This will:
- Train the agent for 1000 episodes
- Save the model every 100 episodes
- Save a final model to `models/splendor_agent.pt`
- Generate learning curve plots in the `models/` directory

You can customize training with the following arguments:
- `--mode`: Choose 'train', 'evaluate', or 'both'
- `--episodes`: Number of episodes to train or evaluate
- `--save_every`: Save frequency during training
- `--model_path`: Custom path to save/load the model

### Parallel Training (Faster)

For much faster training using multiple CPU cores, use the parallel implementation:

```bash
python train_parallel.py --mode train --episodes 5000 --workers 0 --save_every 500
```

This will:
- Train the agent for 5000 episodes
- Use all available CPU cores (setting `--workers 0`)
- Save the model every 500 episodes
- Save the final model to `models/parallel_splendor_agent.pt`
- Generate learning curves with the parallel training progress

Parallel training provides significant speedup:
- Training that would take hours with sequential processing can be completed in minutes
- The implementation uses PyTorch's multiprocessing capabilities
- Memory is shared between processes for efficient training

You can customize parallel training with these arguments:
- `--workers`: Number of parallel workers (0 = use all CPU cores)
- `--update_interval`: Number of episodes per worker before policy update
- Other options are same as sequential training

### Evaluating the Agent

To evaluate a trained agent, run:

```bash
python train_rl_agent.py --mode evaluate --model_path models/splendor_agent.pt
```

This will run the model against the environment and report performance metrics.

### Using the Trained Model

The trained model can be loaded and used as follows:

```python
from rl_agent import PPOAgent

# Load the agent
agent = PPOAgent()
agent.load_model('models/splendor_agent.pt')

# Get action from state
action, _ = agent.get_action(state, valid_moves_mask)
```

The model is saved as a `.pt` file compatible with PyTorch and meets the webgui model requirements:
- Input dimensions: 2300
- Output dimensions: 100
- File format: .pt (PyTorch save format)

## Usage

### Training the Agent

To train the agent from scratch:

```bash
python train_ppo.py
```

This will start the training process with default parameters. The agent will play against itself to learn optimal policies. Model checkpoints will be saved in the `models/` directory.

### Playing Against the Trained Agent

After training, you can use the agent as a bot in the existing Splendor implementation:

```bash
python ppo_bot.py
```

This will create a game between the PPO agent and a RandomAI bot.

### Integration with Bot vs Bot Mode

To use the trained agent in bot vs bot mode, you can import the `PPOBotAI` class from `ppo_bot.py`:

```python
from Tims_splendor_agent.ppo_bot import PPOBotAI
from lapidary.ais import GameManager
from lapidary.ais import RandomAI

# Create PPO bot with a trained model
ppo_bot = PPOBotAI(model_path='Tims_splendor_agent/models/ppo_checkpoint_final.pt')

# Play against RandomAI
game_manager = GameManager(players=2, ais=[ppo_bot, RandomAI()], end_score=15)
game_info = game_manager.run_game(verbose=True)
```

## Model Architecture

The PPO agent uses a neural network with:

- A shared feature extractor (2 hidden layers with 256 neurons each)
- A policy head that outputs action probabilities
- A value head that estimates the state value

## Training Parameters

The default training parameters are:

- Number of episodes: 10,000
- Learning rate: 0.0003
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- PPO clip ratio: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
- Hidden dimension: 256

These parameters can be adjusted in the `train_ppo.py` script.

## Performance

The agent improves over time through self-play. After training, it can make strategic decisions like:

- Prioritizing high-value cards
- Managing gem resources efficiently
- Planning several moves ahead

## Future Improvements

- Implement opponent modeling for better performance against different bot types
- Tune hyperparameters for faster learning
- Add support for 3-4 player games 